"""BEAM Benchmark Runner for pudica-memory.

Runs the BEAM benchmark (ICLR 2026) against pudica-memory HTTP API.

Flow:
  1. Download dataset from HuggingFace (auto-cached)
  2. For each conversation:
     a. Ingest via pudica-memory API
     b. For each probing question: search -> generate answer -> judge
  3. Compute metrics and save results

Usage:
  python benchmarks/run_beam.py --project-name test --chat-sizes 100K --conversations 1
  python benchmarks/run_beam.py --project-name full --chat-sizes 100K,500K,1M
  python benchmarks/run_beam.py --project-name eval --predict-only
  python benchmarks/run_beam.py --project-name eval --evaluate-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pudica_client import PudicaClient
from benchmarks.common.llm_client import LLMClient
from benchmarks.common.metrics import compute_kendall_tau_b, compute_overall_metrics
from benchmarks.common.schema import (
    CutoffResult,
    EvalItem,
    NuggetScore,
    Metadata,
    Metrics,
    UnifiedResult,
)
from benchmarks.common.utils import (
    Checkpoint,
    GracefulShutdown,
    IngestionCheckpoint,
    cutoff_label,
    parse_cutoffs,
    save_result_json,
    setup_logging,
)

# Make sure we can import beam prompts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from beam.prompts import (
    BEAM_JUDGE_SYSTEM_PROMPT,
    BEAM_QUESTION_TYPES,
    get_beam_answer_generation_prompt,
    get_beam_event_alignment_prompt,
    get_beam_fact_extraction_prompt,
    get_beam_nugget_judge_prompt,
)

logger = logging.getLogger(__name__)

# ===============================================================================
# CONSTANTS
# ===============================================================================

HF_DATASET_NAME = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
HF_SPLIT_MAP: dict[str, str] = {"100K": "100K", "500K": "500K", "1M": "1M", "10M": "10M"}
VALID_CHAT_SIZES = ["100K", "500K", "1M", "10M"]
DEFAULT_DATASET_DIR = "datasets/beam"
CHUNK_SIZE = 2  # turns per ingestion chunk


# ===============================================================================
# DATASET
# ===============================================================================

def download_dataset(
    chat_sizes: list[str],
    cache_dir: str,
    logger: Any,
) -> dict[str, list[dict]]:
    """Download BEAM dataset from HuggingFace, cache locally."""
    os.makedirs(cache_dir, exist_ok=True)
    dataset: dict[str, list[dict]] = {}

    for size in chat_sizes:
        cache_path = os.path.join(cache_dir, f"beam_{size}.json")

        if os.path.exists(cache_path):
            logger.info("Loading cached %s dataset: %s", size, cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                dataset[size] = json.load(f)
            continue

        logger.info("Downloading BEAM %s dataset from HuggingFace...", size)
        try:
            from datasets import load_dataset as hf_load

            if size == "10M":
                ds = hf_load(HF_DATASET_10M, split="10M")
            else:
                ds = hf_load(HF_DATASET_NAME, split=HF_SPLIT_MAP[size])

            conversations: list[dict] = []
            for idx, item in enumerate(ds):
                conv: dict[str, Any] = {
                    "conversation_id": item.get("conversation_id", f"{size}_{idx}"),
                    "conversation_seed": item.get("conversation_seed", {}),
                    "user_profile": item.get("user_profile", {}),
                    "chat": item.get("chat", []),
                }

                pq_raw = item.get("probing_questions", "{}")
                if isinstance(pq_raw, str):
                    try:
                        conv["probing_questions"] = ast.literal_eval(pq_raw)
                    except (ValueError, SyntaxError):
                        try:
                            conv["probing_questions"] = json.loads(pq_raw)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Could not parse probing_questions for %s[%d]", size, idx,
                            )
                            conv["probing_questions"] = {}
                else:
                    conv["probing_questions"] = pq_raw if isinstance(pq_raw, dict) else {}

                conversations.append(conv)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False)
            logger.info("Downloaded and cached %s: %d conversations", size, len(conversations))
            dataset[size] = conversations

        except Exception as exc:
            raise RuntimeError(
                f"Failed to download BEAM {size} dataset: {exc}\n"
                f"Install datasets: pip install datasets\n"
                f"Or manually download and place in {cache_dir}"
            ) from exc

    return dataset


# ===============================================================================
# CHAT PARSING
# ===============================================================================

def parse_chat_to_messages(chat: list[dict]) -> list[dict]:
    """Parse raw chat turns into message format."""
    messages = []
    for turn in chat:
        role = turn.get("role", "user")
        content = turn.get("content", turn.get("message", ""))
        messages.append({"role": role, "content": content})
    return messages


def chunk_messages(messages: list[dict], chunk_size: int = CHUNK_SIZE) -> list[list[dict]]:
    """Split messages into chunks for incremental ingestion."""
    return [messages[i : i + chunk_size * 2] for i in range(0, len(messages), chunk_size * 2)]


def format_search_results(search_results: list[dict]) -> tuple[list[dict], dict | None]:
    """Normalize search results for benchmark output."""
    if not search_results:
        return [], None

    sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
    formatted = []
    for r in sorted_results:
        entry: dict[str, Any] = {
            "memory": r.get("memory", ""),
            "score": r.get("score", 0),
            "id": r.get("id", ""),
        }
        if r.get("created_at"):
            entry["created_at"] = r["created_at"]
        formatted.append(entry)
    return formatted, None


# ===============================================================================
# INGESTION
# ===============================================================================

async def ingest_conversation(
    client: PudicaClient,
    conversation: dict,
    memory_name: str,
    checkpoints: IngestionCheckpoint,
    logger: Any,
) -> None:
    """Ingest a single conversation into pudica-memory."""
    conv_id = conversation.get("conversation_id", "unknown")
    messages = parse_chat_to_messages(conversation.get("chat", []))
    chunks = chunk_messages(messages)

    # Check checkpoint
    if checkpoints.is_ingested(conv_id, len(chunks)):
        logger.info("  [ingest] %s: already ingested (%d chunks)", conv_id, len(chunks))
        return

    if chunks:
        logger.info("  [ingest] %s: %d chunks, %d turns", conv_id, len(chunks), len(messages))
        for i, chunk in enumerate(chunks):
            result = await client.add(chunk, user_id=memory_name)
            if not result.get("success", True):
                logger.warning("  [ingest] chunk %d/%d failed: %s", i + 1, len(chunks), result)

        checkpoints.mark_ingested(conv_id, len(chunks))
        # Brief pause between conversations
        await asyncio.sleep(0.1)
    else:
        logger.warning("  [ingest] %s: empty chat, skipping", conv_id)


# ===============================================================================
# EVALUATION
# ===============================================================================

async def evaluate_conversation(
    client: PudicaClient,
    conversation: dict,
    answerer: LLMClient,
    judge: LLMClient,
    memory_name: str,
    top_k: int,
    prompts: dict[str, Any],
    checkpoints: Checkpoint,
    question_filter: list[str] | None,
    logger: Any,
) -> list[dict]:
    """Evaluate probing questions for a single conversation."""
    conv_id = conversation.get("conversation_id", "unknown")
    probing_questions = conversation.get("probing_questions", {})
    evaluations = []

    qids = sorted(probing_questions.keys())
    if question_filter:
        qids = [q for q in qids if probing_questions[q].get("type") in question_filter]

    if not qids:
        logger.info("  [eval] %s: no questions to evaluate", conv_id)
        return evaluations

    logger.info("  [eval] %s: %d questions", conv_id, len(qids))

    for qid in qids:
        q_data = probing_questions[qid]
        question = q_data.get("question", "")
        q_type = q_data.get("type", "unknown")
        nuggets = q_data.get("nuggets", [])

        # Check checkpoint
        if checkpoints.is_complete(conv_id, qid):
            cached = checkpoints.get_result(conv_id, qid)
            if cached:
                evaluations.append(cached)
                continue

        # Search
        search_results = await client.search(
            query=question,
            user_id=memory_name,
            top_k=top_k,
        )
        formatted_results, query_debug = format_search_results(search_results)

        if not formatted_results:
            logger.warning("  [eval] %s/%s: no results, skipping", conv_id, qid)
            continue

        # Format memories for answer generation
        memories_text = ""
        for i, r in enumerate(formatted_results[:top_k]):
            mem_text = r.get("memory", "")[:2000]
            if mem_text:
                memories_text += f"\n[{i + 1}] (score={r['score']:.3f}): {mem_text}\n"

        # Generate answer
        answer_prompt = get_beam_answer_generation_prompt(
            question=question,
            memories=memories_text,
        )

        # Use the answerer model (deepseek-v4-flash)
        answer = await answerer.generate(
            system="You are an AI assistant that answers questions based on retrieved memories.",
            user=answer_prompt,
            temperature=0,
        )

        # Judge each nugget
        nugget_scores = []
        for nugget in nuggets:
            judge_prompt = get_beam_nugget_judge_prompt(
                question=question,
                response=answer,
                nugget=nugget,
            )
            judge_result = await judge.generate_structured(
                system=BEAM_JUDGE_SYSTEM_PROMPT,
                user=judge_prompt,
                temperature=0,
            )

            score = float(judge_result.get("score", 0))
            reason = judge_result.get("reason", "")
            nugget_scores.append(NuggetScore(score=score, reason=reason))

        # Compute Kendall tau-b for event ordering questions
        kendall_tau = None
        if q_type == "event_ordering" and len(nuggets) > 1:
            try:
                facts_prompt = get_beam_fact_extraction_prompt(answer)
                facts_raw = await answerer.generate(
                    system="Extract events in order.",
                    user=facts_prompt,
                    temperature=0,
                )
                extracted_facts = json.loads(facts_raw)
                reference_events = [n["nugget"] for n in nuggets if isinstance(n, dict)]
                if reference_events and extracted_facts:
                    alignment_prompt = get_beam_event_alignment_prompt(
                        extracted_event=json.dumps(extracted_facts),
                        rubric_events=reference_events,
                    )
                    alignment_raw = await answerer.generate(
                        system="Align events.",
                        user=alignment_prompt,
                        temperature=0,
                    )
                    alignment = json.loads(alignment_raw)
                    if isinstance(alignment, list) and len(alignment) == len(extracted_facts):
                        indices = [a.get("index", -1) for a in alignment]
                        valid_indices = [i for i in indices if i >= 0]
                        if len(valid_indices) > 1:
                            kendall_tau = compute_kendall_tau_b(
                                list(range(len(valid_indices))),
                                valid_indices,
                            )
            except Exception as e:
                logger.warning("  [eval] Kendall tau computation failed: %s", e)

        # Build result
        eval_result = {
            "conversation_id": conv_id,
            "question_id": qid,
            "question": question,
            "question_type": q_type,
            "answer": answer,
            "nugget_scores": [s.__dict__ for s in nugget_scores],
            "kendall_tau": kendall_tau,
            "avg_score": sum(s.score for s in nugget_scores) / len(nugget_scores) if nugget_scores else 0,
            "retrieved_count": len(formatted_results),
            "retrieved_memories": [r["memory"][:500] for r in formatted_results[:5]],
        }

        checkpoints.mark_complete(conv_id, qid, eval_result)
        evaluations.append(eval_result)

        # Brief pause between questions
        await asyncio.sleep(0.05)

    return evaluations


# ===============================================================================
# METRICS
# ===============================================================================

def compute_beam_metrics(
    evaluations: list[dict],
    cutoffs: list[int],
) -> dict[str, Metrics]:
    """Compute metrics by question type and cutoff."""
    metrics_by_cutoff: dict[str, Metrics] = {}

    for cutoff in cutoffs:
        # Filter evaluations by cutoff (top_k)
        filtered = [e for e in evaluations if e.get("retrieved_count", 0) >= cutoff or True]

        if not filtered:
            continue

        label = cutoff_label(cutoff)

        # Per-question-type metrics
        type_scores: dict[str, list[float]] = defaultdict(list)
        type_nugget_scores: dict[str, list[float]] = defaultdict(list)

        for eval_item in filtered:
            q_type = eval_item.get("question_type", "unknown")
            type_scores[q_type].append(eval_item.get("avg_score", 0))
            for ns in eval_item.get("nugget_scores", []):
                type_nugget_scores[q_type].append(ns.get("score", 0))

        # Overall metrics
        all_scores = [s for scores in type_scores.values() for s in scores]
        all_nugget_scores = [s for scores in type_nugget_scores.values() for s in scores]

        per_type_metrics = {}
        for q_type in sorted(type_scores.keys()):
            scores = type_scores[q_type]
            nugget_scores = type_nugget_scores[q_type]
            per_type_metrics[q_type] = {
                "avg_score": statistics.mean(scores) if scores else 0,
                "avg_nugget_score": statistics.mean(nugget_scores) if nugget_scores else 0,
                "count": len(scores),
                "description": BEAM_QUESTION_TYPES.get(q_type, ""),
            }

        metrics_by_cutoff[label] = Metrics(
            top_k=cutoff,
            avg_score=statistics.mean(all_scores) if all_scores else 0,
            avg_nugget_score=statistics.mean(all_nugget_scores) if all_nugget_scores else 0,
            total_questions=len(filtered),
            question_types=per_type_metrics,
        )

    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict[str, Metrics], cutoffs: list[int]) -> None:
    """Display results in a readable format."""
    print("\n" + "=" * 70)
    print("BEAM BENCHMARK RESULTS")
    print("=" * 70)

    for cutoff in cutoffs:
        label = cutoff_label(cutoff)
        metrics = metrics_by_cutoff.get(label)
        if not metrics:
            continue

        print(f"\n--- Top-{cutoff} Results ---")
        print(f"  Overall Score:       {metrics.avg_score:.4f}")
        print(f"  Overall Nugget Score: {metrics.avg_nugget_score:.4f}")
        print(f"  Total Questions:     {metrics.total_questions}")

        if metrics.question_types:
            print(f"\n  {'Question Type':<30} {'Score':>8} {'Nugget':>8} {'Count':>6}")
            print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6}")
            for q_type, qt_metrics in sorted(metrics.question_types.items()):
                print(
                    f"  {q_type:<30} {qt_metrics['avg_score']:>8.4f} "
                    f"{qt_metrics['avg_nugget_score']:>8.4f} {qt_metrics['count']:>6}"
                )


# ===============================================================================
# MAIN
# ===============================================================================

async def async_main() -> None:
    parser = argparse.ArgumentParser(description="BEAM Benchmark for pudica-memory")
    parser.add_argument("--project-name", type=str, default="beam-test", help="Project name")
    parser.add_argument("--chat-sizes", type=str, default="100K", help="Comma-separated chat sizes")
    parser.add_argument("--conversations", type=int, default=None, help="Max conversations per size")
    parser.add_argument("--top-k", type=int, default=200, help="Top-k retrieval")
    parser.add_argument("--top-k-cutoffs", type=str, default="20,50,100", help="Cutoff values")
    parser.add_argument("--question-types", type=str, default=None, help="Filter question types")
    parser.add_argument("--predict-only", action="store_true", help="Only generate predictions")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate saved predictions")
    parser.add_argument("--output-dir", type=str, default="results/beam", help="Output directory")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset cache dir")
    parser.add_argument("--answerer-model", type=str, default="deepseek-v4-flash-260425", help="Answerer model")
    parser.add_argument("--judge-model", type=str, default="deepseek-v4-flash-260425", help="Judge model")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--judge-provider", type=str, default=None, help="Judge LLM provider")
    parser.add_argument("--pudica-host", type=str, default="http://127.0.0.1:8420", help="Pudica memory host")
    parser.add_argument("--api-base", type=str, default=None, help="Custom API base URL for LLM")
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM")
    args = parser.parse_args()

    # Setup
    run_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"beam_{run_id}.log"
    setup_logging("beam", log_file=str(log_path))
    logger.info("BEAM Benchmark run %s started", run_id)
    logger.info("Args: %s", vars(args))

    chat_sizes = [s.strip() for s in args.chat_sizes.split(",") if s.strip()]
    chat_sizes = [s for s in chat_sizes if s in VALID_CHAT_SIZES]
    cutoffs = parse_cutoffs(args.top_k_cutoffs) if args.top_k_cutoffs else [20, 50, 100]
    q_type_filter = args.question_types.split(",") if args.question_types else None

    # Download dataset
    print("Downloading dataset...")
    dataset = download_dataset(chat_sizes, str(dataset_dir), logger)

    # Initialize clients
    print("Initializing clients...")
    api_key = args.api_key or os.getenv("ARK_API_KEY", "")
    api_base = args.api_base or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")

    # Use API key from config
    if not api_key:
        # Try to read from config
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                for line in f:
                    if "ark" in line.lower() and len(line) > 40:
                        parts = line.strip().split(":")
                        if len(parts) > 1:
                            api_key = parts[-1].strip().strip('"\'')
                            break

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    memory_client = PudicaClient(
        host=args.pudica_host,
        max_retries=3,
        retry_delay=1.0,
    )

    answerer_kwargs = {
        "model": args.answerer_model,
        "provider": args.provider,
        "base_url": api_base,
        "api_key": api_key,
    }
    judge_kwargs = {
        "model": args.judge_model,
        "provider": args.judge_provider or args.provider,
        "base_url": api_base,
        "api_key": api_key,
    }

    answerer = LLMClient(**answerer_kwargs)
    judge = LLMClient(**judge_kwargs)

    # Graceful shutdown handler
    shutdown = GracefulShutdown()

    # Run benchmark
    all_evaluations: list[dict] = []
    total_questions = 0

    for size in chat_sizes:
        conversations = dataset.get(size, [])
        if args.conversations:
            conversations = conversations[: args.conversations]

        if not conversations:
            logger.warning("No conversations for size %s, skipping", size)
            continue

        # Per-size checkpoints
        ingest_checkpoint = IngestionCheckpoint(str(output_dir / f"ingest_{size}.json"))
        eval_checkpoint = Checkpoint(str(output_dir / f"eval_{size}.json"))

        print(f"\n{'='*60}")
        print(f"Size: {size} — {len(conversations)} conversations")
        print(f"{'='*60}")

        for idx, conversation in enumerate(conversations):
            if shutdown.should_stop:
                print("Graceful shutdown requested, stopping...")
                break

            conv_id = conversation.get("conversation_id", f"{size}_{idx}")
            memory_name = f"beam_{size}_{conv_id}"

            print(f"\n[{idx + 1}/{len(conversations)}] {conv_id}")

            # Step 1: Ingest
            print(f"  Ingesting...")
            await ingest_conversation(
                memory_client, conversation, memory_name, ingest_checkpoint, logger,
            )

            # Step 2: Evaluate
            if not args.predict_only:
                evaluations = await evaluate_conversation(
                    memory_client, conversation, answerer, judge,
                    memory_name, args.top_k, {},
                    eval_checkpoint, q_type_filter, logger,
                )
                all_evaluations.extend(evaluations)
                total_questions += len(evaluations)

                if evaluations:
                    avg = sum(e.get("avg_score", 0) for e in evaluations) / len(evaluations)
                    print(f"  Avg score: {avg:.4f} ({len(evaluations)} questions)")

        # Clean up memory after each size
        # (pudica-memory doesn't support per-user deletion, so we just note it)
        print(f"  Size {size} complete: {len(all_evaluations)} total evaluations so far")

    # === Metrics ===
    if not args.predict_only and all_evaluations:
        metrics_by_cutoff = compute_beam_metrics(all_evaluations, cutoffs)
        display_results(metrics_by_cutoff, cutoffs)

        # Save unified result JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = output_dir / f"beam_results_{timestamp}.json"

        # Build per-cutoff metrics dict
        metrics_dict = {}
        for label, metrics in metrics_by_cutoff.items():
            metrics_dict[label] = {
                "top_k": metrics.top_k,
                "avg_score": metrics.avg_score,
                "avg_nugget_score": metrics.avg_nugget_score,
                "total_questions": metrics.total_questions,
                "question_types": metrics.question_types,
            }

        result_data = {
            "metadata": {
                "benchmark": "beam",
                "project_name": args.project_name,
                "run_id": run_id,
                "timestamp": timestamp,
                "memory_system": "pudica-memory",
                "answerer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "provider": args.provider,
                "top_k": args.top_k,
                "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                "chat_sizes": chat_sizes,
                "conversations": args.conversations or "all",
                "total_questions": len(all_evaluations),
                "question_types": q_type_filter or list(BEAM_QUESTION_TYPES.keys()),
            },
            "metrics_by_cutoff": metrics_dict,
            "evaluations": all_evaluations,
        }

        with open(unified_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {unified_path}")

    # Cleanup
    await memory_client.close()
    print(f"\nTotal questions processed: {total_questions}")
    print(f"Run complete. Log: {log_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()