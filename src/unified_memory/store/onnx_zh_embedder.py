"""bge-small-zh-v1.5 ONNX 中文嵌入器（无需 torch，纯 onnxruntime）。

模型来源：ModelScope Xenova/bge-small-zh-v1.5（transformers.js ONNX 转换版）
本地目录：<project>/models/bge-small-zh-v1.5/
  - model_fp16.onnx   fp16 BERT 模型（47MB，CPU 可跑）
  - tokenizer.json    快速分词器
  - config.json / tokenizer_config.json / special_tokens_map.json / vocab.txt

用法（作为 chromadb EmbeddingFunction）：
    from onnx_zh_embedder import BgeZhOnnxEmbeddingFunction
    ef = BgeZhOnnxEmbeddingFunction()
    vecs = ef(["你好", "世界"])   # -> list[list[float]]，512 维，L2 归一化
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "bge-small-zh-v1.5")
MODEL_FILE_CANDIDATES = ["model.onnx", "model_quantized.onnx", "model_fp16.onnx"]
MAX_SEQ_LEN = 512
DIM = 512

_EF_CACHE: dict = {}


def _load_ort_session(model_path: str):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        providers.insert(0, "DmlExecutionProvider")
    sess = ort.InferenceSession(
        model_path,
        providers=providers,
        sess_options=ort.SessionOptions(),
    )
    return sess


def _load_tokenizer(model_dir: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    if tok.token_to_id("[PAD]") is not None:
        tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    else:
        tok.enable_padding()
    return tok


def _model_path(model_dir: str) -> Optional[str]:
    for name in MODEL_FILE_CANDIDATES:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    return None


def get_bge_onnx_embedding_function():
    """获取缓存的 bge-small-zh 中文 ONNX 嵌入器（全局只加载一次）。"""
    cached = _EF_CACHE.get("default")
    if cached is not None:
        return cached
    if not os.path.isdir(MODEL_DIR):
        logger.warning("bge ONNX 模型目录不存在: %s", MODEL_DIR)
        return None
    path = _model_path(MODEL_DIR)
    if path is None:
        logger.warning("bge ONNX 模型文件缺失: %s", MODEL_DIR)
        return None
    try:
        sess = _load_ort_session(path)
        tok = _load_tokenizer(MODEL_DIR)
    except Exception as e:  # noqa: BLE001
        logger.warning("bge ONNX 嵌入器加载失败: %s", e)
        return None
    ef = _BgeZhOnnxEmbeddingFunction(sess, tok)
    _EF_CACHE["default"] = ef
    logger.info("bge-small-zh ONNX 嵌入器已加载: %s (dim=%d)", os.path.basename(path), DIM)
    return ef


class _BgeZhOnnxEmbeddingFunction:
    """chromadb EmbeddingFunction 兼容封装。"""

    def __init__(self, session, tokenizer):
        self._session = session
        self._tokenizer = tokenizer
        # 自动探测 ONNX 图输入/输出名，避免硬编码
        self._input_names = [i.name for i in session.get_inputs()]
        self._output_names = [o.name for o in session.get_outputs()]
        logger.debug("ONNX inputs=%s outputs=%s", self._input_names, self._output_names)

    def __call__(self, input):
        return self._embed_documents(input)

    def name(self) -> str:
        # chromadb 校验用：返回 "default" 表示跟随集合默认嵌入函数，避免冲突报错
        return "default"

    def _embed_documents(self, texts):
        import numpy as np

        encodings = self._tokenizer.encode_batch(list(texts))
        max_len = max(len(e.ids) for e in encodings) or 1

        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        token_type_ids = np.zeros_like(input_ids)
        for i, e in enumerate(encodings):
            input_ids[i, : len(e.ids)] = e.ids
            attention_mask[i, : len(e.ids)] = e.attention_mask
            token_type_ids[i, : len(e.ids)] = e.type_ids

        feeds = {"input_ids": input_ids}
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = token_type_ids

        outs = self._session.run(self._output_names, feeds)
        # 取 last_hidden_state（第一个输出），CLS 池化 + L2 归一化
        hidden = outs[0]
        cls_vec = hidden[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(cls_vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cls_vec = cls_vec / norms
        return cls_vec.tolist()

    # chromadb 需要的别名（新版用关键字 input= 调用）
    def embed_documents(self, input):
        return self._embed_documents(input)

    def embed_query(self, input):
        # chromadb 协议：embed_query 需返回向量列表（与输入一一对应）
        if isinstance(input, str):
            return self._embed_documents([input])
        return self._embed_documents(input)


def _test():
    logging.basicConfig(level=logging.INFO)
    ef = get_bge_onnx_embedding_function()
    assert ef is not None, "嵌入器加载失败"
    vecs = ef(["用户喜欢使用Python进行开发", "用户正在学习pudica-memory的使用方法"])
    print("向量维度:", len(vecs[0]))
    import numpy as np

    a, b = np.array(vecs[0]), np.array(vecs[1])
    print("余弦相似度:", float(np.dot(a, b)))


if __name__ == "__main__":
    _test()
