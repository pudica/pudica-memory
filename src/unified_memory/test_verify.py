# 4c. 重排器（Hindsight: Cross-Encoder）
        if self.config.reranker.enabled:
            cross_encoder_model = getattr(self.config.reranker, "cross_encoder_model", None)
            self.reranker = Reranker(
                strategy=self.config.reranker.strategy,
                top_n=self.config.reranker.top_n,
                final_k=self.config.reranker.final_k,
                llm=self.llm if self.config.reranker.strategy == "llm" else None,
                max_concurrent=self.config.reranker.max_concurrent,
                cross_encoder_model=cross_encoder_model,
            )
            logger.info("  重排器: strategy=%s, top_n=%d, final_k=%d",
                         self.config.reranker.strategy, self.config.reranker.top_n,
                         self.config.reranker.final_k)

        # 5. Pipeline 管线
        dedup = L0Dedup(maxsize=self.config.pipeline.l0_maxsize)