"""Unified Memory — 包入口。

v3.4.0 新增：
- L3 用户画像层（Persona Distiller）
- 记忆过期 TTL
- KG 代码符号索引
- Reflect disposition 推断
- 嵌入式运行模式（import + create_embedded_app）
- 多 agent 隔离（agent_id 列）
"""

__version__ = "3.4.0"

from unified_memory.main import UnifiedMemoryApp, create_embedded_app

__all__ = ["UnifiedMemoryApp", "create_embedded_app"]