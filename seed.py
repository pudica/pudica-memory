"""seed pudica-memory with test data."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import asyncio
import time
from unified_memory.main import UnifiedMemoryApp
from unified_memory.config import Config

MEMORIES = [
    "王哥气阴两虚体质，复合用药方案：生脉饮+玉屏风+参苓白术散+知柏地黄丸",
    "零刻SER9 Pro，Ryzen AI 9 HX 370，32GB板载不可升级，近3TB存储",
    "DeepSeek V4远程API为主力，GLM备用，本地Qwen离线备份",
    "五套记忆系统：MemPalace(ChromaDB)、Hindsight(演化)、TencentDB(Gateway)、pudica-Memory(综合)、Hermes memory",
    "中医学习：倪海厦全套课程，东直门看病，气阴两虚+脾湿+血虚风燥湿疹",
    "A股投资：腾讯行情API(qt.gtimg.cn)，PE/市值批量查询，每批最多50只",
    "Obsidian知识中枢E盘，08-资源库默认归档，06-玄学含易魂39部",
]

async def seed():
    c = Config.load()
    c.log_level = 'WARNING'
    app = UnifiedMemoryApp(c)
    await app.initialize()
    t0 = time.time()
    for m in MEMORIES:
        await app.pipeline.ingest(m, source='seed')
    await app.pipeline.flush()
    await asyncio.sleep(0.5)
    print(f"seed完成: {len(MEMORIES)}条, {time.time()-t0:.2f}s")
    await app.shutdown()

asyncio.run(seed())