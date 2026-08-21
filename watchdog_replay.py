"""ark 限流恢复后自动完成 L1 回放的后台守护。

循环探测 ark LLM 端点连通性，恢复后：
  1. 备份 unified_memory.db + chroma
  2. 正式回放（--no-kill，含清 KG 垃圾）
  3. 重启 Hermes 网关 + 恢复计划任务自启
全程日志写 watchdog.log。
"""
import subprocess, os, time, shutil, datetime

PROJ = r'D:\pudica\pudica-memory\pudica-memory-v3.2.2'
VENV = os.path.join(PROJ, 'venv', 'Scripts', 'python.exe')
WATCHLOG = os.path.join(PROJ, 'watchdog.log')
PROBE_INTERVAL = 180   # 每 3 分钟探测一次
MAX_PROBES = 40        # 最多约 2 小时

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(WATCHLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def ark_ok():
    test = (
        'import sys, asyncio\n'
        'sys.path.insert(0, r"' + PROJ + r'\\src")\n'
        'from unified_memory.config import Config\n'
        'from unified_memory.main import LLMClient\n'
        'c=Config.load(); llm=LLMClient(c.llm)\n'
        'async def t():\n'
        '    r=await asyncio.wait_for(llm.call(\'{"ok":1}\', response_format={"type":"json_object"}), timeout=30)\n'
        '    return r\n'
        'print("OK" if asyncio.run(t()) else "NO")\n'
    )
    try:
        out = subprocess.run([VENV, '-c', test], cwd=PROJ,
                             capture_output=True, text=True, timeout=70)
        return 'OK' in out.stdout
    except Exception as e:
        log(f'ark 探测异常: {e}')
        return False

def backup():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = os.path.join(PROJ, 'data', 'backup_' + ts)
    os.makedirs(bak, exist_ok=True)
    for name in ['unified_memory.db']:
        src = os.path.join(PROJ, 'data', name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(bak, name))
    chroma = os.path.join(PROJ, 'data', 'chroma')
    if os.path.isdir(chroma):
        shutil.copytree(chroma, os.path.join(bak, 'chroma'), dirs_exist_ok=True)
    log(f'已备份到 {bak}')
    return bak

def run_replay(extra_args, logpath, timeout):
    cmd = [VENV, 'scripts/l1_llm_replay.py'] + extra_args
    with open(logpath, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=PROJ, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode

def fail_rate(logpath):
    try:
        data = open(logpath, 'rb').read()
        succ = data.count('→ 实体'.encode('utf-8'))
        fail = data.count('LLM 失败'.encode('utf-8'))
        tot = succ + fail
        return (fail / tot) if tot else 1.0
    except Exception:
        return 1.0

def restart_gateway():
    log('重启 Hermes 网关...')
    subprocess.run('hermes gateway restart', shell=True, timeout=90)
    time.sleep(15)
    try:
        st = subprocess.run('hermes gateway status', shell=True,
                            capture_output=True, text=True, timeout=30).stdout
        log('gateway status: ' + ('running' if 'running' in st.lower() else st[:200]))
    except Exception as e:
        log(f'status 检查异常: {e}')
    try:
        subprocess.run('powershell -Command "Enable-ScheduledTask -TaskName Hermes_Gateway"',
                       shell=True, timeout=30)
        log('已重新启用 Hermes_Gateway 计划任务（恢复自启能力）')
    except Exception as e:
        log(f'启用计划任务异常: {e}')

def main():
    log('=== watchdog 启动，开始探测 ark 限流恢复（间隔 %ds，最多 %d 次）===' % (PROBE_INTERVAL, MAX_PROBES))
    for i in range(MAX_PROBES):
        log(f'探测 #{i+1}/{MAX_PROBES}')
        if ark_ok():
            log('ark 已恢复，备份并正式回放（--no-kill，含清 KG）')
            backup()
            rc = run_replay(['--no-kill'], os.path.join(PROJ, 'replay_final.log'), timeout=7200)
            log(f'正式回放返回码: {rc}')
            fr = fail_rate(os.path.join(PROJ, 'replay_final.log'))
            log(f'正式回放失败率(估算): {fr:.1%}')
            if rc == 0:
                if fr > 0.2:
                    log('警告：失败率偏高，但已执行完成（已备份，可回滚）。建议人工核查 replay_final.log')
                restart_gateway()
                log('=== 全部完成：回放 + 清KG + 重启网关 ===')
                return
            else:
                log('正式回放异常退出，未重启网关，保留现状。请人工核查 replay_final.log')
                return
        time.sleep(PROBE_INTERVAL)
    log('=== 达到最大探测次数（约 2 小时），ark 仍未恢复。请人工处理。===')

if __name__ == '__main__':
    main()
