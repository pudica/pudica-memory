@echo off
chcp 65001 >nul
title pudica-Memory 安装脚本

echo ========================================
echo   pudica-Memory 安装脚本
echo ========================================
echo.

:: 获取脚本所在目录
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

echo 项目目录: %PROJECT_DIR%

:: 查找 Python
set "PYTHON="
for %%p in (python3.11 python3 python) do (
    where %%p >nul 2>&1 && set "PYTHON=%%p" && goto :found
)

:: 如果没找到，尝试 ji 目录
if exist "C:\Users\Administrator\Documents\Codex\2026-08-03\ji\work\python311\python.exe" (
    set "PYTHON=C:\Users\Administrator\Documents\Codex\2026-08-03\ji\work\python311\python.exe"
    goto :found
)

echo [错误] 找不到 Python 3.11，请先安装 Python 3.11
pause
exit /b 1

:found
echo Python: %PYTHON%
"%PYTHON%" --version

:: 安装依赖
echo.
echo 安装 Python 依赖...
"%PYTHON%" -m pip install -r "%PROJECT_DIR%\requirements.txt" -q
if %ERRORLEVEL% neq 0 (
    echo [警告] 部分依赖安装失败
)

:: 创建 MCP 配置文件
echo.
echo 创建 MCP 配置...
set "MCP_FILE=%PROJECT_DIR%\.mcp.json"
(
echo {
echo   "mcpServers": {
echo     "pudica-memory": {
echo       "command": "%PYTHON:\=/%",
echo       "args": [
echo         "%PROJECT_DIR:\=/%/run_mcp.py"
echo       ],
echo       "env": {
echo         "PYTHONUTF8": "1",
echo         "UNIFIED_MEMORY_STORAGE_DIR": "%PROJECT_DIR:\=/%/data"
echo       }
echo     }
echo   }
echo }
) > "%MCP_FILE%"

echo 已创建: %MCP_FILE%

:: 测试运行
echo.
echo 测试 MCP 服务器...
"%PYTHON%" -c "import sys; sys.path.insert(0, r'%PROJECT_DIR%\src'); from unified_memory.api.mcp_server import create_mcp_server; print('MCP 模块导入成功')" 2>&1 | findstr /V "Loading weights"

echo.
echo ========================================
echo   安装完成!
echo.
echo   启动方式:
echo     HTTP 模式: run_http.py
echo     MCP 模式:  run_mcp.py
echo.
echo   将 .mcp.json 复制到 Codex 插件目录即可自动集成
echo ========================================
pause
