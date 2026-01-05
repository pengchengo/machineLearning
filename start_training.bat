@echo off
REM 快速启动 Anaconda Prompt 并激活环境
REM 使用方法: 双击此文件即可

REM 设置 Anaconda 安装路径
set "CONDA_BASE=D:\anaconda3"

REM 设置工作目录
set "WORK_DIR=D:\MachineLearning\machineLearning"

REM 设置 conda 环境名称
set "ENV_NAME=sd"

REM 检查 Anaconda 是否存在
if not exist "%CONDA_BASE%\Scripts\activate.bat" (
    echo 错误: 未找到 Anaconda: %CONDA_BASE%
    echo 请检查路径是否正确
    pause
    exit /b 1
)

REM 直接启动新的命令窗口，激活环境并切换目录
REM 使用 /k 参数确保窗口保持打开
start "Anaconda - %ENV_NAME%" cmd /k ""%CONDA_BASE%\Scripts\activate.bat" %CONDA_BASE% && conda activate %ENV_NAME% && cd /d "%WORK_DIR%""

