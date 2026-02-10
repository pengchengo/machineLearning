"""
纯 Python Agent 入口脚本（无框架）

使用前请：
1. pip install -r requirements.txt
2. 设置环境变量 OPENAI_API_KEY，或在项目根目录创建 .env 文件
"""
import sys
from pathlib import Path

# 确保项目根目录在 path 中，支持 python agent/main.py 和 python -m agent.main
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass

from agent import run_agent


def main():
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("Agent 已启动（纯 Python，无框架）")
        print("输入任务后回车执行，输入 quit 或 exit 退出。\n")
        user_input = input("你: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            return

    result = run_agent(user_input)
    print("\n--- 结果 ---")
    print(result)


if __name__ == "__main__":
    main()
