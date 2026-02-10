"""
纯 Python Agent：直接调用大模型 API，无框架依赖
实现工具调用循环：LLM 决策 -> 执行工具 -> 将结果反馈给 LLM -> 继续直到完成
"""
import json
import os
from typing import Any, Optional

try:
    from .tools import TOOLS_SCHEMA, execute_tool
except ImportError:
    from tools import TOOLS_SCHEMA, execute_tool

# 使用 OpenAI 官方 Python 客户端（仅为 API 封装，非 Agent 框架）
from openai import OpenAI


SYSTEM_PROMPT = """你是一个智能助手，可以使用以下工具完成任务：

1. **web_search** - 在互联网上搜索资料，获取最新信息
2. **read_file** - 读取本地文件内容
3. **write_file** - 写入或修改本地文件
4. **list_dir** - 列出目录下的文件和子目录
5. **search_files** - 按文件名模式搜索文件（如 *.py）

请根据用户需求选择合适的工具，按步骤完成任务。工作区根目录为项目根目录，文件路径可使用相对路径。"""


def create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """创建 OpenAI 客户端"""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "请设置 OPENAI_API_KEY 环境变量，或在 .env 中配置。"
        )
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def run_agent(
    user_input: str,
    *,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None,
    max_iterations: int = 10,
    verbose: bool = True,
) -> str:
    """
    Agent 主循环：调用 LLM，解析 tool_calls，执行工具，直到返回最终答案。

    Args:
        user_input: 用户输入
        model: 模型名称
        client: OpenAI 客户端，不传则自动创建
        max_iterations: 最大工具调用轮数
        verbose: 是否打印每步执行信息

    Returns:
        最终回复文本
    """
    if client is None:
        client = create_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for iteration in range(max_iterations):
        # 调用 LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message

        # 无 tool_calls，直接返回
        if not msg.tool_calls:
            return (msg.content or "").strip()

        # 收集并执行所有 tool_calls
        tool_messages = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            if verbose:
                print(f"  [工具] {name}({args}) -> {result[:80]}...")
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 将 assistant 消息和 tool 结果加入对话
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        messages.extend(tool_messages)

    return "达到最大迭代次数，任务未完成。"
