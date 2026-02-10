"""
工具定义与执行：纯 Python 实现，无框架依赖
- 网上搜索（DuckDuckGo）
- 文件读写、目录列表、文件搜索
"""
import json
from pathlib import Path

# 工作区根目录，限制文件操作范围
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _safe_path(path_str: str) -> Path:
    """限制路径在工作区内"""
    path = Path(path_str).resolve()
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise PermissionError(f"只能访问工作区内的文件: {WORKSPACE_ROOT}")
    return path


# ============ OpenAI Function Calling 格式的工具定义 ============

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "在互联网上搜索资料，获取最新信息。当需要查找新闻、技术文档或任何网上资料时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"},
                    "max_results": {"type": "integer", "description": "返回结果数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容，用于查看、分析文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径（相对或绝对，需在工作区内）"},
                    "encoding": {"type": "string", "description": "文件编码", "default": "utf-8"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入或覆盖本地文件，用于创建新文件或修改现有文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                    "encoding": {"type": "string", "description": "文件编码", "default": "utf-8"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录下的文件和子目录，用于查找、浏览本地文件结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "目录路径，默认为当前目录"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "按文件名模式搜索文件，支持通配符，如 *.py 查找所有 Python 文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "文件名匹配模式，如 *.py"},
                    "root_dir": {"type": "string", "description": "搜索根目录", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
]


# ============ 工具执行函数 ============

def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"未找到与 '{query}' 相关的搜索结果。"

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"{i}. 【{r.get('title', '')}】\n   {r.get('body', '')}\n   链接: {r.get('href', '')}"
            )
        return "\n\n".join(output)
    except ImportError:
        return "请安装 duckduckgo-search: pip install duckduckgo-search"
    except Exception as e:
        return f"搜索失败: {str(e)}"


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    try:
        path = _safe_path(file_path)
        if not path.exists():
            return f"文件不存在: {file_path}"
        if not path.is_file():
            return f"路径不是文件: {file_path}"
        with open(path, "r", encoding=encoding, errors="replace") as f:
            return f.read()
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"读取失败: {str(e)}"


def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    try:
        path = _safe_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return f"已成功写入文件: {file_path}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"写入失败: {str(e)}"


def list_dir(dir_path: str = ".") -> str:
    try:
        path = _safe_path(dir_path)
        if not path.exists():
            return f"目录不存在: {dir_path}"
        if not path.is_dir():
            return f"路径不是目录: {dir_path}"

        items = []
        for p in sorted(path.iterdir()):
            prefix = "[DIR] " if p.is_dir() else "[FILE]"
            items.append(f"{prefix} {p.name}")
        return "\n".join(items) if items else "(空目录)"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"列出目录失败: {str(e)}"


def search_files(pattern: str, root_dir: str = ".") -> str:
    try:
        root = _safe_path(root_dir)
        if not root.exists() or not root.is_dir():
            return f"目录不存在或不是目录: {root_dir}"

        matches = list(root.rglob(pattern))
        if not matches:
            return f"未找到匹配 '{pattern}' 的文件"

        files = [str(p.relative_to(root)) for p in matches if p.is_file()]
        return "\n".join(sorted(files)[:50])
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"搜索失败: {str(e)}"


# 工具名 -> 执行函数映射
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "search_files": search_files,
}


def execute_tool(name: str, arguments: dict) -> str:
    """根据工具名和参数执行工具，返回结果字符串"""
    if name not in TOOL_FUNCTIONS:
        return f"未知工具: {name}"

    fn = TOOL_FUNCTIONS[name]
    try:
        return str(fn(**arguments))
    except TypeError as e:
        return f"参数错误: {e}"
