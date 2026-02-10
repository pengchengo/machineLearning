# 纯 Python Agent（无框架）

直接调用大模型 API，自行实现工具编排，不依赖 LangChain 等框架。

## 功能

| 工具 | 说明 |
|------|------|
| `web_search` | 互联网搜索（DuckDuckGo，无需 API Key） |
| `read_file` | 读取本地文件 |
| `write_file` | 写入/修改本地文件 |
| `list_dir` | 列出目录内容 |
| `search_files` | 按文件名模式搜索（如 `*.py`） |

## 依赖

- `openai`：OpenAI API 客户端
- `duckduckgo-search`：网页搜索
- `python-dotenv`：加载 .env 配置

## 配置

在项目根目录创建 `.env` 或设置环境变量：

```
OPENAI_API_KEY=sk-xxx
```

## 使用

```bash
# 从项目根目录运行
python agent/main.py

# 命令行传入任务
python agent/main.py 搜索人脸检测最新进展
python agent/main.py 列出 zerotoone 文件夹下的 py 文件
```

## 实现说明

- **tools.py**：工具定义（OpenAI function calling 格式）与执行逻辑
- **agent.py**：Agent 循环：调用 LLM → 解析 tool_calls → 执行工具 → 将结果反馈 → 继续
- 无 LangChain、LangGraph 等框架，仅用 `openai` 作为 API 客户端
