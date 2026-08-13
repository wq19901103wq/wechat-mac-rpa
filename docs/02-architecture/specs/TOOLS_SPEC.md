# Tools Module Spec

## 1. 模块职责
管理 Bot 可调用的外部工具，提供统一的注册、查询、执行接口，适配 OpenAI function calling 格式。

## 2. 功能需求 (FR)

- **FR-1**: `ToolRegistry.register(name, description, parameters, func)`：注册一个新工具。
- **FR-2**: `Tool.to_openai_schema()`：转换为 OpenAI function calling schema。
- **FR-3**: `Tool.execute(arguments)`：解析 JSON 参数并调用底层函数，返回文本结果。
- **FR-4**: 内置工具注册：`get_current_time`、`get_weather`、`web_search`、`browse_url`、`stock_query`、`search_memory`。
- **FR-5**: 全局单例注册表，支持跨模块动态注册（如 `ReplyGenerator` 动态注册 `search_memory`）。

## 3. 非功能需求 (NFR)

- **NFR-1**: 工具执行异常不得抛给 LLM 调用链，必须包装为错误字符串返回。
- **NFR-2**: 参数解析使用 `json.loads`，支持空参数 `{}`。

## 4. 接口契约

### 输入
```python
ToolRegistry().register(
    name: str,
    description: str,
    parameters: Dict[str, Any],  # JSON Schema
    func: Callable,
) -> Tool

tool.execute(arguments: str) -> str
```

### 输出
- `to_openai_schemas()` → `List[Dict]`，供 LLM `tools` 参数使用
- `execute()` → `str`，工具执行结果或错误信息

## 5. 核心规则与约束

### 规则 1: 工具执行异常内部消化
```python
try:
    args = json.loads(arguments) if arguments else {}
    result = self.func(**args)
    return str(result) if result is not None else ""
except Exception as e:
    return f"工具执行出错: {e}"  # 绝不抛异常
```

### 规则 2: `search_memory` 由 MemoryEngine 动态注册
`search_memory` 不在内置工具中硬编码，而是由 `ReplyGenerator` 在初始化时根据 `memory_engine` 是否存在动态注册。确保无记忆引擎时不会产生无效工具。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| 工具不存在 | 返回 `"工具 {name} 不存在"` |
| 参数 JSON 解析失败 | 返回 `"工具执行出错: ..."` |
| 底层函数抛异常 | 捕获后返回错误字符串 |

## 7. 依赖关系
- `builtin_tools.py` 依赖外部 API（天气、搜索、股票）
- `search_memory` 依赖 `src.memory.engine.MemoryEngine`
- 被 `src.reply.generator.ReplyGenerator` 调用
