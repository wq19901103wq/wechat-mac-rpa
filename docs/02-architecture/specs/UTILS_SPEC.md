# Utils Module Spec

## 1. 模块职责
提供全项目共享的通用工具函数。**业务规则必须在 utils 中单点实现，所有消费者统一导入。**

## 2. 功能需求 (FR)

- **FR-1**: `chat_utils._is_group_chat_name(chat_name)`：判断聊天名称是否为群聊（以群人数结尾，如 `xxx（128）` 或 `xxx (5)`）。
- **FR-2**: `chat_utils._normalize_chat_name(name)`：Unicode 归一化 + 去掉群人数后缀 + 去掉序号前缀 + 压缩空白。用于生成稳定的 session key。
- **FR-3**: `chat_utils._extract_session_key(name)`：提取稳定的 session key（调用 `_normalize_chat_name`）。
- **FR-4**: `xml_utils._extract_xml_text(xml)`：从 XML 消息中提取可读文本（title + des）。
- **FR-5**: `text_utils._truncate_text(text, max_len)` / `_compress_text(text, max_chars)`：通用文本截断/压缩。
- **FR-6**: `debug_logger.DebugLogger`：tick 级调试信息收集和持久化（JSON + Markdown）。

## 3. 非功能需求 (NFR)

- **NFR-1**: 纯函数，无副作用，不依赖项目内其他模块（除 models 外）。
- **NFR-2**: 线程安全：工具函数无共享状态。

## 4. 接口契约

### chat_utils
```python
_is_group_chat_name(chat_name: str) -> bool   # 正则匹配中英文括号+数字结尾
_normalize_chat_name(name: str) -> str         # 归一化 + 去后缀
_extract_session_key(name: str) -> str         # 别名
```

### xml_utils
```python
_extract_xml_text(xml: str) -> Optional[str]   # 提取 title + des
```

### text_utils
```python
_truncate_text(text: str, max_len: int, suffix="\n\n... [truncated]") -> str
_compress_text(text: str, max_chars: int) -> str   # 保留头 40% + 尾 40%
```

## 5. 核心规则与约束（红线）

### 规则 1: `_is_group_chat_name` 是群聊判断的唯一实现
**禁止**在任何其他模块中重新定义群聊判断逻辑。所有模块必须从 `chat_utils` 导入：
```python
from src.utils.chat_utils import _is_group_chat_name
```
违反此规则会导致感知层与存储层的群聊判断不一致（历史教训：eef109f commit 引入的 bug）。

### 规则 2: `_normalize_chat_name` 只做 Unicode 归一化和后缀去除
不得在此函数中添加业务逻辑（如根据内容判断群聊/私聊）。该函数仅用于生成稳定的 session key，防止 chat_name 因群人数变化而分裂。

### 规则 3: 新增共享规则必须放到 utils
如果某个判断逻辑被 2 个及以上模块使用，必须提取到 `utils/` 下的相应模块，禁止分散实现。

## 6. 错误处理
纯函数，不抛异常。输入为空或非法时返回空字符串 / False / None。

## 7. 依赖关系
- `chat_utils` / `xml_utils` / `text_utils` 只依赖标准库
- `debug_logger` 依赖 `src.models.base`
- 被几乎所有其他模块依赖
