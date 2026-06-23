"""工具注册表 - 管理所有可用工具"""

import json
import logging
import threading
from typing import Any, Callable, Dict, List

_logger = logging.getLogger("src.tool_registry")


class Tool:
    """单个工具定义"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_schema(self) -> Dict:
        """转换为 OpenAI function calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: str) -> str:
        """执行工具，返回文本结果"""
        try:
            args = json.loads(arguments) if arguments else {}
            result = self.func(**args)
            return str(result) if result is not None else ""
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            _logger.warning("工具 %s 参数解析/执行失败: %s", self.name, e)
            return f"工具执行出错: {e}"
        except Exception as e:
            _logger.warning("工具 %s 执行异常: %s", self.name, e, exc_info=True)
            return f"工具执行出错: {e}"


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ) -> Tool:
        """注册一个新工具"""
        tool = Tool(name, description, parameters, func)
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_openai_schemas(self) -> List[Dict]:
        """获取所有工具的 OpenAI schema"""
        return [t.to_openai_schema() for t in self._tools.values()]


# 全局单例（受 Lock 保护的双重检查锁定，见 AGENTS.md 3.6）
_registry: "ToolRegistry | None" = None
_registry_lock = threading.Lock()


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ToolRegistry()
    return _registry
