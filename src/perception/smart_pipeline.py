"""Compatibility alias for the shared WeChat reader."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("wechat_reader.perception.smart_pipeline")
