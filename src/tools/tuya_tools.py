"""Tuya Smart Home Tools — 涂鸦智能家居控制

通过微信 Agent 调用 Tuya Cloud API 或 Local LAN 控制智能家居设备。

两种模式：
- Cloud（默认）：通过 Tuya Cloud API 控制，配置简单，需联网
- Local：直接通过 WiFi 局域网控制，延迟低，不依赖外网，但需手动配置设备 ID/IP/Key

配置来源（优先级从高到低）：
1. 环境变量 TUYA_ACCESS_ID / TUYA_ACCESS_SECRET / TUYA_ENDPOINT / TUYA_UID / TUYA_MODE
2. ~/.openclaw/tuya_config.json
3. ~/.openclaw/tuya_local_devices.json
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .tool_registry import get_registry

_logger = logging.getLogger("src.tools.tuya")

_CONFIG_FILE = Path.home() / ".openclaw" / "tuya_config.json"
_DEVICES_FILE = Path.home() / ".openclaw" / "tuya_devices.json"
_LOCAL_DEVICES_FILE = Path.home() / ".openclaw" / "tuya_local_devices.json"

# 设备类别映射
category_map = {
    "dj": "💡 灯",
    "kt": "❄️ 空调",
    "cl": "🪟 窗帘",
    "qn": "🔥 地暖",
    "xfj": "🌀 新风",
    "wg2": "📡 网关",
}

# 控制指令映射：code -> {on_value, off_value}
_control_codes = {
    "dj": {"code": "switch_led", "on": True, "off": False},
    "kt": {"code": "switch", "on": True, "off": False},
    "cl": {"code": "control", "on": "open", "off": "close"},
    "qn": {"code": "switch", "on": True, "off": False},
    "xfj": {"code": "switch", "on": True, "off": False},
    "wg2": {"code": "master_mode", "on": True, "off": False},
}


def _get_mode() -> str:
    """获取控制模式：cloud / local / auto"""
    mode = os.getenv("TUYA_MODE", "auto").lower()
    if mode in ("cloud", "local"):
        return mode
    return "auto"


def _load_config() -> Dict:
    """加载 Tuya Cloud 配置"""
    config = {}
    if _CONFIG_FILE.exists():
        config = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    # 环境变量覆盖
    env_mapping = {
        "access_id": "TUYA_ACCESS_ID",
        "access_secret": "TUYA_ACCESS_SECRET",
        "api_endpoint": "TUYA_ENDPOINT",
        "uid": "TUYA_UID",
    }
    for key, env_key in env_mapping.items():
        val = os.getenv(env_key)
        if val:
            config[key] = val
    return config


def _load_local_devices() -> List[Dict]:
    """加载本地局域网设备配置"""
    if _LOCAL_DEVICES_FILE.exists():
        return json.loads(_LOCAL_DEVICES_FILE.read_text(encoding="utf-8"))
    return []


def _find_local_device(name: str) -> Optional[Dict]:
    """在本地设备列表中模糊匹配"""
    devices = _load_local_devices()
    name_lower = name.lower()
    for d in devices:
        if name_lower in d.get("name", "").lower():
            return d
    return None


def _load_devices() -> List[Dict]:
    """加载本地缓存的设备列表"""
    if _DEVICES_FILE.exists():
        return json.loads(_DEVICES_FILE.read_text(encoding="utf-8"))
    return []


def _save_devices(devices: List[Dict]) -> None:
    """保存设备列表到本地缓存"""
    _DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DEVICES_FILE.write_text(json.dumps(devices, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_api():
    """获取 Tuya API 连接"""
    try:
        from tuya_connector import TuyaOpenAPI
    except ImportError:
        _logger.warning("[Tuya] tuya-connector-python 未安装")
        return None, "tuya-connector-python 未安装，运行: pip3 install tuya-connector-python"

    config = _load_config()
    access_id = config.get("access_id", "")
    access_secret = config.get("access_secret", "")
    endpoint = config.get("api_endpoint", "https://openapi.tuyacn.com")

    if not access_id or not access_secret:
        _logger.warning("[Tuya] 缺少 API 凭证")
        return None, "缺少 Tuya API 凭证。请先运行: python3 scripts/tuya_setup.py"

    try:
        api = TuyaOpenAPI(endpoint, access_id, access_secret)
        api.connect()
        _logger.info("[Tuya] Cloud API 连接成功: endpoint=%s", endpoint)
        return api, ""
    except Exception as e:
        _logger.error("[Tuya] Cloud API 连接失败: %s", e)
        return None, f"Tuya API 连接失败: {e}"


def _fetch_devices_from_api() -> List[Dict]:
    """从 Tuya API 拉取设备列表并缓存"""
    api, err = _get_api()
    if not api:
        _logger.warning("[Tuya] 拉取设备列表失败: %s", err)
        return []

    config = _load_config()
    uid = config.get("uid", "")

    try:
        if uid:
            response = api.get(f"/v1.0/users/{uid}/devices")
        else:
            response = api.get("/v1.0/users/-/devices")
    except Exception as e:
        _logger.error("[Tuya] API 请求异常: %s", e)
        return []

    if not response.get("success"):
        _logger.warning("[Tuya] API 返回失败: %s", response.get('msg', 'Unknown'))
        return []

    devices = []
    for d in response.get("result", []):
        devices.append({
            "name": d.get("name", "Unknown"),
            "id": d.get("id"),
            "category": d.get("category", ""),
            "online": d.get("online", False),
        })

    _logger.info("[Tuya] 从 Cloud API 拉取 %d 个设备", len(devices))
    _save_devices(devices)
    return devices


def _get_devices() -> List[Dict]:
    """获取设备列表（优先本地缓存，没有则拉取 API）"""
    devices = _load_devices()
    if not devices:
        devices = _fetch_devices_from_api()
    return devices


def _find_device(name: str) -> Optional[Dict]:
    """模糊匹配设备名称（Cloud 设备列表）"""
    devices = _get_devices()
    name_lower = name.lower()

    # 1. 精确匹配
    for d in devices:
        if d.get("name", "").lower() == name_lower:
            _logger.info("[Tuya] 精确匹配设备: %s -> %s", name, d.get("name"))
            return d

    # 2. 包含匹配
    for d in devices:
        if name_lower in d.get("name", "").lower():
            _logger.info("[Tuya] 包含匹配设备: %s -> %s", name, d.get("name"))
            return d

    # 3. 反向包含
    for d in devices:
        if d.get("name", "").lower() in name_lower:
            _logger.info("[Tuya] 反向包含匹配设备: %s -> %s", name, d.get("name"))
            return d

    _logger.warning("[Tuya] 未匹配到设备: %s", name)
    return None


def _control_device_local(device_info: Dict, action: str) -> Optional[str]:
    """通过 Local LAN 控制设备，返回结果字符串或 None（失败时）"""
    try:
        import tinytuya
    except ImportError:
        _logger.debug("[Tuya] tinytuya 未安装，跳过 Local 模式")
        return None

    device_name = device_info.get("name", "Unknown")
    try:
        _logger.info("[Tuya] Local 控制: %s %s", device_name, action)
        device = tinytuya.Device(
            device_info["id"],
            device_info["ip"],
            device_info["key"],
            version=device_info.get("version", "3.1"),
        )
        device.set_dpsUsed({"1": None})

        if action == "on":
            device.turn_on()
            _logger.info("[Tuya] Local 控制成功: %s -> on", device_name)
            return f"✅ {device_name} 已打开（Local）"
        elif action == "off":
            device.turn_off()
            _logger.info("[Tuya] Local 控制成功: %s -> off", device_name)
            return f"✅ {device_name} 已关闭（Local）"
        elif action == "status":
            status = device.status()
            _logger.info("[Tuya] Local 状态获取成功: %s", device_name)
            lines = [f"📟 {device_name} 状态（Local）："]
            lines.append(f"   在线: {'🟢 是' if status else '⚫ 否'}")
            if status:
                for k, v in status.items():
                    lines.append(f"   • {k}: {v}")
            return "\n".join(lines)
    except Exception as e:
        _logger.warning("[Tuya] Local 控制失败: %s %s — %s", device_name, action, e)
        return None
    return None


def tuya_list_devices() -> str:
    """列出所有 Tuya 智能家居设备及其状态"""
    _logger.info("[Tuya] Tool 调用: tuya_list_devices")
    devices = _get_devices()
    if not devices:
        devices = _fetch_devices_from_api()

    if not devices:
        _logger.warning("[Tuya] 无可用设备")
        return "未找到设备。请确认：1) Tuya API 凭证已配置 2) 涂鸦 App 账号已关联到 IoT 项目 3) 设备在线。运行 `python3 scripts/tuya_setup.py` 进行配置。"

    _logger.info("[Tuya] 返回 %d 个设备列表", len(devices))
    lines = [f"📱 发现 {len(devices)} 个设备："]
    for i, d in enumerate(devices, 1):
        status = "🟢" if d.get("online") else "⚫"
        cat = category_map.get(d.get("category"), "📟 设备")
        lines.append(f"{i}. {status} {cat} {d.get('name', 'Unknown')}")

    return "\n".join(lines)


def tuya_control_device(device_name: str = "", action: str = "") -> str:
    """控制指定 Tuya 设备

    Args:
        device_name: 设备名称（如"客厅主灯"）
        action: 动作，支持 on（打开）、off（关闭）、status（查看状态）
    """
    _logger.info("[Tuya] Tool 调用: tuya_control_device(name=%s, action=%s)", device_name, action)
    if not device_name:
        return "请提供设备名称，例如：tuya_control_device('客厅主灯', 'on')"

    action = action.lower().strip()
    if action not in ("on", "off", "status"):
        return f"不支持的动作：{action}。支持：on（打开）、off（关闭）、status（查看状态）"

    device = _find_device(device_name)
    if not device:
        _logger.warning("[Tuya] 控制失败: 未找到设备 %s", device_name)
        return f"未找到设备：{device_name}。可用设备：\n{tuya_list_devices()}"

    device_id = device.get("id")
    category = device.get("category", "")
    real_name = device.get("name", device_name)

    if action == "status":
        api, err = _get_api()
        if not api:
            return err
        try:
            response = api.get(f"/v1.0/devices/{device_id}")
            if not response.get("success"):
                _logger.warning("[Tuya] 状态获取失败: %s — %s", real_name, response.get('msg'))
                return f"获取 {real_name} 状态失败: {response.get('msg', 'Unknown')}"
            result = response.get("result", {})
            status_lines = [f"📟 {real_name} 状态："]
            status_lines.append(f"   在线: {'🟢 是' if result.get('online') else '⚫ 否'}")
            for s in result.get("status", []):
                code = s.get("code", "")
                value = s.get("value", "")
                status_lines.append(f"   • {code}: {value}")
            _logger.info("[Tuya] 状态获取成功: %s", real_name)
            return "\n".join(status_lines)
        except Exception as e:
            _logger.error("[Tuya] 状态获取异常: %s — %s", real_name, e)
            return f"获取状态出错: {e}"

    # on / off
    mode = _get_mode()
    _logger.info("[Tuya] 控制模式: %s, 设备: %s, 动作: %s", mode, real_name, action)

    # Local 模式优先（如果配置了本地设备）
    if mode in ("local", "auto"):
        local_device = _find_local_device(device_name)
        if local_device:
            _logger.info("[Tuya] 尝试 Local 控制: %s", real_name)
            result = _control_device_local(local_device, action)
            if result:
                return result
            if mode == "local":
                return f"❌ {real_name} 本地控制失败，请检查设备 IP 和 Local Key"
            _logger.info("[Tuya] Local 失败，回退到 Cloud")
        else:
            _logger.debug("[Tuya] 未找到本地设备配置: %s", device_name)

    if mode == "local":
        return f"未找到本地设备：{device_name}。请先将设备信息添加到 {_LOCAL_DEVICES_FILE}"

    # Cloud 模式
    mapping = _control_codes.get(category)
    if not mapping:
        return f"设备类型 {category} 暂不支持控制，支持的类型：{', '.join(category_map.keys())}"

    code = mapping["code"]
    value = mapping["on"] if action == "on" else mapping["off"]

    api, err = _get_api()
    if not api:
        return err

    try:
        response = api.post(
            f"/v1.0/devices/{device_id}/commands",
            {"commands": [{"code": code, "value": value}]},
        )
        if response.get("success"):
            verb = "打开" if action == "on" else "关闭"
            _logger.info("[Tuya] Cloud 控制成功: %s -> %s", real_name, action)
            return f"✅ {real_name} 已{verb}"
        else:
            _logger.warning("[Tuya] Cloud 控制失败: %s -> %s — %s", real_name, action, response.get('msg'))
            return f"❌ {real_name} 控制失败: {response.get('msg', 'Unknown error')}"
    except Exception as e:
        _logger.error("[Tuya] Cloud 控制异常: %s -> %s — %s", real_name, action, e)
        return f"❌ 控制出错: {e}"


def tuya_set_temperature(device_name: str = "", temperature: int = 26) -> str:
    """设置空调/地暖温度

    Args:
        device_name: 设备名称（如"客厅空调"）
        temperature: 目标温度（默认 26）
    """
    _logger.info("[Tuya] Tool 调用: tuya_set_temperature(name=%s, temp=%d)", device_name, temperature)
    if not device_name:
        return "请提供设备名称，例如：tuya_set_temperature('客厅空调', 26)"

    device = _find_device(device_name)
    if not device:
        _logger.warning("[Tuya] 温度设置失败: 未找到设备 %s", device_name)
        return f"未找到设备：{device_name}。可用设备：\n{tuya_list_devices()}"

    device_id = device.get("id")
    real_name = device.get("name", device_name)

    api, err = _get_api()
    if not api:
        return err

    try:
        response = api.post(
            f"/v1.0/devices/{device_id}/commands",
            {"commands": [{"code": "temp_set", "value": temperature}]},
        )
        if response.get("success"):
            _logger.info("[Tuya] 温度设置成功: %s -> %d℃", real_name, temperature)
            return f"✅ {real_name} 温度已设置为 {temperature}℃"
        else:
            _logger.warning("[Tuya] 温度设置失败: %s -> %d℃ — %s", real_name, temperature, response.get('msg'))
            return f"❌ {real_name} 温度设置失败: {response.get('msg', 'Unknown error')}"
    except Exception as e:
        _logger.error("[Tuya] 温度设置异常: %s -> %d℃ — %s", real_name, temperature, e)
        return f"❌ 设置出错: {e}"


def register_tuya_tools(registry=None):
    """注册 Tuya 智能家居工具到全局注册表"""
    if registry is None:
        registry = get_registry()

    registry.register(
        name="tuya_list_devices",
        description="列出所有 Tuya 智能家居设备及其在线状态。当用户问'有哪些设备'、'列出所有设备'时调用。",
        parameters={"type": "object", "properties": {}},
        func=tuya_list_devices,
    )

    registry.register(
        name="tuya_control_device",
        description="控制指定 Tuya 设备（打开/关闭/查看状态）。当用户说'打开客厅灯'、'关空调'、'查看卧室灯状态'时调用。",
        parameters={
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "设备名称，如'客厅主灯'、'主卧空调'",
                },
                "action": {
                    "type": "string",
                    "enum": ["on", "off", "status"],
                    "description": "动作：on（打开）、off（关闭）、status（查看状态）",
                },
            },
            "required": ["device_name", "action"],
        },
        func=tuya_control_device,
    )

    registry.register(
        name="tuya_set_temperature",
        description="设置空调或地暖的目标温度。当用户说'空调设到26度'、'地暖调到28度'时调用。",
        parameters={
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "设备名称，如'客厅空调'、'客餐厅地暖'",
                },
                "temperature": {
                    "type": "integer",
                    "description": "目标温度，如 26、28",
                },
            },
            "required": ["device_name", "temperature"],
        },
        func=tuya_set_temperature,
    )
