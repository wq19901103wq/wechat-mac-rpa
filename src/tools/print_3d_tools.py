"""3D 打印工具 — Bambu Lab 打印机自动化

支持 3MF 文件处理（读取配置、缩放模型、修改支撑）和打印机状态查询。
所有 XML 解析使用 defusedxml.ElementTree（无正则）。
"""

import json
import logging
import ssl
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from defusedxml import ElementTree as ET

from .tool_registry import get_registry

_logger = logging.getLogger("src.tools.print3d")


# ── 3MF 文件处理 ──


class _ThreeMFReader:
    """3MF 文件读取器（内部辅助类）"""

    CONFIG_PATH = "Metadata/project_settings.config"
    MODEL_PATH = "3D/3dmodel.model"

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"3MF 文件不存在: {file_path}")

    def read_config(self) -> Dict:
        """读取 3MF 中的项目配置（JSON）"""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            data = zf.read(self.CONFIG_PATH)
        return json.loads(data.decode("utf-8"))

    def read_model_xml(self) -> str:
        """读取 3D 模型 XML 文本"""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            return zf.read(self.MODEL_PATH).decode("utf-8")

    def read_model_transforms(self) -> List[Dict[str, str]]:
        """解析所有带 transform 属性的元素，返回 {element_tag, transform_value} 列表"""
        xml_text = self.read_model_xml()
        root = ET.fromstring(xml_text)
        results = []
        for elem in root.iter():
            transform = elem.get("transform")
            if transform:
                results.append({"tag": elem.tag, "transform": transform})
        return results

    def copy_to(self, dest_path: str) -> "_ThreeMFWriter":
        """返回写入器，用于基于当前文件创建修改后的副本"""
        return _ThreeMFWriter(str(self.file_path), dest_path)


class _ThreeMFWriter:
    """3MF 文件写入器（内部辅助类）"""

    def __init__(self, source_path: str, dest_path: str):
        self.source_path = Path(source_path)
        self.dest_path = Path(dest_path)

    def write(self, config_bytes: Optional[bytes] = None, model_bytes: Optional[bytes] = None):
        """将修改后的 config 和/或 model 写入新的 3MF 文件"""
        self.dest_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.source_path, "r") as zin:
            with zipfile.ZipFile(self.dest_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)
                    if item == _ThreeMFReader.CONFIG_PATH and config_bytes is not None:
                        zout.writestr(item, config_bytes)
                    elif item == _ThreeMFReader.MODEL_PATH and model_bytes is not None:
                        zout.writestr(item, model_bytes)
                    else:
                        zout.writestr(item, data)


# ── 公开工具函数 ──


def print3d_read_3mf(file_path: str = "") -> str:
    """读取 3MF 文件的配置和模型尺寸信息

    Args:
        file_path: 3MF 文件路径

    Returns:
        配置摘要和模型 transform 矩阵信息
    """
    if not file_path:
        return "请提供 3MF 文件路径，例如：print3d_read_3mf('/path/to/model.3mf')"

    try:
        reader = _ThreeMFReader(file_path)
        config = reader.read_config()
        transforms = reader.read_model_transforms()

        lines = [f"📁 {Path(file_path).name}"]

        # 配置摘要
        lines.append("")
        lines.append("🔧 配置摘要:")
        for key in ("enable_support", "support_type", "layer_height", "initial_layer_height"):
            val = config.get(key)
            if val is not None:
                lines.append(f"  {key}: {val}")

        # Transform 矩阵
        lines.append("")
        lines.append(f"📐 模型 transform 矩阵（共 {len(transforms)} 个）:")
        for i, t in enumerate(transforms[:10], 1):
            vals = t["transform"].split()
            if len(vals) == 12:
                sx, sy, sz = vals[0], vals[4], vals[8]
                lines.append(f"  {i}. {t['tag']}: 缩放=({sx}, {sy}, {sz})")
            else:
                lines.append(f"  {i}. {t['tag']}: {t['transform'][:60]}...")
        if len(transforms) > 10:
            lines.append(f"  ... 还有 {len(transforms) - 10} 个")

        return "\n".join(lines)
    except FileNotFoundError as e:
        return f"❌ 文件未找到: {e}"
    except Exception as e:
        _logger.error("[3D] read_3mf failed: %s", e)
        return f"❌ 读取 3MF 文件失败: {e}"


def print3d_scale_model(input_file: str = "", output_file: str = "", scale: float = 1.0) -> str:
    """缩放 3MF 模型（X/Y/Z 等比缩放）

    Args:
        input_file: 输入 3MF 文件路径
        output_file: 输出 3MF 文件路径（默认在同级目录生成 *_scaled.3mf）
        scale: 缩放比例，如 2.0 表示放大到 2 倍

    Returns:
        操作结果描述
    """
    if not input_file:
        return "请提供输入文件路径，例如：print3d_scale_model('/path/to/model.3mf', scale=2.0)"

    input_path = Path(input_file)
    if not input_path.exists():
        return f"❌ 输入文件不存在: {input_file}"

    if not output_file:
        output_file = str(input_path.parent / f"{input_path.stem}_scaled.3mf")

    try:
        reader = _ThreeMFReader(input_file)
        xml_text = reader.read_model_xml()
        root = ET.fromstring(xml_text)

        modified_count = 0
        for elem in root.iter():
            transform_str = elem.get("transform")
            if not transform_str:
                continue
            values = transform_str.split()
            if len(values) != 12:
                continue
            try:
                # 索引 0, 4, 8 分别是 X, Y, Z 缩放因子
                values[0] = str(float(values[0]) * scale)
                values[4] = str(float(values[4]) * scale)
                values[8] = str(float(values[8]) * scale)
                elem.set("transform", " ".join(values))
                modified_count += 1
            except ValueError:
                continue

        # 写回 XML
        new_xml = ET.tostring(root, encoding="unicode")
        # 确保 XML 声明
        if not new_xml.startswith("<?xml"):
            new_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + new_xml

        writer = reader.copy_to(output_file)
        writer.write(model_bytes=new_xml.encode("utf-8"))

        return (
            f"✅ 缩放完成: {scale} 倍\n"
            f"   修改了 {modified_count} 个 transform 矩阵\n"
            f"   输出文件: {output_file}"
        )
    except Exception as e:
        _logger.error("[3D] scale_model failed: %s", e)
        return f"❌ 缩放模型失败: {e}"


def print3d_update_support(
    input_file: str = "",
    output_file: str = "",
    branch_diameter: str = "3",
    threshold_angle: str = "45",
    wall_count: str = "1",
    interface_layers: str = "3",
) -> str:
    """修改 3MF 文件的支撑配置

    Args:
        input_file: 输入 3MF 文件路径
        output_file: 输出 3MF 文件路径（默认生成 *_supported.3mf）
        branch_diameter: 树状支撑主干直径 (2-15mm)，默认 3
        threshold_angle: 支撑阈值角度 (30-60°)，默认 45
        wall_count: 支撑墙数 (0-4)，默认 1
        interface_layers: 界面层数 (2-6)，默认 3

    Returns:
        操作结果描述
    """
    if not input_file:
        return "请提供输入文件路径，例如：print3d_update_support('/path/to/model.3mf')"

    input_path = Path(input_file)
    if not input_path.exists():
        return f"❌ 输入文件不存在: {input_file}"

    if not output_file:
        output_file = str(input_path.parent / f"{input_path.stem}_supported.3mf")

    try:
        reader = _ThreeMFReader(input_file)
        config = reader.read_config()

        # 计算派生参数
        tip_diameter = str(float(branch_diameter) * 0.2)
        branch_angle = "65" if float(branch_diameter) > 6 else "50"
        support_speed = "150" if float(branch_diameter) > 6 else "80"

        config.update({
            "enable_support": "1",
            "support_type": "tree(auto)",
            "support_threshold_angle": threshold_angle,
            "tree_support_branch_diameter": branch_diameter,
            "tree_support_tip_diameter": tip_diameter,
            "tree_support_branch_angle": branch_angle,
            "tree_support_wall_count": wall_count,
            "support_interface_top_layers": interface_layers,
            "support_interface_bottom_layers": interface_layers,
            "support_xy_distance": "1.0",
            "support_speed": support_speed,
        })

        config_bytes = json.dumps(config, indent=4, ensure_ascii=False).encode("utf-8")
        writer = reader.copy_to(output_file)
        writer.write(config_bytes=config_bytes)

        return (
            f"✅ 支撑配置更新完成\n"
            f"   主干直径: {branch_diameter}mm\n"
            f"   阈值角度: {threshold_angle}°\n"
            f"   墙数: {wall_count}\n"
            f"   界面层数: {interface_layers}\n"
            f"   输出文件: {output_file}"
        )
    except Exception as e:
        _logger.error("[3D] update_support failed: %s", e)
        return f"❌ 更新支撑配置失败: {e}"


def _load_printer_config() -> Dict:
    """读取 ~/.openclaw/printer_config.json 作为默认配置"""
    config_path = Path.home() / ".openclaw" / "printer_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            _logger.debug("[3D] 读取打印机配置失败: %s", e)
    return {}


def print3d_get_printer_status(ip: str = "", access_code: str = "", serial: str = "") -> str:
    """通过 MQTT 获取 Bambu Lab 打印机状态

    参数为空时自动读取 ~/.openclaw/printer_config.json

    Args:
        ip: 打印机 IP 地址
        access_code: 打印机访问码（在打印机设置 → LAN 中查看）
        serial: 打印机序列号

    Returns:
        打印机状态信息
    """
    config = _load_printer_config()
    ip = ip or config.get("ip_address", "")
    access_code = access_code or config.get("access_code", "")
    serial = serial or config.get("serial_number", "")

    if not all([ip, access_code, serial]):
        missing = []
        if not ip:
            missing.append("ip")
        if not access_code:
            missing.append("access_code")
        if not serial:
            missing.append("serial")
        return (
            f"❌ 缺少参数: {', '.join(missing)}\n"
            "可在 ~/.openclaw/printer_config.json 中配置默认值，或手动提供:\n"
            "  ip: 打印机 IP（如 192.168.1.100）\n"
            "  access_code: 访问码（打印机设置 → LAN）\n"
            "  serial: 序列号"
        )

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return "❌ paho-mqtt 未安装，运行: pip3 install paho-mqtt"

    # paho-mqtt 2.x compatibility
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
        )
    except AttributeError:
        # paho-mqtt 1.x fallback
        client = mqtt.Client(protocol=mqtt.MQTTv311)

    client.username_pw_set("bblp", access_code)
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

    status_data: Dict = {}
    connected = False

    def on_connect(c, userdata, flags, rc, properties=None):
        nonlocal connected
        if rc == 0:
            connected = True
            topic = f"device/{serial}/report"
            c.subscribe(topic)
            # Request full status push
            req_topic = f"device/{serial}/request"
            cmd = {"pushing": {"sequence_id": "0", "command": "pushall"}}
            c.publish(req_topic, json.dumps(cmd))
        else:
            _logger.warning("[3D] MQTT connect failed, rc=%s", rc)

    def on_message(c, userdata, msg):
        nonlocal status_data
        try:
            data = json.loads(msg.payload)
            if "print" in data:
                status_data.update(data["print"])
        except Exception as e:
            _logger.warning("parse mqtt payload failed: %s", e)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(ip, 8883, 60)
        client.loop_start()
        # Wait for connection + full status
        for _ in range(20):
            if connected and status_data.get("gcode_state") is not None:
                break
            time.sleep(0.5)
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        _logger.error("[3D] MQTT error: %s", e)
        return f"❌ 连接打印机失败: {e}"

    if not status_data:
        return f"⚠️ 已连接 {ip}，但未收到状态数据（请检查序列号或 Access Code 是否正确）"

    state_map = {
        "IDLE": "🟢 空闲",
        "RUNNING": "🟡 打印中",
        "PAUSE": "⏸️ 暂停",
        "FINISH": "✅ 完成",
        "FAILED": "❌ 失败",
        "PREPARE": "🔄 准备中",
    }
    state = status_data.get("gcode_state", "未知")
    state_display = state_map.get(state, state)

    lines = [
        f"🖨️ Bambu Lab {config.get('model', 'A1')} ({serial[:8]}...)",
        f"   状态: {state_display}",
        f"   热床: {status_data.get('bed_temper', 'N/A')}°C / 目标 {status_data.get('bed_target_temper', 'N/A')}°C",
        f"   喷嘴: {status_data.get('nozzle_temper', 'N/A')}°C / 目标 {status_data.get('nozzle_target_temper', 'N/A')}°C",
        f"   WiFi: {status_data.get('wifi_signal', 'N/A')}",
    ]

    # Light status
    lights = status_data.get("lights_report", [])
    if lights:
        light_states = ", ".join(f"{light.get('node', '?')}={'ON' if light.get('mode') == 'on' else 'OFF'}" for light in lights)
        lines.append(f"   灯光: {light_states}")

    # Print progress
    progress = status_data.get("mc_percent")
    if progress is not None:
        lines.append(f"   进度: {progress}%")
    remaining = status_data.get("mc_remaining_time")
    if remaining is not None:
        lines.append(f"   剩余: {remaining} 分钟")
    layer = status_data.get("layer_num")
    total_layer = status_data.get("total_layer_num")
    if layer is not None and total_layer:
        lines.append(f"   层数: {layer}/{total_layer}")
    task = status_data.get("subtask_name")
    if task:
        lines.append(f"   文件: {task}")

    return "\n".join(lines)


# ── 注册 ──


def register_print3d_tools(registry=None):
    """注册 3D 打印工具到全局注册表"""
    if registry is None:
        registry = get_registry()

    registry.register(
        name="print3d_read_3mf",
        description="读取 3MF 文件的配置和模型尺寸信息",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "3MF 文件路径",
                },
            },
            "required": ["file_path"],
        },
        func=print3d_read_3mf,
    )

    registry.register(
        name="print3d_scale_model",
        description="缩放 3MF 模型（X/Y/Z 等比缩放）",
        parameters={
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "输入 3MF 文件路径",
                },
                "output_file": {
                    "type": "string",
                    "description": "输出 3MF 文件路径（默认生成 *_scaled.3mf）",
                },
                "scale": {
                    "type": "number",
                    "description": "缩放比例，如 2.0 表示放大到 2 倍",
                },
            },
            "required": ["input_file", "scale"],
        },
        func=print3d_scale_model,
    )

    registry.register(
        name="print3d_update_support",
        description="修改 3MF 文件的支撑配置",
        parameters={
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "输入 3MF 文件路径",
                },
                "output_file": {
                    "type": "string",
                    "description": "输出 3MF 文件路径（默认生成 *_supported.3mf）",
                },
                "branch_diameter": {
                    "type": "string",
                    "description": "树状支撑主干直径 (2-15mm)，默认 3",
                },
                "threshold_angle": {
                    "type": "string",
                    "description": "支撑阈值角度 (30-60°)，默认 45",
                },
                "wall_count": {
                    "type": "string",
                    "description": "支撑墙数 (0-4)，默认 1",
                },
                "interface_layers": {
                    "type": "string",
                    "description": "界面层数 (2-6)，默认 3",
                },
            },
            "required": ["input_file"],
        },
        func=print3d_update_support,
    )

    registry.register(
        name="print3d_get_printer_status",
        description="通过 MQTT 获取 Bambu Lab 打印机状态",
        parameters={
            "type": "object",
            "properties": {
                "ip": {
                    "type": "string",
                    "description": "打印机 IP 地址",
                },
                "access_code": {
                    "type": "string",
                    "description": "打印机访问码（在打印机设置 → LAN 中查看）",
                },
                "serial": {
                    "type": "string",
                    "description": "打印机序列号",
                },
            },
            "required": ["ip", "access_code", "serial"],
        },
        func=print3d_get_printer_status,
    )
