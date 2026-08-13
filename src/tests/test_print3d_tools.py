#!/usr/bin/env python3
"""3D 打印工具单元测试 —— print_3d_tools"""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tools.print_3d_tools import (
    print3d_get_printer_status,
    print3d_read_3mf,
    print3d_scale_model,
    print3d_update_support,
    register_print3d_tools,
)
from src.tools.tool_registry import ToolRegistry

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_3MF = FIXTURE_DIR / "test_model.3mf"


def _make_minimal_3mf(path: Path, transforms=None, config=None):
    """辅助：创建最小 3MF 文件用于测试"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if transforms is None:
        transforms = [("1 0 0 0 1 0 0 0 1 0 0 0",)]
    if config is None:
        config = {"enable_support": "0", "layer_height": "0.2"}

    items = ""
    for i, t in enumerate(transforms, 1):
        items += f'    <item objectid="1" transform="{t[0]}" />\n'

    model_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources><object id="1" type="model"/></resources>
  <build>\n{items}  </build>
</model>'''

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/project_settings.config", json.dumps(config))


# ── print3d_read_3mf ──


class TestRead3MF:
    def test_read_success(self):
        result = print3d_read_3mf(str(SAMPLE_3MF))
        assert "test_model.3mf" in result
        assert "enable_support: 0" in result
        assert "layer_height: 0.2" in result
        assert "缩放=(1, 1, 1)" in result
        assert "缩放=(2, 2, 2)" in result

    def test_empty_path(self):
        result = print3d_read_3mf("")
        assert "请提供" in result

    def test_missing_file(self):
        result = print3d_read_3mf("/nonexistent/model.3mf")
        assert "文件未找到" in result or "文件不存在" in result


# ── print3d_scale_model ──


class TestScaleModel:
    def test_scale_success(self, tmp_path: Path):
        src = tmp_path / "input.3mf"
        dst = tmp_path / "output.3mf"
        _make_minimal_3mf(src, transforms=[("1 0 0 0 1 0 0 0 1 0 0 0",)])

        result = print3d_scale_model(str(src), str(dst), scale=3.0)
        assert "✅ 缩放完成" in result
        assert "3.0 倍" in result
        assert "修改了 1 个" in result
        assert dst.exists()

        # 验证输出文件 transform 已更新
        out = print3d_read_3mf(str(dst))
        assert "缩放=(3.0, 3.0, 3.0)" in out

    def test_scale_default_output(self, tmp_path: Path):
        src = tmp_path / "model.3mf"
        _make_minimal_3mf(src)
        result = print3d_scale_model(str(src), scale=2.0)
        assert "model_scaled.3mf" in result
        assert (tmp_path / "model_scaled.3mf").exists()

    def test_empty_input(self):
        result = print3d_scale_model("", scale=2.0)
        assert "请提供" in result

    def test_missing_file(self):
        result = print3d_scale_model("/nonexistent/model.3mf", scale=2.0)
        assert "文件不存在" in result


# ── print3d_update_support ──


class TestUpdateSupport:
    def test_update_success(self, tmp_path: Path):
        src = tmp_path / "input.3mf"
        dst = tmp_path / "output.3mf"
        _make_minimal_3mf(src, config={"enable_support": "0", "layer_height": "0.2"})

        result = print3d_update_support(
            str(src), str(dst),
            branch_diameter="5",
            threshold_angle="60",
            wall_count="2",
            interface_layers="4",
        )
        assert "✅ 支撑配置更新完成" in result
        assert "主干直径: 5mm" in result
        assert "阈值角度: 60°" in result
        assert dst.exists()

        # 验证输出文件配置已更新
        out = print3d_read_3mf(str(dst))
        assert "enable_support: 1" in out
        assert "support_type: tree(auto)" in out

    def test_update_default_output(self, tmp_path: Path):
        src = tmp_path / "model.3mf"
        _make_minimal_3mf(src)
        result = print3d_update_support(str(src))
        assert "model_supported.3mf" in result
        assert (tmp_path / "model_supported.3mf").exists()

    def test_empty_input(self):
        result = print3d_update_support("")
        assert "请提供" in result

    def test_missing_file(self):
        result = print3d_update_support("/nonexistent/model.3mf")
        assert "文件不存在" in result


# ── print3d_get_printer_status ──


def _make_mock_mqtt_client(state_data: dict):
    """辅助：创建模拟 MQTT 客户端，模拟完整的连接-订阅-推送流程"""
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.payload = json.dumps({"print": state_data}).encode()

    def fake_loop_start():
        # Simulate: on_connect fires → subscribe + publish pushall
        # Then on_message fires with status data
        if mock_client.on_connect:
            mock_client.on_connect(mock_client, None, {}, 0)
        if mock_client.on_message:
            mock_client.on_message(mock_client, None, mock_msg)

    mock_client.connect = MagicMock()
    mock_client.loop_start = fake_loop_start
    mock_client.loop_stop = MagicMock()
    mock_client.disconnect = MagicMock()
    mock_client.username_pw_set = MagicMock()
    mock_client.tls_set = MagicMock()
    mock_client.tls_insecure_set = MagicMock()
    mock_client.subscribe = MagicMock()
    mock_client.publish = MagicMock()

    return mock_client


class TestGetPrinterStatus:
    def test_empty_params_no_config(self):
        """无配置文件时，空参数应返回缺少参数提示"""
        with patch("src.tools.print_3d_tools._load_printer_config", return_value={}):
            result = print3d_get_printer_status("", "", "")
            assert "缺少参数" in result
            assert "ip" in result
            assert "access_code" in result
            assert "serial" in result

    @patch("paho.mqtt.client.Client")
    def test_auto_config(self, mock_mqtt_cls):
        """有配置文件时，空参数自动读取并使用"""
        mock_client = _make_mock_mqtt_client({
            "gcode_state": "IDLE",
            "bed_temper": 25.0,
            "nozzle_temper": 26.0,
        })
        mock_mqtt_cls.return_value = mock_client

        with patch("src.tools.print_3d_tools._load_printer_config", return_value={
            "ip_address": "192.168.2.7",
            "access_code": "07decbed",
            "serial_number": "03919D570311382",
            "model": "A1",
        }):
            result = print3d_get_printer_status()
            assert "🟢 空闲" in result
            # Verify it used config values
            mock_client.connect.assert_called_once()

    @patch("paho.mqtt.client.Client")
    def test_printer_idle(self, mock_mqtt_cls):
        mock_client = _make_mock_mqtt_client({
            "gcode_state": "IDLE",
            "bed_temper": 45.0,
            "bed_target_temper": 60.0,
            "nozzle_temper": 210.0,
            "nozzle_target_temper": 220.0,
            "wifi_signal": "-40dBm",
            "lights_report": [{"node": "chamber_light", "mode": "off"}],
        })
        mock_mqtt_cls.return_value = mock_client

        result = print3d_get_printer_status("192.168.1.100", "abc123", "00ABC123")
        assert "🟢 空闲" in result
        assert "45.0°C" in result
        assert "210.0°C" in result
        assert "-40dBm" in result
        assert "chamber_light=OFF" in result

    @patch("paho.mqtt.client.Client")
    def test_printer_finish(self, mock_mqtt_cls):
        mock_client = _make_mock_mqtt_client({
            "gcode_state": "FINISH",
            "bed_temper": 27.5,
            "bed_target_temper": 0.0,
            "nozzle_temper": 28.0,
            "nozzle_target_temper": 0.0,
            "mc_percent": 100,
            "mc_remaining_time": 0,
            "layer_num": 51,
            "total_layer_num": 51,
            "subtask_name": "test_model.3mf",
            "wifi_signal": "-35dBm",
        })
        mock_mqtt_cls.return_value = mock_client

        result = print3d_get_printer_status("192.168.1.100", "abc123", "00ABC123")
        assert "✅ 完成" in result
        assert "100%" in result
        assert "51/51" in result
        assert "test_model.3mf" in result

    @patch("paho.mqtt.client.Client")
    def test_printer_running_with_progress(self, mock_mqtt_cls):
        mock_client = _make_mock_mqtt_client({
            "gcode_state": "RUNNING",
            "bed_temper": 60.0,
            "nozzle_temper": 220.0,
            "mc_percent": 42,
            "mc_remaining_time": 120,
            "layer_num": 20,
            "total_layer_num": 100,
        })
        mock_mqtt_cls.return_value = mock_client

        result = print3d_get_printer_status("192.168.1.100", "abc123", "00ABC123")
        assert "🟡 打印中" in result
        assert "42%" in result
        assert "120 分钟" in result
        assert "20/100" in result

    @patch("paho.mqtt.client.Client")
    def test_connection_failure(self, mock_mqtt_cls):
        mock_client = MagicMock()
        mock_client.connect = MagicMock(side_effect=ConnectionRefusedError("Connection refused"))
        mock_client.username_pw_set = MagicMock()
        mock_client.tls_set = MagicMock()
        mock_client.tls_insecure_set = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        result = print3d_get_printer_status("192.168.1.100", "abc123", "00ABC123")
        assert "❌ 连接打印机失败" in result

    @patch("paho.mqtt.client.Client")
    def test_no_data_received(self, mock_mqtt_cls):
        mock_client = MagicMock()
        mock_client.connect = MagicMock()
        mock_client.loop_start = MagicMock()
        mock_client.loop_stop = MagicMock()
        mock_client.disconnect = MagicMock()
        mock_client.username_pw_set = MagicMock()
        mock_client.tls_set = MagicMock()
        mock_client.tls_insecure_set = MagicMock()
        mock_mqtt_cls.return_value = mock_client

        result = print3d_get_printer_status("192.168.1.100", "abc123", "00ABC123")
        assert "⚠️ 已连接" in result
        assert "未收到状态数据" in result


# ── register_print3d_tools ──


class TestRegister:
    def test_register_all_tools(self):
        reg = ToolRegistry()
        register_print3d_tools(reg)
        assert reg.has("print3d_read_3mf")
        assert reg.has("print3d_scale_model")
        assert reg.has("print3d_update_support")
        assert reg.has("print3d_get_printer_status")
        assert len(reg.list_tools()) == 4

    def test_tool_schemas(self):
        reg = ToolRegistry()
        register_print3d_tools(reg)

        schemas = reg.to_openai_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "print3d_read_3mf" in names
        assert "print3d_scale_model" in names
        assert "print3d_update_support" in names
        assert "print3d_get_printer_status" in names

    def test_read_tool_execute(self):
        reg = ToolRegistry()
        register_print3d_tools(reg)
        tool = reg.get("print3d_read_3mf")
        result = tool.execute(f'{{"file_path": "{SAMPLE_3MF}"}}')
        assert "test_model.3mf" in result
