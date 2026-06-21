#!/usr/bin/env python3
"""MessageSender paste 验证单元测试"""

from unittest.mock import Mock, patch

import pytest

from src.action.message_sender import WeChatMessageSender


@pytest.fixture
def sender():
    return WeChatMessageSender()


class TestPasteVerification:
    """修复方向：paste 后验证输入框内容，不匹配则重试"""

    @patch("subprocess.run")
    def test_paste_verification_success(self, mock_run, sender):
        """paste 验证通过，直接发送，无需重试"""
        # 模拟 pbpaste 返回：第一次读取原始剪贴板，第二次读取验证时的输入框内容
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pbpaste":
                # 第一次：原始剪贴板；第二次：验证时输入框内容
                if not hasattr(side_effect, "call_count"):
                    side_effect.call_count = 0
                side_effect.call_count += 1
                if side_effect.call_count == 1:
                    return Mock(returncode=0, stdout=b"original_clipboard")
                else:
                    # 验证时读到的是要发送的文本
                    return Mock(returncode=0, stdout="测试消息".encode("utf-8"))
            elif cmd[0] == "osascript":
                script = cmd[2] if len(cmd) > 2 else ""
                if "frontApp" in script or "frontmost" in script:
                    return Mock(returncode=0, stdout=b"WeChat\n", stderr=b"")
                return Mock(returncode=0, stderr=b"")
            elif cmd[0] == "pbcopy":
                return Mock(returncode=0, stderr=b"")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = sender.send("测试消息")

        assert result.success is True
        # osascript 调用次数：activate + focus + paste + verify(select+copy) + return
        # 至少应该有 4 次 osascript 调用
        osascript_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "osascript"]
        assert len(osascript_calls) >= 4

    @patch("subprocess.run")
    def test_paste_verification_fail_then_retry(self, mock_run, sender):
        """paste 验证失败，重试后成功"""
        call_log = []

        def side_effect(cmd, **kwargs):
            call_log.append(cmd[0])
            if cmd[0] == "pbpaste":
                # 第1次: 原始剪贴板
                # 第2次: 验证1失败（空内容）
                # 第3次: 验证2成功
                pbpaste_calls = [c for c in call_log if c == "pbpaste"]
                idx = len(pbpaste_calls)
                if idx == 1:
                    return Mock(returncode=0, stdout=b"original")
                elif idx == 2:
                    return Mock(returncode=0, stdout=b"")  # 验证失败：输入框为空
                elif idx == 3:
                    return Mock(returncode=0, stdout="测试消息".encode("utf-8"))  # 重试后成功
                return Mock(returncode=0, stdout=b"")
            elif cmd[0] == "osascript":
                script = cmd[2] if len(cmd) > 2 else ""
                if "frontApp" in script or "frontmost" in script:
                    return Mock(returncode=0, stdout=b"WeChat\n", stderr=b"")
                return Mock(returncode=0, stderr=b"")
            elif cmd[0] == "pbcopy":
                return Mock(returncode=0, stderr=b"")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = sender.send("测试消息")

        assert result.success is True
        # 验证应该有重试：至少多了一次 paste + verify
        pbpaste_count = sum(1 for c in call_log if c == "pbpaste")
        assert pbpaste_count >= 3  # 原始 + 验证1 + 验证2

    @patch("subprocess.run")
    def test_paste_verification_fail_after_max_retries(self, mock_run, sender):
        """paste 验证多次失败，最终返回失败"""
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pbpaste":
                # 总是返回空，验证永远失败
                return Mock(returncode=0, stdout=b"")
            elif cmd[0] == "osascript":
                script = cmd[2] if len(cmd) > 2 else ""
                if "frontApp" in script or "frontmost" in script:
                    return Mock(returncode=0, stdout=b"WeChat\n", stderr=b"")
                return Mock(returncode=0, stderr=b"")
            elif cmd[0] == "pbcopy":
                return Mock(returncode=0, stderr=b"")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = sender.send("测试消息")

        assert result.success is False
        assert "paste" in result.error.lower() or "验证" in result.error.lower() or "发送" in result.error.lower()

    @patch("subprocess.run")
    def test_paste_verification_with_partial_match(self, mock_run, sender):
        """输入框内容包含发送文本（可能还有其他字符），应视为验证通过"""
        def side_effect(cmd, **kwargs):
            if cmd[0] == "pbpaste":
                pbpaste_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "pbpaste"]
                if len(pbpaste_calls) == 0:
                    return Mock(returncode=0, stdout=b"original")
                else:
                    # 输入框内容包含发送文本 + 一些其他字符
                    return Mock(returncode=0, stdout="前缀测试消息后缀".encode("utf-8"))
            elif cmd[0] == "osascript":
                script = cmd[2] if len(cmd) > 2 else ""
                if "frontApp" in script or "frontmost" in script:
                    return Mock(returncode=0, stdout=b"WeChat\n", stderr=b"")
                return Mock(returncode=0, stderr=b"")
            elif cmd[0] == "pbcopy":
                return Mock(returncode=0, stderr=b"")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = sender.send("测试消息")

        assert result.success is True
