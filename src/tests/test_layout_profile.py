#!/usr/bin/env python3
"""L2 LayoutProfile 单元测试"""

from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280, LayoutProfile


class TestLayoutProfile:
    def test_creation(self):
        profile = LayoutProfile(
            name="test",
            window_width=1000,
            window_height=800,
            left_boundary=400,
            chat_list_x_max=350,
            title_y_max=50,
            title_x_max_ratio=0.70,
            input_y_min=700,
            self_green=(100, 200, 100),
            self_green_tolerance=30,
            min_bubble_pixels=500,
            message_cluster_threshold=80,
            nickname_x_min_ratio=0.30,
            nickname_x_max_ratio=0.55,
            nickname_y_offset_min=15,
            nickname_y_offset_max=50,
        )
        assert profile.name == "test"
        assert profile.left_boundary == 400
        assert profile.chat_list_x_max == 350

    def test_preset_exists(self):
        assert PROFILE_WECHAT_MAC_1760X1280 is not None
        assert PROFILE_WECHAT_MAC_1760X1280.name == "wechat_mac_4.1.8_1760x1280"
        assert PROFILE_WECHAT_MAC_1760X1280.window_width == 1760
