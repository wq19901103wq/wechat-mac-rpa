# Capture Module Spec

## 1. 模块职责
查找并截图微信主窗口，返回截图路径和窗口几何信息。

## 2. 功能需求 (FR)

- **FR-1**: 使用 Quartz API 查找微信窗口（支持 `WeChat` / `微信` 双名称），返回面积最大的有效窗口。
- **FR-2**: 优先使用 `screencapture -l <windowid>` 截取指定窗口（不受其他窗口覆盖影响），fallback 到 `-R` 按坐标截取。
- **FR-3**: 返回 `CaptureResult`，包含截图路径、窗口 `Rect`、Retina `scale_factor`。
- **FR-4**: 窗口尺寸异常（过小）时，尝试激活微信并等待重试一次；仍异常则触发登录恢复流程。
- **FR-5**: 截图后通过 OCR 轻量验证内容确实是微信窗口（顶部有"搜索"/"微信"等特征）。
- **FR-6**: 每次 `capture()` 生成新的输出路径，避免覆盖旧截图（SmartPipeline 像素 diff 的前提）。
- **FR-7**: 自动清理 `/tmp/` 下超过 1 小时的旧临时截图。

## 3. 非功能需求 (NFR)

- **NFR-1**: 截图超时 5 秒，激活超时 3 秒。
- **NFR-2**: 最小有效窗口尺寸：800x600（小于则视为登录浮窗/未登录状态）。

## 4. 接口契约

### 输入
```python
WindowCapture(
    output_path: Optional[str] = None,  # 默认 /tmp/wechat_capture_{ts}_{pid}.png
    min_effective_width: int = 800,
    min_effective_height: int = 600,
    login_handler: Optional[WeChatLoginHandler] = None,
)
```

### 输出
```python
CaptureResult(
    image_path: str,
    window_rect: Rect,
    scale_factor: float,  # Retina=2.0, 普通=1.0
)
```

## 5. 核心规则与约束

### 规则 1: 窗口查找 Fallback
先尝试 `kCGWindowListOptionOnScreenOnly`，找不到则去掉 `OnScreenOnly`（兼容多 Space / 外接显示器）。

### 规则 2: 截图验证 Graceful Degrade
Tesseract 未安装时跳过验证，不阻断主流程。

## 6. 错误处理

| 异常 | 触发条件 | 处理 |
|------|---------|------|
| `WindowNotFoundError` | 找不到微信窗口 | Bot 层记录日志，跳过本轮 tick |
| `WeChatNotReadyError` | 窗口尺寸异常且恢复失败 | 提示用户可能需要扫码登录 |
| `CaptureValidationError` | 截图内容不像微信窗口 | 提示可能有其他窗口覆盖 |

## 7. 依赖关系
- 依赖 `src.models.base.Rect`
- 依赖 `src.action.login_recovery.WeChatLoginHandler`（可选）
