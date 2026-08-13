# OCR Module Spec

## 1. 模块职责
基于 macOS Vision 框架进行纯文字识别。**只做 OCR 提取，不做任何业务过滤或布局判断。**

## 2. 功能需求 (FR)

- **FR-1**: 识别图片中的所有文本，返回 `List[OCRTextElement]`。
- **FR-2**: 结果按 `center.y` 升序排列（从上到下）。
- **FR-3**: 坐标转换：Vision 使用左下角原点、归一化坐标 → 转换为左上角原点像素坐标。
- **FR-4**: 提供 `image_width` / `image_height` 属性供 Layout 层使用。

## 3. 非功能需求 (NFR)

- **NFR-1**: 识别语言默认为 `zh-Hans`。
- **NFR-2**: 使用 `VNRequestTextRecognitionLevelAccurate` 级别。
- **NFR-3**: 启用语言校正 `usesLanguageCorrection=True`。

## 4. 接口契约

### 输入
```python
VisionOCREngine(language: str = "zh-Hans")
engine.recognize(image_path: str) -> List[OCRElement]
```

### 输出
`List[OCRElement]`，每个元素包含：
- `text: str`
- `bbox: Rect`（像素坐标，左上角原点）
- `center: Point`
- `confidence: float`

## 5. 核心规则与约束

### 规则 1: 不修改文本内容
OCR 层不得清洗、过滤、截断识别到的文本。所有文本清洗由 Layout 层或感知层负责。

### 规则 2: 坐标系转换必须精确
```
x = vx * image_width
y = (1.0 - vy - vh) * image_height
```
任何坐标错误都会导致 Layout 层区域划分失败。

## 6. 错误处理

| 异常 | 触发条件 | 处理 |
|------|---------|------|
| `FileNotFoundError` | 图片路径不存在 | 抛给调用方 |
| 加载失败 / 请求失败 | CGImage 或 Vision 请求失败 | 返回空列表，记录 warning |

## 7. 依赖关系
- 依赖 `src.models.base.OCRTextElement, Point, Rect`
- 依赖 macOS Vision / Quartz / AppKit 框架
