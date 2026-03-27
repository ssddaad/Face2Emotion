# Face2Emotion

基于 YOLO + EfficientNet-B2 的人脸情绪与微表情实时识别系统。

支持两种运行模式：
- **本地可视化**：直接打开摄像头，画面上实时显示情绪标签
- **服务化部署**：FastAPI + Prometheus，适合集成到后端或监控系统

---

## 核心能力

| 模块 | 技术方案 | 说明 |
|------|----------|------|
| 人脸检测 | YOLOv8n-face | 支持 CUDA，多人同时检测 |
| 人脸跟踪 | 质心距离匹配 | 持续 `track_id`，多人不乱 |
| 人脸对齐 | MediaPipe FaceMesh | 眼睛对齐到固定位置，消除头部旋转干扰 |
| 情绪识别 | hsemotion enet_b2_8 | EfficientNet-B2，AffectNet-8 训练，ONNX 推理 |
| 时序平滑 | 逐人头 EMA | 概率分布级别平滑，标签不乱跳 |
| 微表情 | 像素帧差 + EMA | 量化面部细微运动强度 |
| 服务监控 | Prometheus | FPS、延迟、错误数全量指标 |

---

## 安装

### 基础依赖

```bash
pip install -r requirements.txt
```

### CUDA 加速（推荐）

YOLO 推理支持 GPU，装对版本的 PyTorch 效果明显：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

> 首次运行时 hsemotion 会自动下载 ONNX 权重（约 30MB），之后缓存在本地，不再重复下载。

---

## 运行

### 本地可视化模式

```bash
python main.py
```

常用参数：

```bash
python main.py --yolo-device cuda:0   # 指定 GPU
python main.py --camera 1             # 切换摄像头编号
python main.py --conf 0.5             # 提高检测置信度阈值
python main.py --hide-fps             # 隐藏 FPS 显示
```

按 `Q` 退出。

### 服务化模式

```bash
python service_main.py
```

默认监听 `0.0.0.0:8000`，可通过 `settings.yaml` 或环境变量调整。

#### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/realtime` | 获取最新人脸推理结果快照 |
| POST | `/v1/engine/start` | 启动推理引擎 |
| POST | `/v1/engine/stop` | 停止推理引擎 |
| POST | `/v1/engine/restart` | 重启推理引擎 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/docs` | Swagger 文档 |

> 设置 `api_key` 后，`/v1/*` 接口需在请求头携带 `x-api-key`。

---

## 配置

优先级：**环境变量 > settings.yaml > 内置默认值**

```yaml
# settings.yaml
host: 0.0.0.0
port: 8000
log_level: info

# 输入源
source_type: camera        # camera / rtsp / file
source_value: "0"          # 摄像头编号 / RTSP 地址 / 文件路径
mirror_input: true         # 是否镜像（摄像头场景建议开启）

# YOLO 检测
confidence: 0.4
iou: 0.5
image_size: 640
model_path: models/yolov8n-face.pt
yolo_device: cuda:0        # YOLO 推理设备，不可用时自动回退 CPU
min_face_size: 40          # 低于此尺寸的检测框会被丢弃

# 跟踪与情绪
max_track_distance: 80.0   # 帧间人脸中心点最大匹配距离（像素）
emotion_interval: 3        # 每隔几帧做一次情绪推理（降低 CPU 压力）
micro_ema_alpha: 0.35      # 微表情 EMA 平滑系数

# 引擎行为
max_fps: 20.0
stale_timeout_sec: 2.5     # 多久没有新帧判定为陈旧状态
reconnect_cooldown_sec: 1.0
api_key: ""
```

---

## Prometheus 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `f2e_frames_total` | Counter | 处理总帧数 |
| `f2e_camera_errors_total` | Counter | 摄像头读取错误次数 |
| `f2e_inference_errors_total` | Counter | 推理异常次数 |
| `f2e_processed_faces_total` | Counter | 处理总人脸数 |
| `f2e_inference_latency_seconds` | Histogram | 单帧推理耗时 |
| `f2e_realtime_fps` | Gauge | 当前实时帧率 |
| `f2e_faces_detected` | Gauge | 当前帧检测到的人脸数 |
| `f2e_engine_running` | Gauge | 引擎是否在运行（0/1）|
| `f2e_engine_stale` | Gauge | 引擎是否陈旧（0/1）|

---

## 项目结构

```
Face2Emotion/
├── main.py                  # 本地可视化入口
├── service_main.py          # 服务化入口
├── settings.yaml            # 服务化默认配置
├── models/
│   └── yolov8n-face.pt      # YOLO 人脸检测模型
└── face2emotion/
    ├── emotion.py           # 情绪识别（hsemotion + MediaPipe 对齐 + EMA）
    ├── detector.py          # YOLO 人脸检测封装
    ├── tracker.py           # 多人脸跟踪（质心匹配）
    ├── micro_expression.py  # 微表情帧差计算
    ├── renderer.py          # OpenCV 画框渲染
    ├── app.py               # 本地模式主循环
    ├── service_engine.py    # 后台推理引擎（独立线程）
    ├── service_api.py       # FastAPI 路由
    ├── service_config.py    # 服务配置加载
    ├── config.py            # 本地模式配置
    ├── schema.py            # 数据结构定义
    ├── metrics.py           # Prometheus 指标定义
    ├── model_store.py       # 模型文件管理
    └── logging_utils.py     # 日志初始化
```
