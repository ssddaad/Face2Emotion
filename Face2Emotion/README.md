# Face2Emotion

基于 YOLO + EfficientNet-B2 的人脸情绪、微表情与全身动作捕捉实时识别系统。

支持两种运行模式：
- **本地可视化**：直接打开摄像头，画面上实时显示情绪标签与骨骼覆盖层
- **服务化部署**：FastAPI + Prometheus，适合集成到后端或监控系统

---

## 核心能力

| 模块 | 技术方案 | 说明 |
|------|----------|------|
| 人脸检测 | YOLOv8n-face | 支持 CUDA，多人同时检测 |
| 人脸跟踪 | 质心+尺度联合匹配 | 持续 `track_id`，多人交叉/遮挡下更稳 |
| 人脸对齐 | MediaPipe FaceLandmarker | 468 点关键点，眼部仿射对齐消除旋转干扰 |
| 情绪识别 | hsemotion enet_b2_8 | EfficientNet-B2，AffectNet-8 训练，ONNX 推理，7 类输出 |
| 时序平滑 | 逐人头 EMA | 概率分布级别平滑，标签不乱跳 |
| 微表情 | 增强 LK 光流（低位移增强）+ 帧差高分位兜底 + EMA | 更早捕捉细微变化，量化到 0–100 |
| 精神面貌 | 面色活力 + 眼部聚焦 + PERCLOS + 眨眼率 + 面部能量融合评分 | 输出 `mental_state` 分数、趋势与风险等级 |
| 全身姿态 | MediaPipe PoseLandmarker + 关键点 EMA | 33 点骨骼，肩部以下可视化，抑制抖动 |
| 手部捕捉 | MediaPipe HandLandmarker | 双手 21 点关键点 + 手势识别 |
| 动作分类 | 自适应时序投票分类器 | 基于关节角度序列，动态门限投票，减少误判 |
| 服务监控 | Prometheus | FPS、延迟、错误数、姿态/动作/手势全量指标 |

---

## 最近更新（2026-04）

- 微表情检测：优化 LK 光流参数、加入相对位移去全局平移、局部 ROI 加权（眉眼嘴）与非线性评分，对表情早期（onset）更敏感。
- 动作捕捉：新增 `motion_interval`，支持按帧降频推理，降低延迟。
- 姿态稳定性：新增 `motion_landmark_ema_alpha`，减少关键点抖动。
- 动作分类：阈值与投票策略优化，提升动作标签与真实行为一致性。
- 跟踪稳定性：人脸跟踪升级为常速度预测（简版 Kalman 思路）+ 距离/尺度联合匹配，减少多人场景 ID 串扰。

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

### 动作捕捉模型（可选）

启用 `--motion-capture` 前需将以下模型文件放入 `models/` 目录：

| 文件 | 来源 |
|------|------|
| `pose_landmarker_lite.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| `hand_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |
| `face_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |

> 首次运行时 hsemotion 会自动下载 ONNX 权重（约 30MB），之后缓存在本地，不再重复下载。

---

## 运行

### 本地可视化模式

```bash
python main.py
```

常用参数：

```bash
python main.py --yolo-device cuda:0       # 指定 GPU
python main.py --camera 1                 # 切换摄像头编号
python main.py --conf 0.5                 # 提高检测置信度阈值
python main.py --hide-fps                 # 隐藏 FPS 显示
python main.py --motion-capture           # 启用全身动作捕捉
python main.py --micro-roi-brow 1.3 --micro-roi-eye 1.6 --micro-roi-mouth 1.8  # 调整眉/眼/嘴 ROI 权重
python main.py --motion-capture --motion-complexity 2   # 高精度姿态模型
python main.py --motion-capture --motion-interval 2     # 动作捕捉降频，降低追踪延迟
python main.py --motion-capture --motion-lm-alpha 0.45  # 姿态关键点 EMA 平滑系数
python main.py --hide-skeleton            # 隐藏骨骼线条
python main.py --hide-hands               # 隐藏手部关键点
python main.py --hide-action              # 隐藏动作标签
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

`/v1/realtime` 返回结构示例：

```json
{
  "faces": [
    {
      "track_id": 1,
      "bbox": { "x1": 120, "y1": 80, "x2": 240, "y2": 220 },
      "emotion": { "label": "happy", "score": 0.87 },
      "micro_expression": { "score": 4.2, "level": "Low" },
      "mental_state": {
        "score": 68.4,
        "level": "Good",
        "color_vitality": 66.1,
        "eye_focus": 70.9,
        "facial_energy": 67.8,
        "trend_score": 65.7,
        "risk_level": "Low Risk",
        "perclos": 0.0833,
        "blink_rate": 17.5
      },
      "motion_capture": {
        "pose": { "action_label": "idle", "action_conf": 0.91 },
        "left_hand": { "gesture": "open" },
        "right_hand": null
      }
    }
  ]
}
```

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
micro_roi_brow_weight: 1.25
micro_roi_eye_weight: 1.45
micro_roi_mouth_weight: 1.60

# 全身动作捕捉（可选）
enable_motion_capture: false
motion_complexity: 1            # 姿态模型复杂度 0(快) / 1(标准) / 2(精准)
motion_vote_window: 10          # 动作分类时序投票窗口帧数
motion_interval: 2              # 每隔 N 帧执行一次动作捕捉（降低延迟）
motion_landmark_ema_alpha: 0.45 # 姿态关键点 EMA 平滑系数

# 引擎行为
max_fps: 20.0
stale_timeout_sec: 2.5      # 多久没有新帧判定为陈旧状态
reconnect_cooldown_sec: 1.0
api_key: ""
```

### 服务化环境变量（新增）

- `F2E_MICRO_ROI_BROW`：眉区 ROI 权重（默认 `1.25`）。
- `F2E_MICRO_ROI_EYE`：眼区 ROI 权重（默认 `1.45`）。
- `F2E_MICRO_ROI_MOUTH`：嘴区 ROI 权重（默认 `1.60`）。
- `F2E_MOTION_INTERVAL`：动作捕捉降频间隔（默认 `1`）。
- `F2E_MOTION_LM_ALPHA`：姿态关键点平滑系数（默认 `0.45`）。

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
| `f2e_engine_running` | Gauge | 引擎是否在运行（0/1） |
| `f2e_engine_stale` | Gauge | 引擎是否陈旧（0/1） |
| `f2e_motion_capture_errors_total` | Counter | 动作捕捉异常次数 |
| `f2e_motion_capture_latency_seconds` | Histogram | 动作捕捉单帧耗时 |
| `f2e_pose_detected` | Gauge | 当前帧是否检测到姿态（0/1） |
| `f2e_actions_classified` | Counter | 动作分类次数（按标签） |
| `f2e_gestures_classified` | Counter | 手势识别次数（按手势/手侧） |

---

## 项目结构

```text
Face2Emotion/
├── main.py                  # 本地可视化入口
├── service_main.py          # 服务化入口
├── settings.yaml            # 服务化默认配置
├── requirements.txt
├── models/
│   ├── yolov8n-face.pt
│   ├── face_landmarker.task
│   ├── pose_landmarker_lite.task
│   └── hand_landmarker.task
└── face2emotion/
    ├── emotion.py           # 情绪识别（hsemotion + 对齐 + EMA）
    ├── detector.py          # YOLO 人脸检测封装
    ├── tracker.py           # 多人脸跟踪（质心匹配）
    ├── micro_expression.py  # 微表情增强 LK 光流计算
    ├── motion_capture.py    # 全身姿态 + 手部捕捉引擎
    ├── action_classifier.py # 关节角度时序动作分类
    ├── gesture.py           # 手势规则识别
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
