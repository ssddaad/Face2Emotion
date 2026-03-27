from prometheus_client import Counter, Gauge, Histogram

frames_total = Counter("f2e_frames_total", "Total captured frames")
camera_errors_total = Counter("f2e_camera_errors_total", "Total camera read errors")
inference_errors_total = Counter("f2e_inference_errors_total", "Total inference errors")
processed_faces_total = Counter("f2e_processed_faces_total", "Total processed faces")

inference_latency_seconds = Histogram(
    "f2e_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28),
)

faces_detected = Gauge("f2e_faces_detected", "Detected faces in latest frame")
realtime_fps = Gauge("f2e_realtime_fps", "Realtime FPS")
engine_running = Gauge("f2e_engine_running", "Engine running state")
engine_stale = Gauge("f2e_engine_stale", "Engine stale state")
