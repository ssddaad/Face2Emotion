from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .logging_utils import setup_logger
from .service_config import load_service_config
from .service_engine import RealtimeInferenceEngine

# 模块级只做声明，不做有副作用的初始化。
# 真正的初始化放在 lifespan 里，这样测试时可以 mock，
# 重启 uvicorn worker 也不会留下僵尸引擎线程。
_logger = setup_logger()
_config = load_service_config()
_engine = RealtimeInferenceEngine(_config, _logger)


def _check_key(x_api_key: str | None) -> None:
    """有配置 api_key 时才校验，没配置就完全开放。"""
    if _config.api_key and x_api_key != _config.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _engine.start()
    yield
    _engine.stop()


app = FastAPI(
    title="Face2Emotion Service",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """健康检查。

    - 200 ok：引擎运行正常
    - 200 degraded：引擎在跑但数据陈旧（视频源卡住）
    - 503：引擎未运行
    """
    state = _engine.snapshot()

    if not state["running"]:
        # 引擎完全停了，返回 503 让上层负载均衡摘掉这个节点
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "down",
                "last_error": state["last_error"],
            },
        )

    service_status = "degraded" if state["stale"] else "ok"
    return {
        "status":         service_status,
        "engine_running": state["running"],
        "stale":          state["stale"],
        "last_frame_at":  state["last_frame_at"],
        "uptime_sec":     state["uptime_sec"],
        "fps":            state["fps"],
        "last_error":     state["last_error"],
        "source_type":    state["source_type"],
        "source_value":   state["source_value"],
    }


@app.get("/v1/realtime")
def realtime_result(x_api_key: str | None = Header(default=None)):
    """获取最新一帧的推理结果快照。"""
    _check_key(x_api_key)
    state = _engine.snapshot()
    if not state["running"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="engine_not_running")
    return state


@app.post("/v1/engine/start")
def start_engine(x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    _engine.start()
    return {"ok": True, "message": "engine_started"}


@app.post("/v1/engine/stop")
def stop_engine(x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    _engine.stop()
    return {"ok": True, "message": "engine_stopped"}


@app.post("/v1/engine/restart")
def restart_engine(x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    _engine.restart()
    return {"ok": True, "message": "engine_restarted"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus 指标端点，不计入 Swagger 文档。"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
