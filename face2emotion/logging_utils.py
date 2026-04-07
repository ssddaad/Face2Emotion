import logging
import os



def setup_logger(name: str="face2emotion") -> logging.Logger:
    # 初始化并返回项目日志器
    # 日志级别优先读环境变量 F2E_LOG_LEVEL，没设就用 INFO
    # 重复调用安全：已有 handler 不再重复添加
    logger=logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name=os.getenv("F2E_LOG_LEVEL", "INFO").upper()
    level=getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    handler=logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    # 不向 root logger 传播，避免被 uvicorn/其他库的 handler 重复处理
    logger.propagate=False
    return logger
