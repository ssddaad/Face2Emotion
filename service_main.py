import argparse
import os
import rtx50_compat
import uvicorn
from face2emotion.service_config import load_service_config


def main() -> None:
    parser=argparse.ArgumentParser(description="Face2Emotion FastAPI 服务")
    parser.add_argument("--config", type=str, default="settings.yaml", help="配置文件路径")
    args=parser.parse_args()
    os.environ["F2E_CONFIG_PATH"]=args.config
    cfg=load_service_config(args.config)
    uvicorn.run(
        "face2emotion.service_api:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
    )


if __name__=="__main__":
    main()
