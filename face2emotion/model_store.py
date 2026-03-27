import urllib.request
from pathlib import Path

# 两个备用下载源，第一个挂了自动试第二个
MODEL_URLS = [
    "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
]

_DOWNLOAD_TIMEOUT = 60  # 秒，超时就放弃这个源试下一个，别让程序卡死


def ensure_face_model(model_path: Path, logger) -> None:
    """确保 YOLO 人脸模型文件存在，不存在时从备用源自动下载。

    下载失败会抛 FileNotFoundError，调用方应在启动时尽早调用，
    避免推理线程跑起来之后才发现模型缺失。
    """
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("模型文件不存在，开始自动下载...")

    for url in MODEL_URLS:
        try:
            logger.info("尝试下载: %s", url)
            # 加超时，网络差的环境不会永远卡在这里
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "face2emotion/1.0")]
            urllib.request.install_opener(opener)

            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
                data = resp.read()
            model_path.write_bytes(data)
            logger.info("模型下载成功: %s", model_path)
            return
        except Exception as err:
            logger.warning("下载失败 (%s): %s", url, err)
            # 把可能写了一半的文件清掉，别留着污染下次启动
            if model_path.exists():
                model_path.unlink(missing_ok=True)

    raise FileNotFoundError(
        f"所有下载源均失败，请手动下载 yolov8n-face.pt 放到 {model_path}"
    )
