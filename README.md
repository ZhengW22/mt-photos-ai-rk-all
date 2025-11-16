# MT Photos AI RKNN 一体化服务

本项目是把 `mt-photos-ai-rk`（OCR + CLI/CLIP）和 `mt-photos-insightface-rk`（InspireFace 人脸识别）这两个仓库合并，将三个服务统一到一个项目中，运行一个容器即可获得所有能力。

## 功能概览

| 能力          | 对应 API                                 | 说明                                                 |
| ------------- | ---------------------------------------- | ---------------------------------------------------- |
| 检查服务状态  | `POST /check`                          | 返回 OCR/CLIP/人脸识别配置与版本信息                 |
| OCR 文本识别  | `POST /ocr`                            | PaddleOCR RKNN 推理，返回文本、置信度和坐标          |
| CLIP 图片向量 | `POST /clip/img`                       | Chinese-CLIP RKNN 图片分支                           |
| CLIP 文本向量 | `POST /clip/txt`                       | Chinese-CLIP RKNN 文本分支                           |
| 人脸特征提取  | `POST /represent`                      | InspireFace Gundam_RK3588，会返回 embedding 与检测框 |
| 释放资源      | `POST /restart` / `POST /restart_v2` | 释放资源                                             |

所有请求都需要在 Header 中携带 `api-key: ${API_AUTH_KEY}`。

## 目录说明

| 目录/文件                                           | 说明                                                                        |
| --------------------------------------------------- | --------------------------------------------------------------------------- |
| `rknn/`                                           | 统一 FastAPI 服务源码、Dockerfile、RKNN 模型及脚本                          |
| `rknn/models/`                                    | OCR 检测/识别模型以及字典文件                                               |
| `rknn/ocr/`                                       | PaddleOCR RKNN 推理逻辑（从 `mt-photos-ai-rk` 继承）                      |
| `rknn/utils/`、`clip.py` 等                     | CLIP 推理相关代码与模型                                                     |
| `InspireFace/`                                    | InspireFace Gundam_RK3588 模型缓存，构建时复制到 `/root/.inspireface/...` |
| `inspireface-1.2.3-cp310-cp310-linux_aarch64.whl` | 定制 InspireFace SDK，支持核心分配                                          |
| `rknn-toolkit-lite2/`                             | RKNN Toolkit Lite2 (cp310) 安装包                                           |

## 环境变量

| 变量                    | 默认值                 | 说明                                                           |
| ----------------------- | ---------------------- | -------------------------------------------------------------- |
| `API_AUTH_KEY`        | `mt_photos_ai_extra` | 所有 API 访问时需要在 Header 中传入 `api-key`                |
| `HTTP_PORT`           | `8060`               | FastAPI 服务监听端口                                           |
| `SERVER_RESTART_TIME` | `300`                | (秒) 当容器长时间没有请求时自动重启整个进程，设为 `0` 可关闭 |

## 打包镜像

1. 切换到仓库根目录（与 `rknn/` 同级）
2. 执行（在 ARM64 机器上运行）：
   ```bash
   docker build -t mt-photos-ai-all:rknn -f rknn/Dockerfile .
   ```

> 注意需要先去 `https://github.com/a15355447898a/mt-photos-ai-rk/releases/tag/0.0`把CLIP RKNN 模型 (`vit-b-16.img.fp32.rknn`, `vit-b-16.txt.fp32.rknn`)下载并放置到 `rknn/utils/`目录下

## 运行容器

### docker-compose

```yaml
version: '3.8'
services:
  mt-photos-ai-all:
    image: mt-photos-ai-all:rknn
    build:
      context: .
      dockerfile: rknn/Dockerfile
    container_name: mt-photos-ai-all
    hostname: mt-photos-ai-all
    privileged: true
    devices:
      - /dev/dri:/dev/dri
    volumes:
      - /proc/device-tree/compatible:/proc/device-tree/compatible
      - /usr/lib/librknnrt.so:/usr/lib/librknnrt.so
    environment:
      - API_AUTH_KEY=your_secret_key
    ports:
      - "8060:8060"
    restart: always
```

> 也可以用我预先构建的镜像 `a15355447898a/mt-photos-ai-rk-all:latest`

## API 说明

### /check

检测服务是否可用，及api-key是否正确。

```bash
curl --location --request POST 'http://127.0.0.1:8060/check' \
--header 'api-key: your_api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /ocr

文字识别。

```bash
curl --location --request POST 'http://127.0.0.1:8060/ocr' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- result.texts : 识别到的文本列表
- result.scores : 为识别到的文本对应的置信度分数，1为100%
- result.boxes : 识别到的文本位置，x,y为左上角坐标，width,height为框的宽高

```json
{
  "result": {
    "texts": [
      "识别到的文本1",
      "识别到的文本2"
    ],
    "scores": [
      "0.98",
      "0.97"
    ],
    "boxes": [
      {
        "x": "4.0",
        "y": "7.0",
        "width": "283.0",
        "height": "21.0"
      },
      {
        "x": "7.0",
        "y": "34.0",
        "width": "157.0",
        "height": "23.0"
      }
    ]
  }
}
```

### /clip/img

提取图片特征向量。

```bash
curl --location --request POST 'http://127.0.0.1:8060/clip/img' \
--header 'api-key: your_api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- results : 图片的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /clip/txt

提取文本特征向量。

```bash
curl --location --request POST 'http://127.0.0.1:8060/clip/txt' \
--header "Content-Type: application/json" \
--header 'api-key: your_api_key' \
--data '{"text":"飞机"}'
```

**response:**

- results : 文字的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /represent

```bash
curl --location --request POST 'http://127.0.0.1:8066/represent' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- detector_backend : 人脸检测模型
- recognition_model : 人脸特征提取模型
- result : 识别到的结果

```json
{
  "detector_backend": "insightface",
  "recognition_model": "buffalo_l",
  "result": [
    {
      "embedding": [ 0.5760641694068909,... 512位向量 ],
      "facial_area": {
        "x": 212,
        "y": 112,
        "w": 179,
        "h": 250,
        "left_eye": [ 271, 201 ],
        "right_eye": [ 354, 205 ]
      },
      "face_confidence": 1.0
    }
  ]
}
```

### /restart_v2

通过重启进程来释放内存。

```bash
curl --location --request POST 'http://127.0.0.1:8060/restart_v2' \
--header 'api-key: your_api_key'
```

**response:**

请求中断,没有返回，因为服务重启了。
