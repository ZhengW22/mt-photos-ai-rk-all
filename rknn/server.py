from dotenv import load_dotenv
import os
import sys
import threading
import queue
import logging
from typing import Dict
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import clip as clip
from ocr import TextSystem
from rknnlite.api import RKNNLite
import inspireface as isf

# import onnxruntime as ort
# device = ort.get_device()
# print(f"Using device: {device}")

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = 8060
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
# env_use_dml = os.getenv("MT_USE_DML", "on") == "on" # 是否启用dml加速，当使用onnxruntime-directml加速时，使用这行
env_use_dml = False
env_auto_load_txt_modal = True  # 默认在启动时加载 CLIP 文本模型，避免首次请求冷启动

logging.basicConfig(level=logging.INFO)

ocr_models = queue.Queue()
clip_img_models = queue.Queue()
clip_txt_models = queue.Queue()
restart_timer = None

face_workers = 3
face_detector_backend = os.getenv("FACE_DETECTOR_BACKEND", "inspireface_rknn_gundam")
face_recognition_model = os.getenv("FACE_RECOGNITION_MODEL", "Gundam_RK3588")
facial_min_score = float(os.getenv("FACE_MIN_SCORE", "0.5"))
facial_max_distance = float(os.getenv("FACE_MAX_DISTANCE", "0.5"))
face_task_queue: "queue.Queue[tuple[str, bytes, str | None]]" = queue.Queue()
face_results: Dict[str, Dict] = {}
face_results_lock = threading.Lock()
face_worker_threads = []
MAX_IMAGE_SIDE = 10000


class LazyModelSlot:
    """延迟加载模型，避免启动时占用额外内存，同时保持队列可用。"""

    def __init__(self, factory, preload=False):
        self._factory = factory
        self._model = None
        self._lock = threading.Lock()
        if preload:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._factory()
        return self._model

    def get_model(self):
        return self.ensure_loaded()

# RKNN OCR model paths
DET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocrv4_det.rknn')
REC_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocrv4_rec.rknn')
CHARACTER_DICT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppocr_keys_v1.txt')
RKNN_TARGET = "rk3588"

class ClipTxtRequest(BaseModel):
    text: str

class FaceWorker(threading.Thread):
    def __init__(self, worker_idx: int):
        super().__init__(daemon=True)
        self.worker_idx = worker_idx
        self.face_session = None
        self.core_mask = self._build_core_mask(worker_idx)

    @staticmethod
    def _build_core_mask(worker_idx: int) -> int:
        # RK3588 提供 3 个 NPU 核心，超出部分重复使用
        mapped_idx = worker_idx % 3
        return 1 << mapped_idx

    def run(self):
        self._init_model()
        self._process_tasks()

    def _init_model(self):
        try:
            if self.core_mask:
                isf.set_rknn_core_mask(self.core_mask)
            isf.reload(face_recognition_model)
            opt = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_RECOGNITION
            self.face_session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)
            logging.info("Face worker %s ready (core_mask=0x%x)", self.worker_idx, self.core_mask)
        except Exception as exc:
            logging.exception("Face worker %s failed to init: %s", self.worker_idx, exc)
            raise

    def _process_tasks(self):
        while True:
            task_id, image_bytes, content_type = face_task_queue.get()
            try:
                payload = self._handle_task(image_bytes, content_type)
            except Exception as exc:
                logging.exception("Face worker error on task %s: %s", task_id, exc)
                payload = {'result': [], 'msg': str(exc)}
            finally:
                with face_results_lock:
                    face_results[task_id] = payload
                face_task_queue.task_done()

    def _handle_task(self, image_bytes: bytes, content_type: str | None) -> Dict:
        if not self.face_session:
            raise RuntimeError("Face session not ready")
        img = self._preprocess_image(image_bytes, content_type)
        if isinstance(img, str):
            return {'result': [], 'msg': img}
        embedding_objs = self._represent(img)
        return {
            "detector_backend": face_detector_backend,
            "recognition_model": face_recognition_model,
            "result": embedding_objs
        }

    def _preprocess_image(self, image_bytes: bytes, content_type: str | None):
        img = None
        if content_type == 'image/gif':
            with Image.open(BytesIO(image_bytes)) as pil_img:
                if getattr(pil_img, "is_animated", False):
                    pil_img.seek(0)
                frame = pil_img.convert('RGB')
                img = np.array(frame)
        if img is None:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            return "Invalid or corrupted image"

        h, w, _ = img.shape
        if h > MAX_IMAGE_SIDE or w > MAX_IMAGE_SIDE:
            return 'Image size too large'
        return img

    def _represent(self, img):
        faces = self.face_session.face_detection(img)
        results = []
        for face in faces:
            feature = self.face_session.face_feature_extract(img, face)
            if feature is not None:
                res = {
                    "embedding": feature.tolist(),
                    "facial_area": {
                        "x": int(face.location[0]),
                        "y": int(face.location[1]),
                        "w": int(face.location[2] - face.location[0]),
                        "h": int(face.location[3] - face.location[1])
                    },
                    "face_confidence": float(face.detection_confidence)
                }
                results.append(res)
        return results

@app.on_event("startup")
async def startup_event():
    global face_worker_threads
    # Initialize and populate the model queues
    for i in range(3):
        core_mask = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][i]
        
        # OCR model
        ocr_model = TextSystem(
            det_model_path=DET_MODEL_PATH,
            rec_model_path=REC_MODEL_PATH,
            character_dict_path=CHARACTER_DICT_PATH,
            target=RKNN_TARGET,
            drop_score=0.5,
            core_mask=core_mask
        )
        ocr_models.put(ocr_model)
        
        # CLIP image model
        img_model = clip.load_img_model(use_dml=env_use_dml, core_mask=core_mask)
        clip_img_models.put(img_model)
        
        # CLIP text model（懒加载保证队列不为空）
        txt_model_slot = LazyModelSlot(
            lambda mask=core_mask: clip.load_txt_model(use_dml=env_use_dml, core_mask=mask),
            preload=env_auto_load_txt_modal
        )
        clip_txt_models.put(txt_model_slot)

    face_worker_threads = []
    for i in range(face_workers):
        worker = FaceWorker(i)
        worker.start()
        face_worker_threads.append(worker)


@app.middleware("http")
async def check_activity(request, call_next):
    global restart_timer

    if server_restart_time > 0:
        if restart_timer:
            restart_timer.cancel()
        restart_timer = threading.Timer(server_restart_time, restart_program)
        restart_timer.start()

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))


def trans_result(filter_boxes, filter_rec_res):
    texts = []
    scores = []
    boxes = []
    if filter_boxes is None or filter_rec_res is None:
        return {'texts': texts, 'scores': scores, 'boxes': boxes}

    for dt_box, rec_result in zip(filter_boxes, filter_rec_res):
        text, score = rec_result[0]
        box = {
            'x': to_fixed(dt_box[0][0]),
            'y': to_fixed(dt_box[0][1]),
            'width': to_fixed(dt_box[1][0] - dt_box[0][0]),
            'height': to_fixed(dt_box[2][1] - dt_box[0][1])
        }
        boxes.append(box)
        texts.append(text)
        scores.append(f"{score:.2f}")
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>OCR/CLIP 文档： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
<p>人脸识别文档： <a href="https://mtmt.tech/docs/advanced/insightface_api/">https://mtmt.tech/docs/advanced/insightface_api/</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai-all (RKNN)",
        "services": {
            "ocr": {"endpoint": "/ocr", "doc": "https://mtmt.tech/docs/advanced/ocr_api"},
            "clip_img": {"endpoint": "/clip/img", "doc": "https://mtmt.tech/docs/advanced/ocr_api"},
            "clip_txt": {"endpoint": "/clip/txt", "doc": "https://mtmt.tech/docs/advanced/ocr_api"},
            "face": {"endpoint": "/represent", "doc": "https://mtmt.tech/docs/advanced/insightface_api"}
        },
        "env_use_dml": env_use_dml,
        "detector_backend": face_detector_backend,
        "recognition_model": face_recognition_model,
        "facial_min_score": facial_min_score,
        "facial_max_distance": facial_max_distance,
        "face_workers": face_workers
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 客户端可调用，触发重启进程来释放内存，OCR过程中会触发这个请求；新版本OCR内存增长正常了，此方法不执行
    # restart_program()
    return {'result': 'pass'}

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    ocr_model = ocr_models.get()
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            return {'result': [], 'msg': 'height or width out of range'}

        # Run RKNN OCR
        filter_boxes, filter_rec_res = ocr_model.run(img)
        result = trans_result(filter_boxes, filter_rec_res)
        del img
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        ocr_models.put(ocr_model)

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    clip_img_model = clip_img_models.get()
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = await predict(clip.process_image, img, clip_img_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        clip_img_models.put(clip_img_model)

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    clip_txt_slot = clip_txt_models.get()
    try:
        text = request.text
        clip_txt_model = clip_txt_slot.get_model()
        result = await predict(clip.process_txt, text, clip_txt_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}
    finally:
        clip_txt_models.put(clip_txt_slot)

@app.post("/represent")
async def represent_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    task_id = os.urandom(16).hex()
    image_bytes = await file.read()
    face_task_queue.put((task_id, image_bytes, file.content_type))
    while True:
        await asyncio.sleep(0.01)
        with face_results_lock:
            result = face_results.pop(task_id, None)
        if result is not None:
            return result

async def predict(predict_func, inputs,model):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs,model)

def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=http_port)
