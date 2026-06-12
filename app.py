# app.py
import io, os, requests
from typing import Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import onnxruntime as ort

EMOTION_CLASSES = ['angry', 'fear', 'happy', 'sad', 'sleepy', 'surprised']

API_KEY     = os.getenv("PLUGIN3101CARSOF", "")            # opcional
MODEL_PATH  = os.getenv("MODEL_PATH", "models/emociones_resnet50_2.onnx")
MODEL_URL   = os.getenv("MODEL_URL", "")                  # opcional (descarga en arranque)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")       # lista separada por comas
PORT = int(os.getenv("PORT", "8000"))

MEAN_CAFFE_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)  # B, G, R

app = FastAPI(title="Emotion Inference (Render)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _download_model_if_needed():
    if MODEL_URL and not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

def _load_session():
    providers = ort.get_available_providers()  # CPU en Render free
    return ort.InferenceSession(MODEL_PATH, providers=providers)

def _infer_layout(sess) -> Tuple[bool, Tuple[int,int,int,int]]:
    """Devuelve (is_nchw, shape) donde shape es (N, C/H, H/W, W/C) según corresponda."""
    ishape = sess.get_inputs()[0].shape  # p.ej. [1,224,224,3] o [1,3,224,224]
    # normalizar None/-1
    shp = [int(s) if isinstance(s, (int, np.integer)) else -1 for s in ishape]
    if len(shp) != 4:
        raise HTTPException(500, detail=f"Entrada ONNX no es 4D: shape={ishape}")
    n, d1, d2, d3 = shp
    # heurística: si última dim es 1/3 -> NHWC; si segunda es 1/3 -> NCHW
    if d3 in (1,3):
        return (False, tuple(shp))  # NHWC
    if d1 in (1,3):
        return (True, tuple(shp))   # NCHW
    # fallback: asumir NHWC 3 canales
    return (False, (n, d1 if d1>0 else 224, d2 if d2>0 else 224, d3 if d3>0 else 3))

def _preprocess_caffe(image: Image.Image, is_nchw: bool, target_hw=(224,224), channels=3) -> np.ndarray:
    """Preprocesado compatible con tu frontend:
       - redimensiona a 224x224 (o lo que pida el modelo),
       - convierte a RGB -> reordena a BGR,
       - resta MEAN_CAFFE_BGR,
       - entrega float32 sin escalar (0..255) más el -mean,
       - layout NHWC o NCHW según modelo.
    """
    h, w = target_hw
    img = image.convert("RGB").resize((w, h))
    arr = np.asarray(img, dtype=np.float32)  # H,W,3 en RGB
    # RGB -> BGR
    arr = arr[..., ::-1]  # ahora H,W,3 en BGR
    # restar mean por canal BGR
    arr -= MEAN_CAFFE_BGR
    if channels == 1:
        # si el modelo fuera monocanal, promediar (poco probable en tu caso)
        arr = arr.mean(axis=-1, keepdims=True)
    if is_nchw:
        arr = np.transpose(arr, (2,0,1))  # C,H,W
        arr = arr[None, ...]              # N,C,H,W
    else:
        arr = arr[None, ...]              # N,H,W,C
    return arr.astype(np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# Arranque
_download_model_if_needed()
session = _load_session()
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
IS_NCHW, ISHAPE = _infer_layout(session)

# deducir H,W,C esperados (rellenar -1 si aparecen)
if IS_NCHW:
    _, c, h, w = ISHAPE
else:
    _, h, w, c = ISHAPE
h = 224 if h in (-1,0) else h
w = 224 if w in (-1,0) else w
c = 3   if c in (-1,0) else c
TARGET_HW = (h, w)
CHANNELS  = c

class EmotionResponse(BaseModel):
    label: str
    probs: Dict[str, float]
    topk: list

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_PATH),
        "input": {"name": INPUT_NAME, "shape": ISHAPE, "layout": "NCHW" if IS_NCHW else "NHWC"},
        "classes": EMOTION_CLASSES
    }

@app.post("/emotion", response_model=EmotionResponse)
async def infer_emotion(image: UploadFile = File(...), x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    content = await image.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    x = _preprocess_caffe(img, IS_NCHW, target_hw=TARGET_HW, channels=CHANNELS)
    out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0].squeeze()

    # convertir a probabilidades si viene como logits
    if not (np.all((out >= 0) & (out <= 1)) and abs(out.sum() - 1) < 1e-3):
        probs = _softmax(out)
    else:
        probs = out

    probs = probs.tolist()
    # mapear a tus clases (mismo orden de salida)
    mapping = { EMOTION_CLASSES[i]: float(probs[i]) for i in range(min(len(EMOTION_CLASSES), len(probs))) }
    label = max(mapping.items(), key=lambda kv: kv[1])[0]
    topk  = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:2]

    return EmotionResponse(label=label, probs=mapping, topk=topk)
