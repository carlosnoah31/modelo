# DL Emotion API (Render)

API FastAPI para inferencia con un modelo ONNX (~46 MB). Carga única del modelo en el servidor y endpoint `/infer` para clasificar rostros 224x224.

## Estructura
```
dl-api-render/
├─ app.py
├─ requirements.txt
├─ render.yaml        # opcional (provisiona servicio en Render)
└─ models/
   └─ model.onnx      # coloca tu modelo aquí (o usa MODEL_URL)
```

## Deploy en Render (UI)
1. Sube este folder a **GitHub** como repositorio privado o público.
2. Ve a https://render.com → **New** → **Web Service** → conecta tu repo.
3. Configura:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Plan **Free** (autosleep).
4. Si **NO** subes `models/model.onnx` al repo, define variable de entorno **MODEL_URL** con un enlace público directo al `.onnx`.
5. Deploy. Tu URL será algo como `https://dl-emotion-api.onrender.com`.

## Probar
```
curl -s https://dl-emotion-api.onrender.com/healthz
curl -s -X POST https://dl-emotion-api.onrender.com/infer -F "file=@face.jpg"
```

## Consumir desde el frontend
```js
const form = new FormData();
form.append("file", blobOrFile);
const res = await fetch("https://dl-emotion-api.onrender.com/infer", { method: "POST", body: form });
const data = await res.json(); // {label, confidence, probs, classes}
```

## Notas
- Free tier duerme al estar inactivo; la primera llamada puede tardar (cold start).
- Este backend reescala internamente a 224×224.
- Si tu modelo exige NCHW en lugar de NHWC, el backend ya prueba ambas rutas.
- Si recibes error de entrada, revisa el **nombre del input** esperado por el grafo (`/healthz` muestra `input_name`). Cambia `inp_name` si tu modelo tiene múltiples entradas.
- Para producción, limita `allow_origins` en CORS a tu dominio.
