import base64
import glob
import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

MODEL = None
MODEL_PATH = None
MODEL_ERROR = None


def discover_model():
    preferred = [
        "object_detection/weights/best.pt",
        "object_detection/weights/last.pt",
    ]
    for p in preferred:
        if Path(p).is_file():
            return p

    candidates = glob.glob("**/*.pt", recursive=True)
    if not candidates:
        return None

    candidates.sort(key=lambda p: (Path(p).name != "best.pt", len(p), p))
    return candidates[0]


def get_model():
    global MODEL, MODEL_PATH, MODEL_ERROR
    if MODEL is not None:
        return MODEL

    MODEL_PATH = discover_model()
    if not MODEL_PATH:
        MODEL_ERROR = "Nem találtam .pt YOLO modellt a repositoryban."
        raise RuntimeError(MODEL_ERROR)

    try:
        MODEL = YOLO(MODEL_PATH)
        return MODEL
    except Exception as exc:
        MODEL_ERROR = f"Modellbetöltési hiba: {exc}"
        raise


PAGE = """
<!doctype html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cucumber Harvesting – YOLO demo</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 18px;background:#f6f7f8;color:#17202a}
    .card{background:white;border-radius:16px;padding:24px;box-shadow:0 3px 18px #0001;margin-bottom:18px}
    h1{margin-top:0}
    input[type=file]{display:block;margin:16px 0}
    button{border:0;border-radius:10px;padding:12px 18px;font-size:16px;cursor:pointer;background:#17202a;color:white}
    img{max-width:100%;border-radius:12px;margin-top:14px}
    .muted{color:#667085}
    .ok{padding:10px 12px;border-radius:10px;background:#eef7ee}
    .err{padding:10px 12px;border-radius:10px;background:#fff0f0;color:#9b1c1c}
    code{word-break:break-all}
  </style>
</head>
<body>
<div class="card">
  <h1>🥒 Cucumber Harvesting – YOLO teszt</h1>
  <p>Ez a robot nélküli demo a repositoryban lévő YOLO súlyfájlt használja. Tölts fel egy JPG/PNG képet, és lefuttatja rajta a felismerést.</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/jpeg,image/png,image/webp" required>
    <button type="submit">Felismerés indítása</button>
  </form>
  <p class="muted">Render CPU-n fut, ezért az első felismerés lassabb lehet.</p>
</div>

{% if error %}
<div class="card err"><b>Hiba:</b> {{ error }}</div>
{% endif %}

{% if result %}
<div class="card">
  <div class="ok"><b>Találatok:</b> {{ count }} &nbsp; | &nbsp; <b>Modell:</b> <code>{{ model_path }}</code></div>
  <img src="data:image/jpeg;base64,{{ result }}" alt="YOLO eredmény">
</div>
{% endif %}

<div class="card muted">
  <b>Mit tesztel ez?</b><br>
  A projekt képfelismerő részét. A ROS, RealSense mélységkamera, Interbotix robotkar és AGV vezérlés nincs emulálva ezen a Render demón.
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, error=MODEL_ERROR, result=None)

    upload = request.files.get("image")
    if not upload or not upload.filename:
        return render_template_string(PAGE, error="Válassz ki egy képet.", result=None), 400

    try:
        image = Image.open(upload.stream).convert("RGB")
        model = get_model()
        results = model.predict(
            source=np.asarray(image),
            conf=0.25,
            device="cpu",
            verbose=False,
        )
        r = results[0]
        annotated_bgr = r.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        out = Image.fromarray(annotated_rgb)

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        count = 0 if r.boxes is None else len(r.boxes)
        return render_template_string(
            PAGE,
            error=None,
            result=encoded,
            count=count,
            model_path=MODEL_PATH,
        )
    except Exception as exc:
        return render_template_string(PAGE, error=str(exc), result=None), 500


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_candidate=discover_model(),
        model_loaded=MODEL is not None,
        model_error=MODEL_ERROR,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
