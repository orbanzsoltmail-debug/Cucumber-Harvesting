import base64
import glob
import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, YOLOWorld

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

MODEL = None
MODEL_PATH = None
MODEL_ERROR = None
MODEL_SOURCE = None


def discover_model():
    preferred = ["object_detection/weights/best.pt", "object_detection/weights/last.pt"]
    for path in preferred:
        if Path(path).is_file():
            return path
    candidates = glob.glob("**/*.pt", recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (Path(path).name != "best.pt", len(path), path))
    return candidates[0]


def get_model():
    global MODEL, MODEL_PATH, MODEL_ERROR, MODEL_SOURCE
    if MODEL is not None:
        return MODEL
    MODEL_PATH = discover_model()
    try:
        if MODEL_PATH:
            MODEL = YOLO(MODEL_PATH)
            MODEL_SOURCE = "A projekt saját uborkamodellje"
        else:
            # The upstream README lists custom weights, but they are not actually
            # present in the repository. YOLO-World gives us a real, usable
            # open-vocabulary fallback instead of a non-functional demo.
            MODEL_PATH = "yolov8s-worldv2.pt"
            MODEL = YOLOWorld(MODEL_PATH)
            MODEL.set_classes(["cucumber", "green cucumber", "cucumber fruit"])
            MODEL_SOURCE = "YOLO-World nyílt szókészletű gyorsmodell"
        return MODEL
    except Exception as exc:
        MODEL_ERROR = f"Modellbetöltési hiba: {exc}"
        raise


# Load once while the service boots. This avoids a long first photo request and
# prevents Ultralytics from trying to install YOLO-World dependencies mid-request.
try:
    get_model()
except Exception:
    pass


def dashed_line(draw, points, fill, width=4, dash=12):
    for start, end in zip(points[::2], points[1::2]):
        draw.line([start, end], fill=fill, width=width)


def dashed_rectangle(draw, box, fill=(255, 145, 0), width=4, dash=13):
    x1, y1, x2, y2 = [int(v) for v in box]
    top = [(x, y1) for x in range(x1, x2 + dash, dash)]
    bottom = [(x, y2) for x in range(x1, x2 + dash, dash)]
    left = [(x1, y) for y in range(y1, y2 + dash, dash)]
    right = [(x2, y) for y in range(y1, y2 + dash, dash)]
    dashed_line(draw, top, fill, width, dash)
    dashed_line(draw, bottom, fill, width, dash)
    dashed_line(draw, left, fill, width, dash)
    dashed_line(draw, right, fill, width, dash)


def estimated_full_box(box, image_size, visible_percent):
    """Symmetric geometric hypothesis, deliberately labelled as an estimate."""
    x1, y1, x2, y2 = [float(v) for v in box]
    image_w, image_h = image_size
    box_w, box_h = max(2.0, x2 - x1), max(2.0, y2 - y1)
    factor = min(10.0, max(1.0, 100.0 / visible_percent))
    if box_h >= box_w:
        new_h = min(image_h * 0.95, box_h * factor)
        center_y = (y1 + y2) / 2
        return (x1, max(0, center_y - new_h / 2), x2, min(image_h - 1, center_y + new_h / 2))
    new_w = min(image_w * 0.95, box_w * factor)
    center_x = (x1 + x2) / 2
    return (max(0, center_x - new_w / 2), y1, min(image_w - 1, center_x + new_w / 2), y2)


def annotate(image, result, visible_percent):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    confirmed = 0
    suspected = 0
    boxes = [] if result.boxes is None else result.boxes
    for detection in boxes:
        xyxy = detection.xyxy[0].cpu().tolist()
        confidence = float(detection.conf[0].cpu())
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        if confidence >= 0.25:
            color = (25, 190, 85)
            label = f"BIZTOSABB {confidence:.0%}"
            confirmed += 1
        else:
            color = (255, 145, 0)
            label = f"GYANUS {confidence:.0%}"
            suspected += 1
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        draw.rectangle((x1, max(0, y1 - 26), x1 + 170, y1), fill=color)
        draw.text((x1 + 5, max(0, y1 - 23)), label, fill="white")

        estimate = estimated_full_box(xyxy, out.size, visible_percent)
        dashed_rectangle(draw, estimate, fill=(255, 145, 0), width=4)
        ex1, ey1, _, _ = [int(v) for v in estimate]
        draw.text((ex1 + 5, min(out.height - 22, ey1 + 5)), "BECSULT REJTETT ALAK", fill=(255, 145, 0))
    return out, confirmed, suspected


PAGE = """
<!doctype html><html lang="hu"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rejtett uborka – gyors AI-teszt</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 14px;background:#f3f6f2;color:#17211a}.card{background:#fff;border-radius:16px;padding:22px;box-shadow:0 3px 18px #0001;margin-bottom:16px}h1{margin:0 0 8px;font-size:clamp(24px,6vw,38px)}input{display:block;margin:14px 0;width:100%}input[type=range]{accent-color:#16834b}button{border:0;border-radius:12px;padding:14px 18px;font-size:17px;font-weight:700;background:#16834b;color:#fff;width:100%}img{width:100%;border-radius:12px;margin-top:14px}.muted{color:#5e6b61}.legend{display:grid;gap:8px;margin-top:12px}.green,.orange{padding:9px;border-radius:9px}.green{background:#e7f8ed}.orange{background:#fff1dc}.err{background:#ffecec;color:#8c1b1b}.num{font-size:22px;font-weight:800}code{word-break:break-all}
</style></head><body>
<div class="card"><h1>🥒 Rejtett uborka teszt</h1>
<p>Fotózd le a fóliában úgy, hogy az uborkából akár csak egy kis rész látszódjon ki a levél mögül.</p>
<form method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" capture="environment" required>
<label><b>Becsült látható rész: <span id="pct">10</span>%</b></label>
<input type="range" name="visible_percent" min="5" max="50" value="10" step="5" oninput="pct.textContent=this.value">
<button type="submit">10%-os felismerés indítása</button></form>
<p class="muted">Az első futás Renderen 30–90 másodperc is lehet.</p></div>
{% if error %}<div class="card err"><b>Hiba:</b> {{ error }}</div>{% endif %}
{% if result %}<div class="card"><div><span class="num">{{ total }}</span> lehetséges uborka</div>
<div class="legend"><div class="green">🟩 Biztosabb találat: {{ confirmed }}</div><div class="orange">🟧 Gyenge találat / szaggatott becslés: {{ suspected }}. A szaggatott alak hipotézis, nem a levélen való átlátás.</div></div>
<img src="data:image/jpeg;base64,{{ result }}" alt="Felismert és becsült uborkák"></div>{% endif %}
<div class="card muted"><b>Fontos:</b> ez most gyors prototípus. A 10%-os találatok pontosságához később valódi, részben takart uborkás fotókkal kell finomhangolni az amodális modellt.<br><small>Modell: <code>{{ model_path or 'betöltéskor jelenik meg' }}</code></small></div>
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, error=MODEL_ERROR, result=None, model_path=MODEL_PATH, model_source=MODEL_SOURCE)
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return render_template_string(PAGE, error="Válassz ki egy képet.", result=None), 400
    try:
        visible_percent = max(5, min(50, int(request.form.get("visible_percent", "10"))))
        image = Image.open(upload.stream).convert("RGB")
        image.thumbnail((1600, 1600))
        model = get_model()
        results = model.predict(source=np.asarray(image), conf=0.03, iou=0.45, imgsz=960, augment=True, device="cpu", verbose=False)
        annotated, confirmed, suspected = annotate(image, results[0], visible_percent)
        buf = io.BytesIO(); annotated.save(buf, format="JPEG", quality=91)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return render_template_string(PAGE, error=None, result=encoded, total=confirmed + suspected, confirmed=confirmed, suspected=suspected, model_path=MODEL_PATH, model_source=MODEL_SOURCE)
    except Exception as exc:
        return render_template_string(PAGE, error=str(exc), result=None, model_path=MODEL_PATH), 500


@app.get("/health")
def health():
    return jsonify(status="ok", model_candidate=discover_model() or "yolov8s-worldv2.pt", model_loaded=MODEL is not None, model_source=MODEL_SOURCE, model_error=MODEL_ERROR)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
