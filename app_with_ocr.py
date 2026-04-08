
from pathlib import Path
import json
import re

import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from ultralytics import YOLO
import easyocr
import gradio as gr


try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path().resolve()

CLASS_NAMES = {1: "licence"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def find_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def recursive_find(filename):
    matches = list(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def load_json_if_exists(path):
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def locate_yolo_weights():
    candidates = [
        PROJECT_ROOT / "runs" / "plate_detector" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "plate_detector" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "runs" / "plate_detector" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
    ]
    return find_first_existing(candidates) or recursive_find("best.pt")


def locate_faster_weights():
    candidates = [
        PROJECT_ROOT / "fasterrcnn_outputs" / "best_fasterrcnn_license_plate.pth",
    ]
    return find_first_existing(candidates) or recursive_find("best_fasterrcnn_license_plate.pth")


def locate_retina_weights():
    candidates = [
        PROJECT_ROOT / "retinanet_outputs" / "best_retinanet_license_plate.pth",
    ]
    return find_first_existing(candidates) or recursive_find("best_retinanet_license_plate.pth")


YOLO_WEIGHTS = locate_yolo_weights()
FRCNN_WEIGHTS = locate_faster_weights()
RETINA_WEIGHTS = locate_retina_weights()

FRCNN_RESULTS_JSON = find_first_existing([
    PROJECT_ROOT / "fasterrcnn_outputs" / "fasterrcnn_results.json",
])

RETINA_RESULTS_JSON = find_first_existing([
    PROJECT_ROOT / "retinanet_outputs" / "retinanet_results.json",
])


def load_yolo_model():
    if YOLO_WEIGHTS is None:
        raise FileNotFoundError("YOLO best.pt not found.")
    return YOLO(str(YOLO_WEIGHTS))


def load_faster_model():
    if FRCNN_WEIGHTS is None:
        raise FileNotFoundError("Faster R-CNN weights not found.")

    num_classes = 2
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state = torch.load(FRCNN_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_retina_model():
    if RETINA_WEIGHTS is None:
        raise FileNotFoundError("RetinaNet weights not found.")

    num_classes = 2
    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=512, max_size=512)

    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    model.score_thresh = 0.5
    model.nms_thresh = 0.5
    model.detections_per_img = 100

    state = torch.load(RETINA_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


print(f"Using device: {DEVICE}")
print(f"YOLO weights: {YOLO_WEIGHTS}")
print(f"Faster R-CNN weights: {FRCNN_WEIGHTS}")
print(f"RetinaNet weights: {RETINA_WEIGHTS}")

YOLO_MODEL = None
FRCNN_MODEL = None
RETINA_MODEL = None
OCR_READER = None

YOLO_LOAD_ERROR = None
FRCNN_LOAD_ERROR = None
RETINA_LOAD_ERROR = None
OCR_LOAD_ERROR = None

try:
    YOLO_MODEL = load_yolo_model()
except Exception as e:
    YOLO_LOAD_ERROR = str(e)
    print("YOLO loading error:", e)

try:
    FRCNN_MODEL = load_faster_model()
except Exception as e:
    FRCNN_LOAD_ERROR = str(e)
    print("Faster R-CNN loading error:", e)

try:
    RETINA_MODEL = load_retina_model()
except Exception as e:
    RETINA_LOAD_ERROR = str(e)
    print("RetinaNet loading error:", e)

try:
    OCR_READER = easyocr.Reader(['en'], gpu=(DEVICE.type == "cuda"))
except Exception as e:
    OCR_LOAD_ERROR = str(e)
    print("EasyOCR loading error:", e)


def draw_boxes(pil_image, boxes, scores, labels=None, score_thresh=0.25, color="red"):
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        score = float(scores[i]) if scores is not None else 1.0
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = ""
        if labels is not None and i < len(labels):
            label_value = labels[i]
            if isinstance(label_value, (int, np.integer)):
                label_text = CLASS_NAMES.get(int(label_value), str(label_value))
            else:
                label_text = str(label_value)

        text = f"{label_text} {score:.2f}".strip()
        draw.text((x1, max(0, y1 - 18)), text, fill=color)

    return img


def pil_to_tensor(image):
    return F.to_tensor(image).to(DEVICE)


def predict_yolo(pil_image, conf_thresh):
    if YOLO_MODEL is None:
        raise RuntimeError(f"YOLO model could not be loaded: {YOLO_LOAD_ERROR}")

    results = YOLO_MODEL.predict(
        source=np.array(pil_image),
        conf=conf_thresh,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return pil_image.copy(), 0, "No detections", None

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = ["licence"] * len(boxes)

    annotated = draw_boxes(pil_image, boxes, scores, labels=labels, score_thresh=conf_thresh, color="red")
    best_idx = int(np.argmax(scores))
    best_box = boxes[best_idx]
    return annotated, len(boxes), f"Detections: {len(boxes)}", best_box


def predict_torch_detector(model, pil_image, conf_thresh, color="red"):
    if model is None:
        return pil_image.copy(), 0, "Model not loaded"

    img_tensor = pil_to_tensor(pil_image)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    boxes = pred["boxes"].detach().cpu().numpy() if "boxes" in pred else np.empty((0, 4))
    scores = pred["scores"].detach().cpu().numpy() if "scores" in pred else np.empty((0,))
    labels = pred["labels"].detach().cpu().numpy() if "labels" in pred else np.empty((0,))

    kept = scores >= conf_thresh
    boxes = boxes[kept]
    scores = scores[kept]
    labels = labels[kept]

    if len(boxes) == 0:
        return pil_image.copy(), 0, "No detections"

    annotated = draw_boxes(pil_image, boxes, scores, labels=labels, score_thresh=conf_thresh, color=color)
    return annotated, len(boxes), f"Detections: {len(boxes)}"


def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def expand_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.08):
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return x1, y1, x2, y2


def crop_plate_from_box(pil_image, box, pad_ratio=0.08):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, pad_ratio=pad_ratio)
    crop_bgr = image_bgr[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def preprocess_plate_for_ocr(pil_crop):
    crop_rgb = np.array(pil_crop.convert("RGB"))
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(thresh)


def run_ocr_on_plate(processed_pil):
    if OCR_READER is None:
        return "OCR not loaded", "", 0.0

    arr = np.array(processed_pil)
    results = OCR_READER.readtext(arr)

    if not results:
        return "No text detected", "", 0.0

    raw_text = " ".join([item[1] for item in results]).strip()
    cleaned_text = clean_plate_text(raw_text)
    avg_conf = float(sum(float(item[2]) for item in results) / len(results))

    return raw_text, cleaned_text, avg_conf


def build_metrics_markdown():
    frcnn = load_json_if_exists(FRCNN_RESULTS_JSON) if FRCNN_RESULTS_JSON else None
    retina = load_json_if_exists(RETINA_RESULTS_JSON) if RETINA_RESULTS_JSON else None

    lines = []
    lines.append("### Saved training metrics")
    lines.append("")
    lines.append(f"- **Device used by GUI now:** `{DEVICE}`")
    lines.append(f"- **YOLO loaded:** `{YOLO_MODEL is not None}`")
    lines.append(f"- **Faster R-CNN loaded:** `{FRCNN_MODEL is not None}`")
    lines.append(f"- **RetinaNet loaded:** `{RETINA_MODEL is not None}`")
    lines.append(f"- **EasyOCR loaded:** `{OCR_READER is not None}`")
    lines.append("")

    if frcnn:
        tm = frcnn.get("test_metrics", {})
        lines.append("**Faster R-CNN**")
        lines.append(f"- Best val F1: `{frcnn.get('best_val_f1', 'N/A')}`")
        lines.append(f"- Test precision: `{tm.get('precision', 'N/A')}`")
        lines.append(f"- Test recall: `{tm.get('recall', 'N/A')}`")
        lines.append(f"- Test F1: `{tm.get('f1', 'N/A')}`")
        lines.append("")

    if retina:
        tm = retina.get("test_metrics", {})
        lines.append("**RetinaNet**")
        lines.append(f"- Best val F1: `{retina.get('best_val_f1', 'N/A')}`")
        lines.append(f"- Test precision: `{tm.get('precision', 'N/A')}`")
        lines.append(f"- Test recall: `{tm.get('recall', 'N/A')}`")
        lines.append(f"- Test F1: `{tm.get('f1', 'N/A')}`")
        lines.append("")

    if not frcnn and not retina:
        lines.append("No JSON metrics files were found yet. The GUI will still work if the model weights exist.")

    if YOLO_LOAD_ERROR:
        lines.append(f"- YOLO load error: `{YOLO_LOAD_ERROR}`")
    if FRCNN_LOAD_ERROR:
        lines.append(f"- Faster R-CNN load error: `{FRCNN_LOAD_ERROR}`")
    if RETINA_LOAD_ERROR:
        lines.append(f"- RetinaNet load error: `{RETINA_LOAD_ERROR}`")
    if OCR_LOAD_ERROR:
        lines.append(f"- EasyOCR load error: `{OCR_LOAD_ERROR}`")

    return "\n".join(lines)


def compare_models_and_ocr(image, conf_thresh, ocr_pad_ratio):
    try:
        if image is None:
            raise gr.Error("Please upload an image first.")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        yolo_img, yolo_count, yolo_text, best_yolo_box = predict_yolo(image, conf_thresh)
        frcnn_img, frcnn_count, frcnn_text = predict_torch_detector(FRCNN_MODEL, image, conf_thresh, color="blue")
        retina_img, retina_count, retina_text = predict_torch_detector(RETINA_MODEL, image, conf_thresh, color="green")

        plate_crop = None
        processed_plate = None
        raw_text = "No plate detected"
        cleaned_text = ""
        ocr_conf = 0.0

        if best_yolo_box is not None:
            plate_crop = crop_plate_from_box(image, best_yolo_box, pad_ratio=ocr_pad_ratio)
            if plate_crop is not None:
                processed_plate = preprocess_plate_for_ocr(plate_crop)
                raw_text, cleaned_text, ocr_conf = run_ocr_on_plate(processed_plate)

        summary = f"""
### Prediction summary
- **YOLOv8:** {yolo_text}
- **Faster R-CNN:** {frcnn_text}
- **RetinaNet:** {retina_text}

### OCR summary
- **Raw OCR text:** {raw_text}
- **Cleaned plate text:** {cleaned_text if cleaned_text else "N/A"}
- **OCR confidence:** {ocr_conf:.4f}
"""

        table = [
            ["YOLOv8", yolo_count],
            ["Faster R-CNN", frcnn_count],
            ["RetinaNet", retina_count],
        ]

        return yolo_img, frcnn_img, retina_img, plate_crop, processed_plate, summary, table

    except Exception as e:
        return None, None, None, None, None, f"ERROR:\n{str(e)}", []


with gr.Blocks(title="License Plate Detector Comparison + OCR") as demo:
    gr.Markdown("# License Plate Detector GUI + OCR")
    gr.Markdown(
        "Upload one image to compare **YOLOv8**, **Faster R-CNN**, and **RetinaNet**. "
        "The GUI also crops the best **YOLO** plate detection and runs **EasyOCR** on it."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload image")
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Detection confidence threshold",
            )
            pad_slider = gr.Slider(
                minimum=0.00,
                maximum=0.30,
                value=0.08,
                step=0.01,
                label="OCR crop padding ratio",
            )
            run_btn = gr.Button("Run comparison + OCR", variant="primary")
            metrics_md = gr.Markdown(value=build_metrics_markdown())

        with gr.Column(scale=2):
            summary_md = gr.Markdown("### Prediction summary\nUpload an image, then click **Run comparison + OCR**.")
            result_table = gr.Dataframe(
                headers=["Model", "Detections"],
                datatype=["str", "number"],
                row_count=3,
                col_count=(2, "fixed"),
                label="Detection counts"
            )

    with gr.Row():
        yolo_out = gr.Image(label="YOLOv8 result")
        frcnn_out = gr.Image(label="Faster R-CNN result")
        retina_out = gr.Image(label="RetinaNet result")

    with gr.Row():
        crop_out = gr.Image(label="Best YOLO plate crop")
        ocr_pre_out = gr.Image(label="OCR preprocessed crop")

    run_btn.click(
        fn=compare_models_and_ocr,
        inputs=[input_image, conf_slider, pad_slider],
        outputs=[yolo_out, frcnn_out, retina_out, crop_out, ocr_pre_out, summary_md, result_table],
    )

if __name__ == "__main__":
    demo.launch()
