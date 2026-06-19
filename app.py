from pathlib import Path
import zipfile
import gc

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


# ============================================================
# EfficientNet-B0 Driver Distraction Demo
# Supports:
#   1) Uploading all_effb0_pth_checkpoints.zip in the Space root.
#   2) Or putting .pth files directly in models/.
#   3) Or putting .pth files directly in the Space root.
# ============================================================

NUM_CLASSES = 10
IMAGE_SIZE = 256

ROOT_DIR = Path(__file__).resolve().parent
ZIP_NAME = "all_effb0_pth_checkpoints.zip"
ZIP_PATH = ROOT_DIR / ZIP_NAME

LOCAL_MODEL_DIR = ROOT_DIR / "models"
TMP_MODEL_DIR = Path("/tmp/effb0_models")

CLASS_NAMES = [f"c{i}" for i in range(NUM_CLASSES)]

CLASS_DESCRIPTION = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}

IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

MODEL_DESCRIPTIONS = {
    "effb0_pretrained_finetune": "Có pretrained ImageNet, fine-tune trực tiếp toàn bộ EfficientNet-B0.",
    "effb0_no_pretrain_scratch": "Không dùng pretrained ImageNet, train EfficientNet-B0 từ đầu.",
    "effb0_head_only_frozen": "Chỉ train classifier head, giữ backbone pretrained cố định.",
    "effb0_head_warmup": "Head warm-up trước, sau đó fine-tune toàn bộ.",
    "effb0_no_aug": "Có pretrained nhưng không dùng augmentation.",
    "effb0_strong_aug": "Có pretrained và dùng augmentation mạnh.",
    "effb0_low_lr": "Có pretrained nhưng learning rate thấp.",
    "effb0_high_lr": "Có pretrained nhưng learning rate cao.",
    "effb0_no_dropout": "Bỏ dropout ở classifier.",
    "effb0_high_dropout": "Tăng dropout ở classifier.",
    "effb0_no_label_smoothing": "Không dùng label smoothing.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def ensure_models_available():
    """Find .pth files from models/, root, or extract from zip into /tmp."""
    TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. If models are already in models/ or root, use them.
    candidates = []
    candidates.extend(sorted(LOCAL_MODEL_DIR.glob("best_effb0*.pth")))
    candidates.extend(sorted(ROOT_DIR.glob("best_effb0*.pth")))
    candidates.extend(sorted(TMP_MODEL_DIR.glob("best_effb0*.pth")))

    if len(candidates) > 0:
        return sorted(set(candidates))

    # 2. If zip is present, extract all .pth files.
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            pth_members = [m for m in zf.namelist() if m.endswith(".pth")]
            if len(pth_members) == 0:
                raise FileNotFoundError(f"Zip found but no .pth inside: {ZIP_PATH}")

            for member in pth_members:
                filename = Path(member).name
                target = TMP_MODEL_DIR / filename
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        candidates = sorted(TMP_MODEL_DIR.glob("best_effb0*.pth"))
        if len(candidates) > 0:
            return candidates

    raise FileNotFoundError(
        "Không tìm thấy checkpoint. Hãy upload một trong các dạng sau:\n"
        "1) all_effb0_pth_checkpoints.zip ở root của Space\n"
        "2) hoặc models/best_effb0_*.pth\n"
        "3) hoặc best_effb0_*.pth ở root."
    )


def build_efficientnet_b0(num_classes=10, dropout=0.2):
    """Rebuild the same EfficientNet-B0 classifier head used during training."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features

    # All checkpoints in the supplied zip use classifier.1.weight/bias,
    # so this Sequential(Dropout, Linear) structure is required.
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def extract_state_dict(checkpoint):
    """Support raw state_dict or checkpoint dictionaries."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # Raw state_dict commonly looks like {"features.0.0.weight": tensor, ...}
        if any(str(k).startswith(("features.", "classifier.", "module.")) for k in checkpoint.keys()):
            return checkpoint

    return checkpoint


def clean_state_dict(state):
    """Remove DataParallel prefix if present."""
    clean = {}
    for k, v in state.items():
        clean[str(k).replace("module.", "")] = v
    return clean


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = extract_state_dict(ckpt)
    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # These prints appear in Space logs, useful for debugging.
    if len(missing) > 0:
        print(f"[WARN] Missing keys for {checkpoint_path.name}: {missing[:8]} ... total={len(missing)}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys for {checkpoint_path.name}: {unexpected[:8]} ... total={len(unexpected)}")

    return model


CHECKPOINTS = ensure_models_available()

CHECKPOINT_MAP = {
    p.stem.replace("best_", ""): p
    for p in CHECKPOINTS
}

if len(CHECKPOINT_MAP) == 0:
    raise RuntimeError("Không có checkpoint nào để chạy demo.")

DEFAULT_MODEL = None
for preferred in ["effb0_high_lr", "effb0_strong_aug", "effb0_pretrained_finetune"]:
    if preferred in CHECKPOINT_MAP:
        DEFAULT_MODEL = preferred
        break

if DEFAULT_MODEL is None:
    DEFAULT_MODEL = list(CHECKPOINT_MAP.keys())[0]


CURRENT = {
    "name": None,
    "model": None,
}


def get_model(model_name):
    """Lazy-load one checkpoint at a time to avoid filling RAM/VRAM."""
    if model_name not in CHECKPOINT_MAP:
        raise ValueError(f"Không tìm thấy model: {model_name}")

    if CURRENT["name"] == model_name and CURRENT["model"] is not None:
        return CURRENT["model"]

    # Free previous model.
    CURRENT["name"] = None
    CURRENT["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_efficientnet_b0(num_classes=NUM_CLASSES, dropout=0.2)
    model = load_checkpoint(model, CHECKPOINT_MAP[model_name])
    model.to(DEVICE)
    model.eval()

    CURRENT["name"] = model_name
    CURRENT["model"] = model

    return model


def model_info(model_name):
    path = CHECKPOINT_MAP.get(model_name)
    desc = MODEL_DESCRIPTIONS.get(model_name, "Checkpoint EfficientNet-B0.")
    device_text = "GPU" if DEVICE.type == "cuda" else "CPU"

    return f"""
### Model đang chọn

**Tên model:** `{model_name}`  
**Mô tả:** {desc}  
**Checkpoint:** `{path.name if path else "unknown"}`  
**Thiết bị chạy:** `{device_text}`
"""


@torch.no_grad()
def predict(image, model_name):
    if image is None:
        empty_probs = {f"{c}: {CLASS_DESCRIPTION[c]}": 0.0 for c in CLASS_NAMES}
        empty_table = pd.DataFrame({
            "rank": [],
            "class": [],
            "description": [],
            "probability": [],
            "percent": [],
        })
        return "Chưa có ảnh.", empty_probs, empty_table, model_info(model_name)

    model = get_model(model_name)

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    order = np.argsort(-probs)

    rows = []
    label_dict = {}

    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        cls = IDX_TO_CLASS[idx]
        desc = CLASS_DESCRIPTION[cls]
        prob = float(probs[idx])

        label_name = f"{cls}: {desc}"
        label_dict[label_name] = prob

        rows.append({
            "rank": rank,
            "class": cls,
            "description": desc,
            "probability": prob,
            "percent": f"{prob * 100:.2f}%",
        })

    top = rows[0]

    result_md = f"""
### Kết quả dự đoán

**Model:** `{model_name}`  
**Dự đoán cao nhất:** `{top["class"]}` — {top["description"]}  
**Độ tin cậy:** **{top["percent"]}**
"""

    return result_md, label_dict, pd.DataFrame(rows), model_info(model_name)


with gr.Blocks(title="EfficientNet-B0 Driver Distraction Demo") as demo:
    gr.Markdown("# Driver Distraction Detection — EfficientNet-B0")
    gr.Markdown(
        "Demo nhận diện hành vi tài xế bằng các checkpoint EfficientNet-B0. "
        "Upload ảnh, chọn model, rồi xem xác suất 10 lớp."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(CHECKPOINT_MAP.keys()),
                value=DEFAULT_MODEL,
                label="Chọn model/checkpoint",
            )

            image_input = gr.Image(
                type="pil",
                label="Upload ảnh tài xế",
            )

            predict_button = gr.Button("Dự đoán", variant="primary")

        with gr.Column(scale=2):
            model_md = gr.Markdown(value=model_info(DEFAULT_MODEL))
            result_md = gr.Markdown()
            result_label = gr.Label(num_top_classes=10, label="Xác suất 10 lớp")
            result_table = gr.Dataframe(label="Bảng xác suất đầy đủ", interactive=False)

    predict_button.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_md, result_label, result_table, model_md],
    )

    image_input.change(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_md, result_label, result_table, model_md],
    )

    model_dropdown.change(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[result_md, result_label, result_table, model_md],
    )

if __name__ == "__main__":
    demo.launch()
