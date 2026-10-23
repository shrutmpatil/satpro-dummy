
import gc
import math
import pathlib
import time
from types import SimpleNamespace
from collections.abc import Mapping

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
import networkx as nx

try:
    from skimage.morphology import skeletonize as sk_skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False



# ============================================================
# ROUTEFORMER-RC DASHBOARD
# Architecture and preprocessing aligned to the supplied notebook
# ============================================================

st.set_page_config(
    page_title="SatPro | Satellite Road Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN SYSTEM -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg: #eef3f8;
    --surface: #ffffff;
    --surface-alt: #f7faff;
    --navy: #102a43;
    --navy-2: #173b5e;
    --blue: #1677ff;
    --cyan: #06a6c8;
    --teal: #0e8f82;
    --text: #172b3f;
    --muted: #667b90;
    --line: #d7e2ec;
    --line-strong: #c2d2e0;
    --good: #18864b;
    --warning: #a86100;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(22,119,255,.075), transparent 26%),
        radial-gradient(circle at 100% 10%, rgba(6,166,200,.055), transparent 24%),
        var(--bg);
}

.block-container {
    max-width: 1540px;
    padding: 1.15rem 2rem 3rem;
}

/* ---------- SATPRO HEADER ---------- */
.hero {
    background:
        linear-gradient(115deg, #0d2945 0%, #123c61 62%, #0d6f78 100%);
    border: 0;
    border-radius: 20px;
    padding: 25px 29px 24px;
    margin-bottom: 17px;
    box-shadow: 0 14px 35px rgba(16,42,67,.18);
    color: white;
    overflow: hidden;
    position: relative;
}

.hero:after {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    right: -130px;
    top: -180px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.11);
    box-shadow:
        0 0 0 35px rgba(255,255,255,.025),
        0 0 0 70px rgba(255,255,255,.018);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    position: relative;
    z-index: 2;
}

.brand-mark {
    width: 43px;
    height: 43px;
    border-radius: 12px;
    background: linear-gradient(135deg, #36d6c4, #4da3ff);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #08233c;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 14px;
    box-shadow: 0 8px 20px rgba(0,0,0,.18);
}

.brand-name {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 16px;
    color: white;
}

.brand-meta {
    color: #b8d3e8;
    font-size: 12px;
    margin-left: 3px;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 2.2rem;
    line-height: 1.12;
    font-weight: 800;
    margin: 0 0 8px;
    color: white;
    position: relative;
    z-index: 2;
}

.hero-sub {
    color: #c5d8e9;
    font-size: .91rem;
    line-height: 1.65;
    max-width: 1000px;
    position: relative;
    z-index: 2;
}

.protocol-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
    position: relative;
    z-index: 2;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    color: #e3f1fc;
    font-size: 11px;
    font-weight: 700;
}

/* ---------- CONTENT ---------- */
.section {
    color: #71849a;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin: 21px 0 9px;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 16px;
    box-shadow: 0 5px 18px rgba(31,62,89,.045);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 13px 14px;
    min-height: 89px;
    box-shadow: 0 4px 13px rgba(31,62,89,.035);
}

.metric-label,
.data-label {
    color: #71849a;
    font-size: .65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .075em;
}

.metric-value,
.data-value {
    color: #122b43;
    font-family: 'Manrope', sans-serif;
    font-size: 1.38rem;
    font-weight: 800;
    margin-top: 5px;
}

.metric-note,
.data-desc {
    color: #8a9aac;
    font-size: .67rem;
    margin-top: 2px;
}

.process-step,
.data-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 13px rgba(31,62,89,.035);
}

.process-step {
    min-height: 103px;
}

.process-num {
    color: var(--blue);
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: .67rem;
}

.process-title {
    color: #27415a;
    font-family: 'Manrope', sans-serif;
    font-size: .81rem;
    font-weight: 800;
    margin-top: 5px;
}

.process-desc {
    color: #7e90a2;
    font-size: .67rem;
    line-height: 1.42;
    margin-top: 3px;
}

.result-strip {
    background: linear-gradient(90deg, #edf7ff, #effaf8);
    border: 1px solid #d4e5ed;
    border-left: 3px solid var(--blue);
    border-radius: 10px;
    padding: 11px 13px;
    color: #536b81;
    font-size: .76rem;
    line-height: 1.55;
    margin: 9px 0;
}

.research-note {
    background: #f7fafc;
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    border-radius: 9px;
    padding: 10px 12px;
    color: #62778b;
    font-size: .76rem;
    line-height: 1.55;
}

.image-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 9px;
    margin-bottom: 9px;
}

.image-title {
    font-family: 'Manrope', sans-serif;
    font-size: .86rem;
    font-weight: 800;
    color: #29435c;
    margin: 2px 2px 8px;
}

.image-subtitle {
    color: #8091a3;
    font-size: .69rem;
    margin: -4px 2px 8px;
}

.small-note {
    color: #687d91;
    font-size: .76rem;
    line-height: 1.55;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #d8e3ed;
}

[data-testid="stSidebar"] * {
    color: #2a3d52;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #102a43 !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #708297 !important;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #bfd0df;
    border-radius: 11px;
    background: #ffffff;
}

[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border-radius: 9px;
}

[data-testid="stFileUploader"] button {
    border: 1px solid #ccd9e5 !important;
    background: #f8fbfe !important;
    color: #315a7d !important;
    border-radius: 8px !important;
}

/* ---------- CONTROLS ---------- */
button[kind="primary"],
button[kind="secondary"],
.stDownloadButton button {
    border-radius: 8px !important;
}

div[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--line);
}

button[data-baseweb="tab"] {
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stImage"] {
    border-radius: 9px;
    overflow: hidden;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 9px;
}

hr {
    border-color: #dfe7ef;
}
</style>
""", unsafe_allow_html=True)


# ------------------------- CONSTANTS -------------------------

IMAGE_SIZE = 512
THRESHOLD = 0.55
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------------- RUNTIME DEVICE -------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LABEL = "CUDA / GPU" if DEVICE.type == "cuda" else "CPU"



# ------------------------- MODEL ARCHITECTURE -------------------------

class HFMiTB0Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Lazy import: Transformers is loaded only when the model is actually needed.
        from transformers import SegformerModel
        self.model = SegformerModel.from_pretrained("nvidia/mit-b0")

    def forward(self, x):
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        expected_channels = [32, 64, 160, 256]
        selected = []

        for state in hidden_states:
            if state.ndim == 4:
                if state.shape[1] in expected_channels:
                    selected.append(state)
            elif state.ndim == 3:
                b, hw, c = state.shape
                if c in expected_channels:
                    side = int(hw ** 0.5)
                    if side * side != hw:
                        continue
                    selected.append(
                        state.transpose(1, 2).reshape(
                            b, c, side, side
                        ).contiguous()
                    )

        if len(selected) != 4:
            raise RuntimeError(
                "Could not identify the four MiT-B0 feature stages. "
                f"Found {len(selected)} stages."
            )

        order = {32: 0, 64: 1, 160: 2, 256: 3}
        selected.sort(key=lambda x: order[x.shape[1]])
        return selected


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * len(in_channels),
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, feats):
        target_hw = feats[0].shape[-2:]
        projected = [
            proj(f) for proj, f in zip(self.projections, feats)
        ]
        upsampled = [projected[0]] + [
            F.interpolate(
                p,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            for p in projected[1:]
        ]
        return self.fuse(torch.cat(upsampled, dim=1))


class SharedDecoder(nn.Module):
    def __init__(self, in_channels, decoder_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, decoder_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(
                decoder_channels, decoder_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class RoadHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, d):
        return self.conv(d)


class ContinuityHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, groups=channels
        )
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, d):
        return self.pw(self.act(self.dw(d)))


class BoundaryHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, d):
        return self.conv(d)


class ContinuityGuidedRefinement(nn.Module):
    def __init__(self, channels, alpha=0.35):
        super().__init__()
        assert 0.2 <= alpha <= 0.5
        self.alpha = alpha
        self.gate_conv = nn.Conv2d(
            channels + 1, channels, kernel_size=1
        )

    def forward(self, d, continuity_logits):
        r = torch.sigmoid(continuity_logits)
        z = torch.cat([d, r], dim=1)
        g = torch.sigmoid(self.gate_conv(z))
        return d * (1.0 + self.alpha * g)


class RouteFormerRC(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = HFMiTB0Encoder()
        enc_channels = [32, 64, 160, 256]
        dc = 128

        self.fusion = MultiScaleFusion(enc_channels, dc)
        self.decoder = SharedDecoder(dc, dc)
        self.road_head = RoadHead(dc)
        self.continuity_head = ContinuityHead(dc)
        self.boundary_head = BoundaryHead(dc)
        self.refine = ContinuityGuidedRefinement(dc, 0.35)
        self.final_head = RoadHead(dc)

    def forward(self, x):
        in_hw = x.shape[-2:]
        feats = self.encoder(x)
        fused = self.fusion(feats)
        d = self.decoder(fused)

        road_logits = F.interpolate(
            self.road_head(d),
            size=in_hw,
            mode="bilinear",
            align_corners=False,
        )
        cont_logits_low = self.continuity_head(d)
        continuity_logits = F.interpolate(
            cont_logits_low,
            size=in_hw,
            mode="bilinear",
            align_corners=False,
        )
        boundary_logits = F.interpolate(
            self.boundary_head(d),
            size=in_hw,
            mode="bilinear",
            align_corners=False,
        )

        d_refined = self.refine(d, cont_logits_low)
        final_logits = F.interpolate(
            self.final_head(d_refined),
            size=in_hw,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "road_logits": road_logits,
            "continuity_logits": continuity_logits,
            "boundary_logits": boundary_logits,
            "final_logits": final_logits,
        }


# ------------------------- CHECKPOINT -------------------------

# Exact checkpoint path recorded in the supplied evaluation notebook.
# On the original Windows machine this is expected to be:
# C:\\Users\\HP\\Downloads\\best.pt
DEFAULT_CHECKPOINT = pathlib.Path(r"C:\\Users\\HP\\Downloads\\best.pt")
# Dataset root recorded in the supplied SatPro evaluation notebook.
DATASET_ROOT = pathlib.Path(r"F:\Dataset\raw_data\deepglobe")


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_bytes):
    checkpoint_path = pathlib.Path("routeformer_uploaded_best.pt")
    checkpoint_path.write_bytes(checkpoint_bytes)

    # Support checkpoints saved on Windows when loading on Linux.
    # WindowsPath cannot be instantiated on Linux, so temporarily map it
    # to PosixPath only while deserializing the checkpoint.
    original_windows_path = pathlib.WindowsPath
    try:
        pathlib.WindowsPath = pathlib.PosixPath
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    finally:
        pathlib.WindowsPath = original_windows_path

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Unsupported checkpoint format.")

    state_dict = None
    for key in ("state_dict", "model_state_dict", "model", "weights"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            state_dict = dict(checkpoint[key])
            break

    if state_dict is None:
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = dict(checkpoint)

    if state_dict is None:
        raise RuntimeError(
            "Could not find model_state_dict/state_dict in the .pt checkpoint."
        )

    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # The supplied notebook/checkpoint was created with Transformers 5.18.0.
    # Its MiT-B0 keys use `encoder.model.stages.*`. Older/newer incompatible
    # Transformers builds may expose `encoder.model.encoder.*` instead.
    checkpoint_uses_stages = any(
        str(k).startswith("encoder.model.stages.")
        for k in state_dict.keys()
    )

    try:
        import transformers as _transformers
        installed_transformers = _transformers.__version__
    except Exception:
        installed_transformers = "unknown"

    if checkpoint_uses_stages and installed_transformers != "5.18.0":
        raise RuntimeError(
            "SatPro checkpoint architecture mismatch. "
            "The supplied best.pt was evaluated with Transformers 5.18.0 "
            "and contains `encoder.model.stages.*` keys, but this environment "
            f"has Transformers {installed_transformers}. "
            "Install the exact notebook environment with: "
            "`pip install --upgrade --force-reinstall transformers==5.18.0` "
            "and restart Streamlit."
        )

    model = RouteFormerRC()
    missing, unexpected = model.load_state_dict(
        state_dict, strict=True
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Strict checkpoint load failed. "
            f"Missing={missing[:5]}, Unexpected={unexpected[:5]}"
        )

    model.eval()

    # Use CUDA automatically when available; keep CPU as a safe fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, checkpoint


# ------------------------- PREPROCESSING -------------------------

def image_quality_stats(image_rgb):
    """Lightweight input-quality diagnostics; does not alter inference."""
    arr = np.asarray(image_rgb.convert("RGB")).astype(np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    brightness = float(arr.mean() / 255.0)
    contrast = float(arr.std() / 255.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def apply_experimental_enhancement(image_rgb):
    """Optional display/inference enhancement.

    This is deliberately OFF by default because the supplied research
    evaluation notebook does not use CLAHE or denoising. When enabled,
    the image is mildly denoised and contrast-enhanced before the exact
    resize/normalization pipeline.
    """
    img = np.asarray(image_rgb.convert("RGB")).copy()

    # Very light edge-preserving denoising.
    img = cv2.bilateralFilter(
        img,
        d=5,
        sigmaColor=25,
        sigmaSpace=25,
    )

    # CLAHE on luminance only; preserve RGB chromatic relationships.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(8, 8),
    )
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(
        cv2.merge([l, a, b]),
        cv2.COLOR_LAB2RGB,
    )

    return Image.fromarray(enhanced)


def preprocess_image(image_rgb, experimental_enhancement=False):
    """Exact notebook preprocessing, with optional experimental enhancement.

    Default path:
      1. RGB
      2. resize 512x512 using bilinear interpolation
      3. scale to [0,1]
      4. ImageNet mean/std normalization
      5. CHW tensor + batch dimension

    The default path therefore remains equivalent to the supplied
    SatPro evaluation pipeline.
    """
    source = image_rgb.convert("RGB")

    if experimental_enhancement:
        source = apply_experimental_enhancement(source)

    image = np.asarray(source)
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    img_f = image.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(
        np.ascontiguousarray(img_f.transpose(2, 0, 1))
    ).float().unsqueeze(0)

    return image, tensor, source


# ------------------------- METRICS -------------------------

def safe_div(num, den):
    return float(num / den) if den > 0 else float("nan")


def binary_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(
        2 * precision * recall,
        precision + recall
    )
    accuracy = safe_div(
        tp + tn,
        tp + fp + fn + tn
    )

    return {
        "IoU": iou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
    }


def derive_centerline(mask_bin):
    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin)

    if HAS_SKIMAGE:
        return sk_skeletonize(
            mask_bin.astype(bool)
        ).astype(np.uint8)

    img = (mask_bin > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )
    work = img.copy()

    for _ in range(200):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, subset)
        work = eroded.copy()
        if cv2.countNonZero(work) == 0:
            break

    return (skel > 0).astype(np.uint8)


def topology_refine_mask(mask, close_kernel=3, min_component_area=12):
    b = (mask > 0).astype(np.uint8)

    k = max(1, int(close_kernel))
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)
    b = cv2.morphologyEx(
        b, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        b, connectivity=8
    )

    if n > 1 and min_component_area > 0:
        keep = np.zeros_like(b)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= min_component_area:
                keep[labels == lab] = 1
        b = keep

    return b


def skeleton_to_graph(skeleton):
    skel = (skeleton > 0).astype(np.uint8)
    g = nx.Graph()

    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return g

    coords = set(zip(ys.tolist(), xs.tolist()))

    def neighbors8(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p = (y + dy, x + dx)
                if p in coords:
                    yield p

    degree = {
        p: sum(1 for _ in neighbors8(*p))
        for p in coords
    }

    special_nodes = {
        p for p, d in degree.items() if d != 2
    }

    if not special_nodes:
        special_nodes = {next(iter(coords))}

    for p in special_nodes:
        g.add_node(p)

    visited = set()

    for start in special_nodes:
        for nb in neighbors8(*start):
            edge_key = tuple(sorted((start, nb)))
            if edge_key in visited:
                continue

            path = [start, nb]
            visited.add(edge_key)

            prev, cur = start, nb
            safety = 0

            while cur not in special_nodes and safety < len(coords):
                nxts = [
                    n for n in neighbors8(*cur)
                    if n != prev
                ]
                if not nxts:
                    break

                nxt = nxts[0]
                edge_key = tuple(sorted((cur, nxt)))
                if edge_key in visited:
                    break

                visited.add(edge_key)
                path.append(nxt)
                prev, cur = cur, nxt
                safety += 1

            end = cur

            if end in special_nodes and len(path) >= 2:
                arr = np.asarray(path, dtype=np.float32)
                seg = np.diff(arr, axis=0)
                path_len = float(
                    np.linalg.norm(seg, axis=1).sum()
                )

                if not g.has_edge(start, end):
                    g.add_edge(
                        start,
                        end,
                        path_length=path_len,
                    )

    return g


def raster_derived_apls(
    gt_mask,
    pred_mask,
    num_sample_nodes=40,
    seed=0,
    snap_tolerance_px=20.0,
):
    gt_mask = topology_refine_mask(gt_mask)
    pred_mask = topology_refine_mask(pred_mask)

    gt_skel = derive_centerline(gt_mask)
    pred_skel = derive_centerline(pred_mask)

    if gt_skel.sum() < 2 or pred_skel.sum() < 2:
        return None

    g_gt = skeleton_to_graph(gt_skel)
    g_pred = skeleton_to_graph(pred_skel)

    if (
        g_gt.number_of_nodes() < 2
        or g_pred.number_of_nodes() < 2
    ):
        return None

    gt_nodes = list(g_gt.nodes())
    sample_n = min(int(num_sample_nodes), len(gt_nodes))
    if sample_n < 2:
        return None

    rng = np.random.RandomState(seed)

    degrees = dict(g_gt.degree())
    structural = [
        n for n in gt_nodes if degrees.get(n, 0) != 2
    ]
    base = structural[:sample_n]

    if len(base) < sample_n:
        remaining = [
            n for n in gt_nodes if n not in set(base)
        ]
        extra_n = min(sample_n - len(base), len(remaining))
        if extra_n:
            extra = rng.choice(
                len(remaining),
                size=extra_n,
                replace=False,
            )
            base.extend(
                [remaining[i] for i in np.atleast_1d(extra)]
            )

    sampled = base

    pred_nodes = np.asarray(
        list(g_pred.nodes()), dtype=np.float32
    )

    tree = cKDTree(pred_nodes)
    distances, indices = tree.query(
        np.asarray(sampled, dtype=np.float32),
        k=1,
    )

    snapped = {
        n: (
            tuple(pred_nodes[j].astype(int))
            if float(d) <= snap_tolerance_px
            else None
        )
        for n, d, j in zip(
            sampled, distances, indices
        )
    }

    sims = []

    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            a, b = sampled[i], sampled[j]

            try:
                l_gt = float(
                    nx.shortest_path_length(
                        g_gt,
                        a,
                        b,
                        weight="path_length",
                    )
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            pa, pb = snapped.get(a), snapped.get(b)

            if pa is None or pb is None:
                sims.append(0.0)
                continue

            try:
                l_pred = float(
                    nx.shortest_path_length(
                        g_pred,
                        pa,
                        pb,
                        weight="path_length",
                    )
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                sims.append(0.0)
                continue

            if l_gt <= 1e-6:
                sims.append(
                    1.0 if l_pred <= 1e-6 else 0.0
                )
                continue

            denom = max(l_gt, l_pred, 1e-6)
            sims.append(
                max(
                    0.0,
                    1.0 - abs(l_gt - l_pred) / denom,
                )
            )

    return float(np.mean(sims)) if sims else None


# ------------------------- VISUALIZATION -------------------------

def normalize_u8(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip(
        (x - lo) / (hi - lo) * 255,
        0, 255
    ).astype(np.uint8)


def probability_rgb(prob):
    # OpenCV TURBO gives a clear confidence heatmap.
    return cv2.cvtColor(
        cv2.applyColorMap(
            (np.clip(prob, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        ),
        cv2.COLOR_BGR2RGB,
    )


def overlay_mask(image_rgb, mask, alpha=0.50):
    out = image_rgb.astype(np.float32).copy()
    road = mask.astype(bool)
    # Neutral cyan/teal road highlight.
    overlay = np.zeros_like(out)
    overlay[..., 1] = 225
    overlay[..., 2] = 190
    out[road] = (
        (1 - alpha) * out[road]
        + alpha * overlay[road]
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def make_probability_overlay(image_rgb, prob):
    """Probability heatmap blended over the original satellite image."""
    heat = probability_rgb(prob)
    return cv2.addWeighted(
        image_rgb.astype(np.uint8),
        0.60,
        heat.astype(np.uint8),
        0.40,
        0,
    )


def black_probability_heatmap(prob):
    """Road probability heatmap on a true black background.

    Low-probability pixels remain black; increasing probability produces
    increasingly visible heatmap intensity.
    """
    p = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(
        (p * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Suppress the low-probability blue floor so the background is black.
    intensity = np.power(p, 0.85)[..., None]
    heat *= intensity
    heat[p < 0.03] = 0

    return np.clip(heat, 0, 255).astype(np.uint8)


def binary_mask_rgb(mask):
    """Explicit black/white road mask: background=black, road=white."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return np.repeat(m[..., None], 3, axis=2)


# ------------------------- GROUND-TRUTH RESOLUTION -------------------------

def find_matching_ground_truth(image_name: str):
    """Find the DeepGlobe mask paired with an uploaded *_sat image.

    The supplied notebook documents the pairing convention:
        <id>_sat.jpg  <->  <id>_mask.png

    Returns the first exact filename match under DATASET_ROOT, or None.
    """
    if not image_name or not DATASET_ROOT.exists():
        return None

    stem = Path(image_name).stem
    image_id = stem[:-4] if stem.lower().endswith("_sat") else stem
    if not image_id:
        return None

    exact_names = [
        f"{image_id}_mask.png",
        f"{image_id}_mask.jpg",
        f"{image_id}_mask.jpeg",
        f"{image_id}_mask.tif",
        f"{image_id}_mask.tiff",
    ]

    for name in exact_names:
        matches = list(DATASET_ROOT.rglob(name))
        if matches:
            return matches[0]

    return None


# ------------------------- UI -------------------------


st.markdown("""
<div class="hero">
    <div class="brand-row">
        <div class="brand-mark">RF</div>
        <div>
            <div class="brand-name">SatPro</div>
            <div class="brand-meta">Satellite Road Intelligence • Research Studio</div>
        </div>
    </div>
    <div class="hero-title">Satellite road intelligence, clearly visualized.</div>
    <div class="hero-sub">
        Single-image inference with segmentation, result processing, data representation, quantitative evaluation and topology-aware road-network diagnostics.
    </div>
    <div class="protocol-row">
        <span class="badge">MiT-B0 / SegFormer</span>
        <span class="badge">512 × 512</span>
        <span class="badge">Threshold 0.55</span>
        <span class="badge">TTA OFF</span>
        <span class="badge">Auto GPU / CPU</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Control Center")

    default_checkpoint_exists = DEFAULT_CHECKPOINT.exists()

    if default_checkpoint_exists:
        st.success("✓ best.pt detected automatically")
        st.caption(f"Using notebook checkpoint: `{DEFAULT_CHECKPOINT}`")
        st.caption("Checkpoint compatibility: Transformers 5.18.0")
        st.caption(f"DeepGlobe dataset: {'available' if DATASET_ROOT.exists() else 'not found'}")
    else:
        st.warning("Checkpoint not found at the default path.")
        st.caption(f"Expected: `{DEFAULT_CHECKPOINT}`")

    checkpoint_file = st.file_uploader(
        "Override checkpoint (.pt / .pth)",
        type=["pt", "pth"],
        help="Optional. If supplied, this checkpoint takes priority over the default best.pt.",
    )

    image_file = st.file_uploader(
        "Satellite image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

    st.markdown("#### Preprocessing")

    experimental_enhancement = st.checkbox(
        "Experimental enhancement",
        value=False,
        help=(
            "Applies light bilateral denoising + CLAHE before inference. "
            "OFF preserves the exact preprocessing used by the supplied "
            "evaluation notebook."
        ),
    )

    if experimental_enhancement:
        st.warning(
            "Experimental mode changes the model input distribution. "
            "Do not use it for direct comparison with the reported research metrics."
        )
    else:
        st.caption(
            "Research-safe mode: RGB → 512×512 → [0,1] → ImageNet normalization."
        )

    gt_file = st.file_uploader(
        "Ground-truth road mask (optional override)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Manual override. Otherwise the dashboard searches the configured DeepGlobe dataset for the matching *_mask file.",
    )

    st.markdown("---")
    st.markdown("#### Evaluation Protocol")
    st.write("Input size: **512 × 512**")
    st.write("Road threshold: **0.55**")
    st.write("TTA: **OFF**")
    st.write(f"Device: **{DEVICE_LABEL}**")

    st.markdown("---")
    st.caption(
        "APLS is the notebook's raster-derived APLS-like metric. "
        "It is not official SpaceNet APLS because DeepGlobe uses raster masks."
    )

if not image_file:
    st.markdown("""
    <div class="start-analysis-card">
        <h3>Start an analysis</h3>
        <p class="small-note">
            Upload a satellite image to begin. The dashboard will automatically
            use <b>C:\\Users\\HP\\Downloads\\best.pt</b> when available.
            You can override it with another <b>.pt</b> checkpoint.
            Add the corresponding ground-truth mask when you need quantitative
            segmentation and topology metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Resolve checkpoint source:
# 1) Explicit upload, if provided.
# 2) Exact best.pt path from the supplied evaluation notebook.
if checkpoint_file is not None:
    checkpoint_source = checkpoint_file.getvalue()
    checkpoint_label = "uploaded checkpoint"
elif DEFAULT_CHECKPOINT.exists():
    checkpoint_source = DEFAULT_CHECKPOINT.read_bytes()
    checkpoint_label = str(DEFAULT_CHECKPOINT)
else:
    st.error(
        "No SatPro checkpoint is available. "
        f"Place best.pt at `{DEFAULT_CHECKPOINT}` or upload a .pt/.pth file."
    )
    st.stop()

# Load model
try:
    with st.spinner(f"Loading SatPro model • {DEVICE_LABEL}..."):
        model, checkpoint = load_model(checkpoint_source)
except Exception as e:
    st.error("Checkpoint could not be loaded with strict SatPro architecture.")
    st.exception(e)
    st.stop()

# Input + preprocessing diagnostics
input_pil = Image.open(image_file).convert("RGB")
original_stats = image_quality_stats(input_pil)

display_image, input_tensor, processed_source = preprocess_image(
    input_pil,
    experimental_enhancement=experimental_enhancement,
)

processed_stats = image_quality_stats(processed_source)

# Inference
with st.spinner("Running SatPro inference..."):
    start = time.perf_counter()

    input_tensor = input_tensor.to(DEVICE)

    with torch.inference_mode():
        outputs = model(input_tensor)
        final_probability = torch.sigmoid(
            outputs["final_logits"]
        )[0, 0].cpu().numpy()

        continuity_probability = torch.sigmoid(
            outputs["continuity_logits"]
        )[0, 0].cpu().numpy()

        boundary_probability = torch.sigmoid(
            outputs["boundary_logits"]
        )[0, 0].cpu().numpy()

    latency_ms = (time.perf_counter() - start) * 1000.0

pred_mask = (
    final_probability > THRESHOLD
).astype(np.uint8)

# Ground-truth metrics
# Priority:
#   1) manually uploaded GT mask
#   2) automatically matched DeepGlobe *_mask file
metrics = None
apls = None
gt_mask = None
gt_source = None

auto_gt_path = find_matching_ground_truth(image_file.name)

if gt_file is not None:
    gt_source = "Manual ground-truth upload"
    gt_pil = Image.open(gt_file).convert("L")
elif auto_gt_path is not None:
    gt_source = f"Automatically matched: {auto_gt_path}"
    gt_pil = Image.open(auto_gt_path).convert("L")

if gt_source is not None:
    try:
        gt = np.asarray(gt_pil)
        gt = cv2.resize(
            gt,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        gt_mask = (gt > 127).astype(np.uint8)
        metrics = binary_metrics(pred_mask, gt_mask)
        apls = raster_derived_apls(
            gt_mask,
            pred_mask,
            num_sample_nodes=40,
            seed=0,
            snap_tolerance_px=20.0,
        )
        st.markdown(
            f'<div class="status-good">✓ Ground truth found · {gt_source}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Ground-truth mask could not be evaluated: {e}")
else:
    st.markdown(
        '<div class="status-warn">Ground truth not found automatically. '
        'Upload the matching road mask if quantitative metrics are required.</div>',
        unsafe_allow_html=True,
    )

# Main metrics
road_coverage = float(pred_mask.mean() * 100.0)
mean_confidence = float(
    final_probability[pred_mask > 0].mean()
) if pred_mask.any() else 0.0

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(container, label, value, note=""):
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

metric_card(
    c1, "IoU",
    f"{metrics['IoU']:.4f}" if metrics else "—",
    "ground truth required"
)
metric_card(
    c2, "F1 / Dice",
    f"{metrics['F1']:.4f}" if metrics else "—",
    "ground truth required"
)
metric_card(
    c3, "Precision",
    f"{metrics['Precision']:.4f}" if metrics else "—",
    "ground truth required"
)
metric_card(
    c4, "Recall",
    f"{metrics['Recall']:.4f}" if metrics else "—",
    "ground truth required"
)
metric_card(
    c5, "Latency",
    f"{latency_ms:.0f} ms",
    "this uploaded image"
)

# ------------------------- RESULTS PROCESSING + DATA REPRESENTATION -------------------------
st.markdown('<div class="section">Processing pipeline · from pixels to road intelligence</div>', unsafe_allow_html=True)

proc_cols = st.columns(6)
proc_items = [
    ("01", "Input", f"{original_stats['width']} × {original_stats['height']}", "Satellite RGB"),
    ("02", "Preprocess", "512 × 512", "Resize + normalize"),
    ("03", "Inference", "4 outputs", "Road + continuity + boundary + final"),
    ("04", "Decision", f"τ = {THRESHOLD:.2f}", "Probability → binary mask"),
    ("05", "Topology", "Skeleton + graph", "Connectivity analysis"),
    ("06", "Evaluate", "GT comparison", "IoU / F1 / P / R / APLS"),
]
for col, (num, title, value, desc) in zip(proc_cols, proc_items):
    with col:
        st.markdown(
            f'''
            <div class="process-step">
                <div class="process-num">{num}</div>
                <div class="process-title">{title}</div>
                <div class="data-value">{value}</div>
                <div class="process-desc">{desc}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

st.markdown("### Result processing")
result_cols = st.columns(4)
result_items = [
    ("Road probability", f"{float(final_probability.min()):.3f} – {float(final_probability.max()):.3f}", "Pixel-wise sigmoid output"),
    ("Detected pixels", f"{int(pred_mask.sum()):,}", "Pixels above threshold"),
    ("Road coverage", f"{road_coverage:.2f}%", "Detected pixels / image area"),
    ("Mean confidence", f"{mean_confidence:.3f}", "Mean probability of detected road pixels"),
]
for col, (label, value, desc) in zip(result_cols, result_items):
    with col:
        st.markdown(
            f'''
            <div class="data-card">
                <div class="data-label">{label}</div>
                <div class="data-value">{value}</div>
                <div class="data-desc">{desc}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

st.markdown(
    f'''
    <div class="result-strip">
        <b>Processing interpretation:</b>
        The network converts the satellite image into a continuous road-probability field.
        A fixed threshold of <b>{THRESHOLD:.2f}</b> converts that field into the binary road mask.
        The mask is then suitable for pixel-level evaluation and topology-oriented processing,
        including centerline extraction, graph construction and the raster-derived APLS calculation
        when a ground-truth mask is available.
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown("### Data representation")
d1, d2 = st.columns([1, 1], gap="large")

with d1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Model data flow")
    data_rows = [
        ["Input", "RGB satellite image", f"{original_stats['width']} × {original_stats['height']}"],
        ["Tensor", "Normalized CHW tensor", "1 × 3 × 512 × 512"],
        ["Prediction", "Road probability", "512 × 512"],
        ["Decision", "Binary road mask", "512 × 512"],
        ["Auxiliary", "Continuity / boundary", "512 × 512 each"],
    ]
    st.table(data_rows)
    st.markdown('</div>', unsafe_allow_html=True)

with d2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Quantitative representation")
    qdata = {
        "Quantity": [
            "Input brightness", "Input contrast", "Input sharpness",
            "Road probability mean", "Road probability max",
            "Road coverage", "Inference latency",
        ],
        "Value": [
            f"{original_stats['brightness']:.4f}",
            f"{original_stats['contrast']:.4f}",
            f"{original_stats['sharpness']:.2f}",
            f"{float(final_probability.mean()):.4f}",
            f"{float(final_probability.max()):.4f}",
            f"{road_coverage:.2f}%",
            f"{latency_ms:.1f} ms",
        ],
        "Interpretation": [
            "Normalized mean intensity",
            "Normalized intensity spread",
            "Laplacian variance",
            "Average pixel probability",
            "Highest road probability",
            "Detected road area",
            "Single-image processing time",
        ],
    }
    st.dataframe(qdata, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="section">Current image · evaluation</div>', unsafe_allow_html=True)

left, right = st.columns([1.35, .85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### Preprocessing pipeline")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            '<div class="image-card"><div class="image-title">01 · RGB conversion</div>'
            '<div class="image-subtitle">3-channel RGB input</div></div>',
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            '<div class="image-card"><div class="image-title">02 · Resize</div>'
            '<div class="image-subtitle">512 × 512 · bilinear</div></div>',
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            '<div class="image-card"><div class="image-title">03 · Scaling</div>'
            '<div class="image-subtitle">Pixel range → [0, 1]</div></div>',
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            '<div class="image-card"><div class="image-title">04 · Normalize</div>'
            '<div class="image-subtitle">ImageNet mean / std</div></div>',
            unsafe_allow_html=True,
        )

    q1, q2, q3 = st.columns(3)
    metric_card(q1, "Input size", f"{original_stats['width']} × {original_stats['height']}", "uploaded image")
    metric_card(q2, "Brightness", f"{original_stats['brightness']:.3f}", "mean normalized intensity")
    metric_card(q3, "Sharpness", f"{original_stats['sharpness']:.1f}", "Laplacian variance")

    if experimental_enhancement:
        st.markdown(
            '<div class="research-note">'
            '<b>Experimental preprocessing enabled:</b> light bilateral denoising '
            '+ luminance CLAHE. The resulting image is then passed through the '
            'same 512×512 resize and ImageNet normalization pipeline.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Visual analysis")
    st.caption("Four complementary views of the uploaded satellite image.")

    v1, v2 = st.columns(2, gap="medium")
    with v1:
        st.markdown(
            '<div class="image-card"><div class="image-title">01 · Satellite image</div>'
            '<div class="image-subtitle">Model input after preprocessing</div>',
            unsafe_allow_html=True,
        )
        st.image(display_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with v2:
        st.markdown(
            '<div class="image-card"><div class="image-title">02 · Extracted road mask</div>'
            '<div class="image-subtitle">Black = background · White = detected road</div>',
            unsafe_allow_html=True,
        )
        st.image(binary_mask_rgb(pred_mask), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    v3, v4 = st.columns(2, gap="medium")
    with v3:
        st.markdown(
            '<div class="image-card"><div class="image-title">03 · Probability overlay</div>'
            '<div class="image-subtitle">Road probability blended directly over the satellite image</div>',
            unsafe_allow_html=True,
        )
        st.image(make_probability_overlay(display_image, final_probability), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with v4:
        st.markdown(
            '<div class="image-card"><div class="image-title">04 · Probability heatmap</div>'
            '<div class="image-subtitle">Black = near-zero probability · brighter = higher road probability</div>',
            unsafe_allow_html=True,
        )
        st.image(black_probability_heatmap(final_probability), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Image metrics")

    metric_card(
        st.container(), "Road coverage",
        f"{road_coverage:.2f}%",
        "predicted road pixels"
    )
    metric_card(
        st.container(), "Mean confidence",
        f"{mean_confidence:.3f}",
        "detected road pixels"
    )
    metric_card(
        st.container(), "Inference latency",
        f"{latency_ms:.0f} ms",
        "single uploaded image"
    )
    metric_card(
        st.container(), "Decision threshold",
        f"{THRESHOLD:.2f}",
        "fixed notebook threshold"
    )

    if metrics:
        st.markdown("#### Ground-truth comparison")
        q1, q2 = st.columns(2)
        metric_card(q1, "IoU", f"{metrics['IoU']:.4f}", "pixel-level")
        metric_card(q2, "F1 / Dice", f"{metrics['F1']:.4f}", "pixel-level")
        q3, q4 = st.columns(2)
        metric_card(q3, "Precision", f"{metrics['Precision']:.4f}", "pixel-level")
        metric_card(q4, "Recall", f"{metrics['Recall']:.4f}", "pixel-level")
        metric_card(
            st.container(), "APLS",
            f"{apls:.4f}" if apls is not None else "—",
            "raster-derived · GT required"
        )
    else:
        st.markdown(
            '<div class="research-note">'
            '<b>Ground-truth comparison unavailable.</b><br>'
            'Upload the matching road mask to calculate IoU, F1/Dice, '
            'Precision, Recall and the notebook’s raster-derived APLS.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Auxiliary model outputs"):
        st.image(
            probability_rgb(continuity_probability),
            caption="Continuity-head probability",
            use_container_width=True,
        )
        st.image(
            probability_rgb(boundary_probability),
            caption="Boundary-head probability",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# Detailed metrics
# Compact interpretation before research metrics
st.markdown('<div class="section">Interpretation</div>', unsafe_allow_html=True)

if metrics:
    interpretation = (
        f"Detected road pixels cover <b>{road_coverage:.2f}%</b> of the image with a mean "
        f"road confidence of <b>{mean_confidence:.3f}</b>. Against the supplied ground truth, "
        f"the segmentation reaches <b>IoU {metrics['IoU']:.4f}</b> and "
        f"<b>F1/Dice {metrics['F1']:.4f}</b>, while topology is additionally summarized "
        f"by the raster-derived APLS value <b>{apls:.4f}</b> when available."
    )
else:
    interpretation = (
        f"The model detected road pixels across <b>{road_coverage:.2f}%</b> of the image "
        f"with mean detected-road confidence <b>{mean_confidence:.3f}</b>. Quantitative "
        f"segmentation metrics remain pending until the corresponding ground-truth mask is supplied."
    )

st.markdown(f'<div class="result-strip">{interpretation}</div>', unsafe_allow_html=True)


st.markdown('<div class="section">Research evaluation</div>', unsafe_allow_html=True)

if metrics:
    st.markdown(
        '<div class="research-note">'
        '<b>Per-image quantitative evaluation is active.</b> '
        'The values below compare this uploaded image against its uploaded '
        'ground-truth road mask at 512 × 512. APLS follows the raster-derived '
        'implementation used by the supplied evaluation notebook.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="research-note">'
        '<b>Inference-only mode.</b> '
        'The uploaded image has been segmented successfully. '
        'IoU, F1/Dice, Precision, Recall and APLS require its matching '
        'ground-truth road mask.'
        '</div>',
        unsafe_allow_html=True,
    )

# Export
st.markdown('<div class="section">Export</div>', unsafe_allow_html=True)

mask_png = cv2.imencode(
    ".png",
    (pred_mask * 255).astype(np.uint8)
)[1].tobytes()

overlay_png = cv2.imencode(
    ".png",
    cv2.cvtColor(
        make_probability_overlay(display_image, final_probability),
        cv2.COLOR_RGB2BGR,
    )
)[1].tobytes()

heatmap_png = cv2.imencode(
    ".png",
    cv2.cvtColor(
        black_probability_heatmap(final_probability),
        cv2.COLOR_RGB2BGR,
    )
)[1].tobytes()

e1, e2, e3, e4 = st.columns(4)
with e1:
    st.download_button(
        "Download road mask",
        data=mask_png,
        file_name="satpro_road_mask.png",
        mime="image/png",
        use_container_width=True,
    )
with e2:
    st.download_button(
        "Download probability overlay",
        data=overlay_png,
        file_name="satpro_probability_overlay.png",
        mime="image/png",
        use_container_width=True,
    )
with e3:
    st.download_button(
        "Download heatmap",
        data=heatmap_png,
        file_name="satpro_probability_heatmap.png",
        mime="image/png",
        use_container_width=True,
    )
with e4:
    report = {
        "model": "SatPro",
        "encoder": "nvidia/mit-b0",
        "input_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "tta": False,
        "latency_ms": latency_ms,
        "road_coverage_percent": road_coverage,
        "mean_road_confidence": mean_confidence,
        "IoU": metrics["IoU"] if metrics else None,
        "F1": metrics["F1"] if metrics else None,
        "Precision": metrics["Precision"] if metrics else None,
        "Recall": metrics["Recall"] if metrics else None,
        "APLS_raster_derived": apls,
        "image_specific_metrics": {
            "road_coverage_percent": road_coverage,
            "mean_road_confidence": mean_confidence,
            "latency_ms": latency_ms,
            "threshold": THRESHOLD,
            "ground_truth_source": gt_source,
            "experimental_enhancement": experimental_enhancement,
            "preprocessing": {
                "input_size": [IMAGE_SIZE, IMAGE_SIZE],
                "resize_interpolation": "bilinear",
                "scale": "[0,1]",
                "normalization": "ImageNet mean/std",
                "enhancement": "bilateral_filter + CLAHE" if experimental_enhancement else "none",
            },
        },
    }
    import json
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="satpro_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "SatPro • Satellite Road Intelligence Dashboard • "
    "Architecture and evaluation protocol aligned with the supplied notebook • "
    "Threshold 0.55 • TTA OFF"
)
