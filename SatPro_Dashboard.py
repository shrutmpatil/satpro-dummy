
import gc
import math
import pathlib
from pathlib import Path
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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@1,600&display=swap');
:root{--bg:#0b1017;--surface:#101923;--surface2:#0d141d;--text:#eee9e1;--muted:#8b9aaa;--line:#202a35;--accent:#d98253;--accent2:#f0a16f}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text)}
.stApp{background:radial-gradient(ellipse at 78% 12%,rgba(27,67,88,.22),transparent 34%),var(--bg);color:var(--text)}
.block-container{max-width:1440px;padding:1.2rem 2.5rem 3rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0e151e;border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--text)!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#101923;border:1px dashed #34414d;border-radius:8px}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{background:#101923}
h1,h2,h3,h4{font-family:'DM Sans',sans-serif!important;letter-spacing:-.035em;color:var(--text)}
p,li{color:#b5c0cb}
a{color:var(--accent)}
hr{border-color:var(--line)}
.hero{position:relative;overflow:hidden;background:transparent;border:0;border-bottom:1px solid var(--line);border-radius:0;padding:10px 0 38px;margin:0 0 26px;box-shadow:none;color:var(--text)}
.brand-row{display:flex;align-items:center;gap:12px;padding:0 0 17px;border-bottom:1px solid var(--line);margin-bottom:44px;position:relative;z-index:2}
.brand-mark{width:34px;height:34px;border:1px solid #75472f;border-radius:50%;display:flex;align-items:center;justify-content:center;color:var(--accent);font-size:12px;font-weight:700;background:transparent}
.brand-name{font-size:13px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--text)}
.brand-meta{font-family:'DM Mono',monospace;font-size:8px;letter-spacing:.18em;text-transform:uppercase;color:#8292a3;margin-top:4px}
.hero-grid{display:grid;grid-template-columns:1fr 1.08fr;gap:40px;align-items:center;position:relative;z-index:1}
.hero-kicker,.section{font-family:'DM Mono',monospace;color:var(--accent);font-size:10px;letter-spacing:.16em;text-transform:uppercase}
.hero-title{font-size:clamp(2.6rem,4.4vw,4.5rem);line-height:1.02;font-weight:500;letter-spacing:-.065em;margin:18px 0;color:var(--text)}
.hero-title em{font-family:'Playfair Display',serif;color:var(--accent);font-weight:600}
.hero-sub{font-size:13px;line-height:1.8;color:#9aa8b6;max-width:470px}
.hero-art{position:relative;min-height:260px;border:1px solid #243341;background:#08111b;overflow:hidden;isolation:isolate}
.hero-art img{position:relative;z-index:0}
.hero-art::before{
    content:"";position:absolute;z-index:1;inset:-32% -18%;
    border:1px solid rgba(73,174,235,.22);border-radius:50%;
    transform:rotate(-17deg) scale(.78);
    box-shadow:0 0 24px rgba(38,151,226,.06), inset 0 0 18px rgba(38,151,226,.035);
    animation:satpro-orbit 18s linear infinite;
    pointer-events:none
}
.hero-art::after{
    content:"";position:absolute;z-index:1;inset:0;
    background:linear-gradient(115deg,transparent 35%,rgba(91,190,255,.07) 49%,rgba(255,166,103,.09) 51%,transparent 64%);
    transform:translateX(-115%);
    animation:satpro-scan 9s ease-in-out infinite;
    mix-blend-mode:screen;pointer-events:none
}
@keyframes satpro-orbit{
    0%{transform:rotate(-17deg) scale(.78)}
    50%{transform:rotate(163deg) scale(.82)}
    100%{transform:rotate(343deg) scale(.78)}
}
@keyframes satpro-scan{
    0%,15%{transform:translateX(-115%);opacity:0}
    30%{opacity:.75}
    65%{transform:translateX(115%);opacity:.55}
    78%,100%{transform:translateX(115%);opacity:0}
}
.hero-art-label{z-index:2}
@media(prefers-reduced-motion:reduce){
 .hero-art::before,.hero-art::after{animation:none!important}
}
.hero-art img{width:100%;height:300px;object-fit:cover;display:block;opacity:.93}
.hero-art-label{position:absolute;left:14px;bottom:14px;background:rgba(9,16,24,.9);border:1px solid #293746;padding:10px 13px;font-size:11px;color:#dfe7ef}
.protocol-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:23px}
.badge{display:inline-flex;padding:6px 9px;border:1px solid #2a3541;background:#111a24;color:#9eacb9;font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.04em}
.section{margin:28px 0 12px}
.panel,.metric-card,.process-step,.data-card,.image-card{background:linear-gradient(145deg,#101923,#0d141c);border:1px solid #25313e;border-radius:7px;box-shadow:none}
.panel{padding:16px}
.metric-card{padding:14px;min-height:91px}
.metric-label,.data-label{font-family:'DM Mono',monospace;color:#8a9aab;font-size:9px;letter-spacing:.1em;text-transform:uppercase}
.metric-value,.data-value{font-size:1.4rem;color:var(--text);font-weight:600;margin-top:8px;letter-spacing:-.04em}
.metric-note,.data-desc{font-size:10px;color:#8393a3;margin-top:4px}
.process-step{min-height:110px;padding:14px}
.process-num{font-family:'DM Mono',monospace;color:var(--accent);font-size:10px}
.process-title{color:#e9e6df;font-size:13px;font-weight:600;margin-top:8px}
.process-desc{color:#8e9cab;font-size:10px;line-height:1.6;margin-top:5px}
.result-strip,.research-note{background:#111a24;border:1px solid #293744;border-left:2px solid var(--accent);border-radius:4px;padding:13px 15px;color:#aab7c4;font-size:12px;line-height:1.7;margin:12px 0}
.image-card{padding:10px;margin-bottom:10px}
.image-title{font-size:12px;font-weight:600;color:#e5e4df;margin:2px 2px 8px}
.image-subtitle{font-size:10px;color:#8595a5;margin:0 2px 9px}
.small-note{font-size:12px;color:#98a7b6;line-height:1.7}
.stButton>button,.stDownloadButton>button,button[kind="primary"]{background:var(--accent)!important;color:#17120f!important;border:1px solid var(--accent)!important;border-radius:2px!important;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--accent2)!important;border-color:var(--accent2)!important}
div[data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
button[data-baseweb="tab"]{color:#9aa8b6;font-size:12px}
button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:5px;overflow:hidden}
[data-testid="stAlert"]{background:#111a24;border:1px solid #293744;color:var(--text)}
div[data-testid="stImage"]{border-radius:4px;overflow:hidden}
@media(max-width:850px){.block-container{padding:1rem 1rem 2rem}.hero-grid{grid-template-columns:1fr;gap:22px}.hero-art img{height:220px}.brand-row{margin-bottom:28px}.hero-title{font-size:2.8rem}}


.brand-mark.satpro-logo{width:48px;height:48px;border:0;border-radius:0;background:transparent;box-shadow:none;overflow:hidden;flex-shrink:0}
.brand-mark.satpro-logo img{width:100%;height:100%;object-fit:contain;display:block;border-radius:0}

/* High-contrast Streamlit file uploaders */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {
    background: #101923 !important;
    color: #eee9e1 !important;
    border-color: #34414d !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * {
    color: #eee9e1 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
[data-testid="stFileUploader"] span {
    color: #eee9e1 !important;
}
[data-testid="stFileUploader"] button {
    background: #202b37 !important;
    color: #f3eee7 !important;
    border: 1px solid #465463 !important;
}
[data-testid="stFileUploader"] svg {
    fill: #d98253 !important;
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

    # Checkpoints may contain pathlib path objects serialized on Windows.
    # On Linux, map any serialized WindowsPath references to PosixPath.
    # Do NOT map PosixPath to WindowsPath: WindowsPath cannot be instantiated
    # on Linux and causes UnsupportedOperation during torch.load().
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


_SATPRO_LOGO_DATA = "data:image/jpeg;base64,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"

# Optional hero image is bundled beside this script; if absent, the hero remains styled.
_HERO_IMAGE_PATH = pathlib.Path(__file__).with_name("satellite-earth-night.png")
_HERO_IMAGE_HTML = ""
if _HERO_IMAGE_PATH.exists():
    import base64
    _hero_b64 = base64.b64encode(_HERO_IMAGE_PATH.read_bytes()).decode("ascii")
    _HERO_IMAGE_HTML = f'<img src="data:image/png;base64,{_hero_b64}" alt="Earth at night with illuminated road-like networks"/>'

st.markdown(f"""
<div class="hero">
  <div class="brand-row">
    <div class="brand-mark satpro-logo"><img src="{_SATPRO_LOGO_DATA}" alt="SatPro logo"/></div>
    <div><div class="brand-name">SATPRO</div><div class="brand-meta">Earth Intelligence</div></div>
  </div>
  <div class="hero-grid">
    <div>
      <div class="hero-kicker">● Geospatial intelligence platform</div>
      <div class="hero-title">See the roads.<br><em>Understand</em> the<br>network.</div>
      <div class="hero-sub">Turn satellite imagery into structured road intelligence with deep-learning models built for extraction, analysis and connectivity.</div>
      <div class="protocol-row">
        <span class="badge">ROUTEFORMER-RC</span><span class="badge">ROAD NETWORK INTELLIGENCE</span>
      </div>
    </div>
    <div class="hero-art">{_HERO_IMAGE_HTML}<div class="hero-art-label"><span style="color:#d98253;font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em">LIVE ORBITAL VIEW / 01</span><br><b>✳ Earth observation · Road intelligence</b></div></div>
  </div>
  <div class="protocol-row">
    <span class="badge">MiT-B0 / SEGFORMER</span><span class="badge">512 × 512</span><span class="badge">THRESHOLD 0.55</span><span class="badge">TTA OFF</span><span class="badge">AUTO GPU / CPU</span>
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
        st.info("Upload your trained best.pt checkpoint below to enable inference.")

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
