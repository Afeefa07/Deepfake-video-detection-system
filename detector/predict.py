import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import mediapipe as mp

from scipy.signal import butter, filtfilt, welch


# =========================
# PATHS / DEVICE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_BEST = os.path.join(BASE_DIR, "models", "final_visual_rppg_fusion_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(FUSION_BEST):
    raise FileNotFoundError(f"Missing checkpoint: {FUSION_BEST}")


# =========================
# CONFIG
# =========================
T = 16
IMG_SIZE = 224
N_CLIPS = 5

BANDPASS_LOW = 0.7
BANDPASS_HIGH = 4.0
DEFAULT_FPS = 25.0
MIN_FPS = 10.0

ROI_MIN_PIXELS = 80
SAT_LOW = 15
SAT_HIGH = 240

FEATURES = [
    "bpm_mean",
    "bpm_std",
    "peak_ratio_mean",
    "peak_ratio_std",
    "roi_sync_mean",
    "roi_sync_std",
    "snr_forehead_mean",
    "snr_left_mean",
    "snr_right_mean",
    "spectral_entropy_mean",
    "roi_invalid_gap",
    "window_success_ratio",
]

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# =========================
# MEDIAPIPE INIT
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


# =========================
# SIGNAL HELPERS
# =========================
def bandpass_filter(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=3):
    if len(sig) < 16 or fs <= 0:
        return sig

    nyq = 0.5 * fs
    low = max(0.01, low / nyq)
    high = min(0.99, high / nyq)

    if high <= low:
        return sig

    b, a = butter(order, [low, high], btype="band")
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def interp_nan_1d(x):
    x = x.copy()
    nans = np.isnan(x)
    if np.all(nans):
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def interp_nan_2d(arr):
    arr = arr.copy()
    for c in range(arr.shape[1]):
        arr[:, c] = interp_nan_1d(arr[:, c])
    return arr


def pos_rppg(rgb_trace, fs):
    eps = 1e-6
    C = rgb_trace.T
    meanC = np.mean(C, axis=1, keepdims=True) + eps
    Cn = C / meanC - 1.0

    P = np.array([[0, 1, -1],
                  [-2, 1, 1]], dtype=np.float32)

    S = P @ Cn
    s1, s2 = S[0], S[1]

    alpha = np.std(s1) / (np.std(s2) + eps)
    h = s1 + alpha * s2
    h = h - np.mean(h)

    return bandpass_filter(h, fs)


def estimate_bpm(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    if len(sig) < max(32, int(fs * 2)):
        return 0.0

    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= low) & (f <= high)
    if mask.sum() == 0:
        return 0.0

    f2 = f[mask]
    p2 = p[mask]
    if len(p2) == 0 or np.all(p2 <= 0):
        return 0.0

    return float(f2[np.argmax(p2)] * 60.0)


def peak_ratio(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= low) & (f <= high)
    if mask.sum() == 0:
        return 0.0

    p2 = p[mask]
    if len(p2) == 0:
        return 0.0

    pk = float(np.max(p2))
    avg = float(np.mean(p2)) + 1e-6
    return pk / avg


def snr_like(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= low) & (f <= high)
    if mask.sum() == 0:
        return 0.0

    p2 = p[mask]
    if len(p2) == 0:
        return 0.0

    peak_idx = int(np.argmax(p2))
    peak_power = float(p2[peak_idx])

    exclude = max(1, len(p2) // 20)
    bg = np.concatenate([
        p2[:max(0, peak_idx - exclude)],
        p2[min(len(p2), peak_idx + exclude + 1):]
    ])
    bg_power = float(np.mean(bg)) if len(bg) > 0 else 1e-6

    return peak_power / (bg_power + 1e-6)


def spectral_entropy(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= low) & (f <= high)
    if mask.sum() == 0:
        return 0.0

    p2 = p[mask]
    if len(p2) == 0:
        return 0.0

    p2 = p2 / (np.sum(p2) + 1e-8)
    return float(-np.sum(p2 * np.log(p2 + 1e-8)))


def safe_corr(a, b):
    if len(a) < 5 or len(b) < 5:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


# =========================
# ROI HELPERS
# =========================
FOREHEAD_IDS = [10, 67, 103, 109, 338, 332, 297, 284]
LEFT_CHEEK_IDS = [50, 101, 118, 119, 120, 121, 205, 187]
RIGHT_CHEEK_IDS = [280, 330, 347, 348, 349, 350, 425, 411]


def mediapipe_landmarks(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    return np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)


def polygon_mask(shape_hw, pts):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def inward_poly(poly, scale=0.90):
    c = poly.mean(axis=0, keepdims=True)
    p = c + scale * (poly - c)
    return p.astype(np.int32)


def roi_masks_from_landmarks(frame_bgr, landmarks_xy):
    h, w = frame_bgr.shape[:2]

    forehead = inward_poly(landmarks_xy[FOREHEAD_IDS], 0.88)
    left_cheek = inward_poly(landmarks_xy[LEFT_CHEEK_IDS], 0.88)
    right_cheek = inward_poly(landmarks_xy[RIGHT_CHEEK_IDS], 0.88)

    return {
        "forehead": polygon_mask((h, w), forehead),
        "left": polygon_mask((h, w), left_cheek),
        "right": polygon_mask((h, w), right_cheek),
    }


def filtered_mean_rgb(frame_bgr, mask):
    if mask is None or mask.sum() < ROI_MIN_PIXELS:
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    vals = rgb[mask]

    if len(vals) < ROI_MIN_PIXELS:
        return None

    good = np.all((vals >= SAT_LOW) & (vals <= SAT_HIGH), axis=1)
    vals = vals[good]

    if len(vals) < ROI_MIN_PIXELS:
        return None

    if np.mean(np.std(vals.astype(np.float32), axis=0)) < 2.0:
        return None

    return vals.mean(axis=0)


# =========================
# RPPG FEATURE EXTRACTION
# =========================
def extract_window_features(traces_rgb, fs):
    sig_f = pos_rppg(traces_rgb["forehead"], fs)
    sig_l = pos_rppg(traces_rgb["left"], fs)
    sig_r = pos_rppg(traces_rgb["right"], fs)

    bpm_vals = np.array([
        estimate_bpm(sig_f, fs),
        estimate_bpm(sig_l, fs),
        estimate_bpm(sig_r, fs)
    ], dtype=np.float32)

    peak_vals = np.array([
        peak_ratio(sig_f, fs),
        peak_ratio(sig_l, fs),
        peak_ratio(sig_r, fs)
    ], dtype=np.float32)

    sync_vals = np.array([
        safe_corr(sig_f, sig_l),
        safe_corr(sig_f, sig_r),
        safe_corr(sig_l, sig_r)
    ], dtype=np.float32)

    snr_vals = np.array([
        snr_like(sig_f, fs),
        snr_like(sig_l, fs),
        snr_like(sig_r, fs)
    ], dtype=np.float32)

    entropy_vals = np.array([
        spectral_entropy(sig_f, fs),
        spectral_entropy(sig_l, fs),
        spectral_entropy(sig_r, fs)
    ], dtype=np.float32)

    return {
        "bpm_mean": float(np.mean(bpm_vals)),
        "peak_ratio_mean": float(np.mean(peak_vals)),
        "roi_sync_mean": float(np.mean(sync_vals)),
        "snr_forehead_mean": float(snr_vals[0]),
        "snr_left_mean": float(snr_vals[1]),
        "snr_right_mean": float(snr_vals[2]),
        "spectral_entropy_mean": float(np.mean(entropy_vals)),
    }


def extract_rppg_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for rPPG extraction")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps < MIN_FPS or fps > 120:
        fps = DEFAULT_FPS

    rgb_traces = {"forehead": [], "left": [], "right": []}
    invalid_counts = {"forehead": 0, "left": 0, "right": 0}
    total_read = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = mediapipe_landmarks(frame)
        if lm is None:
            for k in rgb_traces:
                rgb_traces[k].append([np.nan, np.nan, np.nan])
                invalid_counts[k] += 1
            total_read += 1
            continue

        masks = roi_masks_from_landmarks(frame, lm)

        for k in ["forehead", "left", "right"]:
            mean_rgb = filtered_mean_rgb(frame, masks[k])
            if mean_rgb is None:
                rgb_traces[k].append([np.nan, np.nan, np.nan])
                invalid_counts[k] += 1
            else:
                rgb_traces[k].append(mean_rgb)

        total_read += 1

    cap.release()

    if total_read < 32:
        raise RuntimeError("Too few usable frames for rPPG")

    for k in rgb_traces:
        rgb_traces[k] = np.array(rgb_traces[k], dtype=np.float32)

    forehead_invalid_frac = invalid_counts["forehead"] / max(1, total_read)
    left_invalid_frac = invalid_counts["left"] / max(1, total_read)
    right_invalid_frac = invalid_counts["right"] / max(1, total_read)

    win_len = max(int(3.0 * fps), 32)
    stride = max(int(1.0 * fps), 8)

    n = total_read
    starts = list(range(0, max(1, n - win_len + 1), stride))
    window_feats = []

    for st in starts:
        en = min(n, st + win_len)
        if en - st < max(32, int(2 * fps)):
            continue

        sub = {}
        sub_invalid = False

        for k in ["forehead", "left", "right"]:
            raw_sub = rgb_traces[k][st:en]
            invalid_frac = float(np.isnan(raw_sub).any(axis=1).mean())

            if invalid_frac > 0.50:
                sub_invalid = True
                break

            sub[k] = interp_nan_2d(raw_sub)

        if sub_invalid:
            continue

        feat = extract_window_features(sub, fps)
        window_feats.append(feat)

    if len(window_feats) < 1:
        raise RuntimeError("No valid rPPG windows extracted")

    def agg(name):
        vals = np.array([wf[name] for wf in window_feats], dtype=np.float32)
        return float(np.mean(vals)), float(np.std(vals))

    bpm_mean, bpm_std = agg("bpm_mean")
    peak_ratio_mean, peak_ratio_std = agg("peak_ratio_mean")
    roi_sync_mean, roi_sync_std = agg("roi_sync_mean")
    spectral_entropy_mean, _ = agg("spectral_entropy_mean")

    snr_forehead_mean = float(np.mean([wf["snr_forehead_mean"] for wf in window_feats]))
    snr_left_mean = float(np.mean([wf["snr_left_mean"] for wf in window_feats]))
    snr_right_mean = float(np.mean([wf["snr_right_mean"] for wf in window_feats]))

    window_success_ratio = len(window_feats) / max(1, len(starts))

    return {
        "bpm_mean": bpm_mean,
        "bpm_std": bpm_std,
        "peak_ratio_mean": peak_ratio_mean,
        "peak_ratio_std": peak_ratio_std,
        "roi_sync_mean": roi_sync_mean,
        "roi_sync_std": roi_sync_std,
        "snr_forehead_mean": snr_forehead_mean,
        "snr_left_mean": snr_left_mean,
        "snr_right_mean": snr_right_mean,
        "spectral_entropy_mean": spectral_entropy_mean,
        "roi_invalid_gap": float(abs(left_invalid_frac - right_invalid_frac)),
        "window_success_ratio": float(window_success_ratio),
        "total_frames_read": int(total_read),
    }


# =========================
# VISUAL BRANCH HELPERS
# =========================
def detect_best_face_mp(frame_rgb):
    h, w = frame_rgb.shape[:2]
    results = face_detector.process(frame_rgb)

    if not results.detections:
        return None

    best_box = None
    best_area = -1

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    return best_box


def crop_face_from_box(frame_rgb, box):
    x1, y1, x2, y2 = box
    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face


def load_faces_for_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video")

    faces = []
    last_box = None
    total_frames = 0
    detected_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = detect_best_face_mp(frame_rgb)

        if box is None:
            box = last_box
        else:
            last_box = box
            detected_faces += 1

        if box is None:
            continue

        face = crop_face_from_box(frame_rgb, box)
        if face is not None:
            faces.append(face)

    cap.release()

    if len(faces) == 0:
        raise RuntimeError("No face crops found for visual stream")

    return np.array(faces), total_frames, detected_faces


def sample_uniform_clips(faces, T=16, n_clips=5):
    n = len(faces)
    clips = []

    if n <= T:
        idx = np.linspace(0, n - 1, T).astype(int)
        clips.append(faces[idx])
        return clips

    max_start = max(0, n - T)
    starts = np.linspace(0, max_start, n_clips).astype(int)

    for st in starts:
        idx = np.arange(st, min(st + T, n))
        if len(idx) < T:
            pad = np.full((T - len(idx),), idx[-1], dtype=int)
            idx = np.concatenate([idx, pad])
        clips.append(faces[idx])

    return clips


def preprocess_clip(clip_np):
    frames = []
    for frame in clip_np:
        x = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        x = (x - IMNET_MEAN) / IMNET_STD
        frames.append(x)

    x = torch.stack(frames, dim=0).unsqueeze(0)
    return x


# =========================
# MODEL DEFINITIONS
# =========================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class NestTemporal(nn.Module):
    def __init__(self, T=16):
        super().__init__()

        self.backbone = timm.create_model(
            "jx_nest_small",
            pretrained=False,
            num_classes=0
        )

        dim = self.backbone.num_features
        self.pos = nn.Parameter(torch.zeros(1, T, dim))

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

        self.norm = nn.LayerNorm(dim)
        self.fake_head = nn.Linear(dim, 2)
        self.domain_head = nn.Linear(dim, 4)

    def forward(self, x, lam=0.0):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        f = self.backbone(x)
        f = f.view(B, T, -1)

        f = f + self.pos[:, :T]
        f = self.temporal(f)

        pooled = self.norm(f).mean(dim=1)
        fake_logits = self.fake_head(pooled)

        rev = GradReverse.apply(pooled, lam)
        domain_logits = self.domain_head(rev)

        return fake_logits, domain_logits, f, pooled


class VisualRPPGFusion(nn.Module):
    def __init__(self, visual_model, visual_dim=384, rppg_dim=12, fused_dim=128):
        super().__init__()
        self.visual_model = visual_model

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.rppg_proj = nn.Sequential(
            nn.Linear(rppg_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x, rppg):
        v_fake_logits, _, _, visual_feat = self.visual_model(x, lam=0.0)

        v = self.visual_proj(visual_feat)
        r = self.rppg_proj(rppg)

        g = self.gate(torch.cat([v, r], dim=1))
        fused = g * v + (1.0 - g) * r

        logits = self.head(fused)
        rppg_logits = self.head(r)
        return logits, g, v, r, v_fake_logits, rppg_logits


# =========================
# LOAD MODEL ONCE
# =========================
ck = torch.load(FUSION_BEST, map_location=device)
visual_model = NestTemporal(T=T).to(device)
fusion_model = VisualRPPGFusion(visual_model).to(device)
fusion_model.load_state_dict(ck["model"])
fusion_model.eval()

rppg_mean = np.array(list(ck["rppg_mean"].values()), dtype=np.float32)
rppg_std = np.array(list(ck["rppg_std"].values()), dtype=np.float32)


# =========================
# DJANGO ENTRY & VISUALIZATION
# =========================
import uuid

def generate_visualization(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Find a frame with a face
    for _ in range(30): # check first 30 frames
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = detect_best_face_mp(frame_rgb)
        lm = mediapipe_landmarks(frame)
        
        if box is not None and lm is not None:
            # Draw box
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw ROIs
            forehead = inward_poly(lm[FOREHEAD_IDS], 0.88)
            left_cheek = inward_poly(lm[LEFT_CHEEK_IDS], 0.88)
            right_cheek = inward_poly(lm[RIGHT_CHEEK_IDS], 0.88)
            
            # Draw Bounding Boxes instead of Polylines for ROIs to make it look cleaner
            for roi_pts, color, label in [(forehead, (255, 0, 0), "Forehead"), 
                                          (left_cheek, (0, 0, 255), "L-Cheek"), 
                                          (right_cheek, (0, 0, 255), "R-Cheek")]:
                rx, ry, rw, rh = cv2.boundingRect(roi_pts.astype(np.int32))
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 2)
                cv2.putText(frame, label, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            filename = f"vis_{uuid.uuid4().hex[:8]}.jpg"
            static_dir = os.path.join(BASE_DIR, "detector", "static", "results")
            os.makedirs(static_dir, exist_ok=True)
            out_path = os.path.join(static_dir, filename)
            cv2.imwrite(out_path, frame)
            cap.release()
            return f"/static/results/{filename}"

    cap.release()
    return None

def predict_video(video_path):
    faces, total_frames, detected_faces = load_faces_for_visual(video_path)

    feat_dict = extract_rppg_features_from_video(video_path)
    feat_vec = np.array([feat_dict[k] for k in FEATURES], dtype=np.float32)
    feat_vec = (feat_vec - rppg_mean) / (rppg_std + 1e-8)
    rppg_tensor = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(device)

    clips = sample_uniform_clips(faces, T=T, n_clips=N_CLIPS)

    probs = []
    visual_probs = []
    rppg_probs = []
    avg_gate = None

    with torch.no_grad():
        gate_vals = []
        for clip in clips:
            x = preprocess_clip(clip).to(device)
            logits, g, _, _, v_fake_logits, rppg_logits = fusion_model(x, rppg_tensor)
            
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
            v_prob_fake = torch.softmax(v_fake_logits, dim=1)[0, 1].item()
            r_prob_fake = torch.softmax(rppg_logits, dim=1)[0, 1].item()
            
            probs.append(prob_fake)
            visual_probs.append(v_prob_fake)
            rppg_probs.append(r_prob_fake)
            gate_vals.append(float(g.mean().item()))

    avg_visual_prob = float(np.mean(visual_probs))
    avg_rppg_prob = float(np.mean(rppg_probs))

    if gate_vals:
        avg_gate = float(np.mean(gate_vals))

    fake_prob = float(np.mean(probs))
    label = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if label == "FAKE" else (1.0 - fake_prob)

    vis_path = generate_visualization(video_path)

    return {
        "label": label,
        "prediction": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "fusion_method": "gated_visual_rppg_fusion",
        "frames_extracted": int(total_frames),
        "faces_detected": int(detected_faces),
        "face_crops_used": int(len(faces)),
        "clips_used": int(len(clips)),
        "clip_fake_probabilities": [round(float(p), 4) for p in probs],
        "average_fusion_gate": round(avg_gate, 4) if avg_gate is not None else None,
        "rppg_features": {
            k: round(float(v), 4) for k, v in feat_dict.items() if k != "total_frames_read"
        },
        "rppg_windows_success_ratio": round(float(feat_dict["window_success_ratio"]), 4),
        "visual_stream_score": round(avg_visual_prob, 4),
        "rppg_stream_score": round(avg_rppg_prob, 4),
        "visualization_path": vis_path
    }