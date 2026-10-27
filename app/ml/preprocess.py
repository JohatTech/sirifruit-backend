"""OpenCV preprocessing: decode, white balance, crop the pack, resize, color histogram."""
import cv2
import numpy as np

INPUT_SIZE = 224
HIST_BINS = (16, 4)  # Hue x Saturation -> 64-dim histogram
HIST_DIM = HIST_BINS[0] * HIST_BINS[1]


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a BGR image. IMREAD_COLOR applies EXIF orientation."""
    if not data:
        raise InvalidImageError("Imagen vacía")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("No se pudo leer la imagen")
    return img


def limit_size(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def gray_world_balance(img: np.ndarray) -> np.ndarray:
    """Normalize color cast from store lighting (gray-world assumption)."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    gains = gray / np.maximum(means, 1e-6)
    gains = np.clip(gains, 0.6, 1.6)  # avoid over-correcting strongly colored packs
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def _square_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int, pad: float = 0.08):
    side = int(max(w, h) * (1 + pad))
    cx, cy = x + w // 2, y + h // 2
    side = min(side, img_w, img_h)
    x0 = int(np.clip(cx - side // 2, 0, img_w - side))
    y0 = int(np.clip(cy - side // 2, 0, img_h - side))
    return x0, y0, side


def crop_main_object(img: np.ndarray) -> np.ndarray:
    """Crop to the largest object near the center; fall back to a centered square."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    img_area = float(h * w)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.10 * img_area or area > 0.95 * img_area:
            continue
        # Prefer large boxes close to the center (where the framing guide is).
        cx, cy = x + cw / 2, y + ch / 2
        dist = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        score = (area / img_area) * (1.0 - dist)
        if score > best_score:
            best, best_score = (x, y, cw, ch), score

    if best is None:
        side = int(min(h, w) * 0.85)
        x0, y0 = (w - side) // 2, (h - side) // 2
    else:
        x0, y0, side = _square_box(*best, w, h)
    return img[y0:y0 + side, x0:x0 + side]


def hsv_histogram(img: np.ndarray) -> np.ndarray:
    """L1-normalized Hue/Saturation histogram of colorful pixels (ignores white/black/gray)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    if cv2.countNonZero(mask) < 0.02 * mask.size:
        mask = None  # pack has almost no saturated color (e.g. coco): use all pixels
    hist = cv2.calcHist([hsv], [0, 1], mask, list(HIST_BINS), [0, 180, 0, 256]).flatten()
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


def preprocess(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (RGB 224x224 uint8 image for the CNN, HSV histogram)."""
    img = limit_size(decode_image(data))
    img = gray_world_balance(img)
    crop = crop_main_object(img)
    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    hist = hsv_histogram(crop)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb, hist
