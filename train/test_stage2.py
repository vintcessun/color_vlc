import argparse
import glob
import json
import os
from typing import Optional

import cv2
import numpy as np
import torch

from model_stage2 import CANON_KPTS_NORM, ColorQRStage2, OUT_SIZE


IDX_TL, IDX_TR, IDX_BR, IDX_BL = 49, 50, 51, 52


def _find_stage2_model(model_path: Optional[str]) -> str:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        return model_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(current_dir, "runs")
    run_names = ["stage2_stn"]

    for run_name in run_names:
        best_cands = sorted(
            glob.glob(
                os.path.join(runs_dir, "**", run_name, "best.pt"),
                recursive=True,
            )
        )
        if best_cands:
            return best_cands[-1]

        final_cands = sorted(
            glob.glob(
                os.path.join(runs_dir, "**", run_name, "stage2_stn_final.pt"),
                recursive=True,
            )
        )
        if final_cands:
            return final_cands[-1]

        ckpt_cands = sorted(
            glob.glob(
                os.path.join(runs_dir, "**", run_name, "checkpoints", "epoch_*.pt"),
                recursive=True,
            )
        )
        if ckpt_cands:
            return ckpt_cands[-1]

    raise RuntimeError("stage2 model not found under train/runs/stage2_stn")


def _resolve_image_path(image_path: str) -> str:
    # 1) 用户提供路径直接可用
    if os.path.exists(image_path):
        return image_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 2) 常见候选路径
    candidates = [
        os.path.join(project_root, image_path),
        os.path.join(current_dir, image_path),
    ]

    # 3) test_wraped.jpg 常见拼写兜底：test_wraped.jpg
    if os.path.basename(image_path).lower() == "test_wraped.jpg":
        candidates.extend(
            [
                os.path.join(project_root, "test_wraped.jpg"),
                os.path.join(current_dir, "test_wraped.jpg"),
            ]
        )

    for c in candidates:
        if os.path.exists(c):
            return c

    raise RuntimeError(
        f"failed to locate image: {image_path}. "
        f"also tried fallback 'test_wraped.jpg' in project root/train"
    )


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(
        np.uint8
    )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _heatmap_to_color(hm: np.ndarray, out_size: int) -> np.ndarray:
    """将单通道 heatmap 可视化为伪彩色图。"""
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    den = float(hm.max()) + 1e-6
    hm = hm / den
    hm_u8 = (hm * 255.0).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def _norm_to_px(kpts_norm: np.ndarray, out_size: int) -> np.ndarray:
    half = (out_size - 1) / 2.0
    return (kpts_norm + 1.0) * half


def _letterbox_to_square(img_bgr: np.ndarray, out_size: int):
    src_h, src_w = img_bgr.shape[:2]
    scale = min(out_size / float(src_w), out_size / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    pad_x = (out_size - new_w) // 2
    pad_y = (out_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    meta = {
        "scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
    }
    return canvas, meta


def _square_px_to_orig(
    pts_sq: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    out = pts_sq.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - float(pad_x)) / float(scale)
    out[:, 1] = (out[:, 1] - float(pad_y)) / float(scale)
    out[:, 0] = np.clip(out[:, 0], 0.0, float(src_w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(src_h - 1))
    return out


def _draw_kpts(
    bgr: np.ndarray,
    kpts_norm: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
):
    h, w = bgr.shape[:2]
    half = (w - 1) / 2.0
    out = bgr.copy()
    for xn, yn in kpts_norm:
        x = int(round((float(xn) + 1.0) * half))
        y = int(round((float(yn) + 1.0) * half))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, -1)
    return out


def _draw_kpts_px(
    bgr: np.ndarray,
    kpts_px: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
):
    h, w = bgr.shape[:2]
    out = bgr.copy()
    for x, y in kpts_px:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(out, (xi, yi), radius, color, -1)
    return out


def _load_stage2_weights(model: ColorQRStage2, weight_path: str, device: torch.device):
    state = torch.load(weight_path, map_location=device)
    try:
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=True)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=True)
        else:
            raise RuntimeError(f"unsupported checkpoint format: {type(state)}")
    except RuntimeError as e:
        raise RuntimeError(
            "checkpoint incompatible with current Stage2 architecture. "
            "Please retrain Stage2 after refactoring LocalizationNet (deconv + heatmap + soft-argmax), "
            f"then test again. details: {e}"
        )


def run_test(
    image_path: str,
    model_path: Optional[str],
    out_dir: str,
    out_size: int,
    device_str: str,
):
    model_path = _find_stage2_model(model_path)
    image_path = _resolve_image_path(image_path)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[Test-Stage2] use model: {model_path}")
    print(f"[Test-Stage2] device: {device}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    src_h, src_w = img.shape[:2]
    letterbox, meta = _letterbox_to_square(img, out_size)

    rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)
    inp = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    inp = inp.to(device)

    model = ColorQRStage2(pretrained_locnet=False, out_size=out_size).to(device)
    _load_stage2_weights(model, model_path, device)
    model.eval()

    with torch.no_grad():
        pred_kpts, rectified, heatmaps = model(inp, return_heatmaps=True)

    pred_kpts_np = pred_kpts[0].detach().cpu().numpy()
    heatmaps_np = heatmaps[0].detach().cpu().numpy()  # (53, h, w)
    rect_bgr = _tensor_to_bgr(rectified[0])

    pred_px_sq = _norm_to_px(pred_kpts_np, out_size)
    pred_px_orig = _square_px_to_orig(
        pred_px_sq,
        scale=meta["scale"],
        pad_x=meta["pad_x"],
        pad_y=meta["pad_y"],
        src_w=src_w,
        src_h=src_h,
    )

    canon_px_sq = _norm_to_px(CANON_KPTS_NORM, out_size)
    H_can2pred, _ = cv2.findHomography(
        canon_px_sq.astype(np.float32), pred_px_sq.astype(np.float32), method=0
    )
    if H_can2pred is None:
        raise RuntimeError(
            "failed to estimate homography from canon points to predictions"
        )
    mapped_canon_sq = cv2.perspectiveTransform(
        canon_px_sq.reshape(-1, 1, 2).astype(np.float32), H_can2pred
    ).reshape(-1, 2)
    mapped_canon_orig = _square_px_to_orig(
        mapped_canon_sq,
        scale=meta["scale"],
        pad_x=meta["pad_x"],
        pad_y=meta["pad_y"],
        src_w=src_w,
        src_h=src_h,
    )

    os.makedirs(out_dir, exist_ok=True)

    vis_src = _draw_kpts(letterbox, pred_kpts_np, color=(0, 0, 255), radius=3)
    vis_rect = _draw_kpts(rect_bgr, CANON_KPTS_NORM, color=(255, 128, 0), radius=3)
    vis_orig_pred = _draw_kpts_px(img, pred_px_orig, color=(0, 0, 255), radius=3)
    vis_orig_canon = _draw_kpts_px(
        vis_orig_pred, mapped_canon_orig, color=(255, 128, 0), radius=2
    )

    corners_orig = pred_px_orig[[IDX_TL, IDX_TR, IDX_BR, IDX_BL]].astype(np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [out_size - 1.0, 0.0],
            [out_size - 1.0, out_size - 1.0],
            [0.0, out_size - 1.0],
        ],
        dtype=np.float32,
    )
    H_orig2canon = cv2.getPerspectiveTransform(corners_orig, dst)
    rect_from_orig = cv2.warpPerspective(
        img,
        H_orig2canon,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    cv2.imwrite(
        os.path.join(out_dir, "01_input_letterbox_with_pred_kpts.jpg"),
        vis_src,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "02_rectified_tps.jpg"),
        rect_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "03_rectified_with_canon_kpts.jpg"),
        vis_rect,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "04_side_by_side_letterbox_vs_rectified.jpg"),
        np.hstack([vis_src, vis_rect]),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "05_original_with_pred_and_canon_kpts.jpg"),
        vis_orig_canon,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "06_original_slice_perspective_rectified.jpg"),
        rect_from_orig,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # Heatmap 可视化：均值热力图 + 四角点热力图拼图
    hm_mean_color = _heatmap_to_color(heatmaps_np.mean(axis=0), out_size)
    hm_overlay = cv2.addWeighted(letterbox, 0.55, hm_mean_color, 0.45, 0.0)
    cv2.imwrite(
        os.path.join(out_dir, "07_heatmap_mean_overlay.jpg"),
        hm_overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    hm_tl = _heatmap_to_color(heatmaps_np[IDX_TL], out_size)
    hm_tr = _heatmap_to_color(heatmaps_np[IDX_TR], out_size)
    hm_br = _heatmap_to_color(heatmaps_np[IDX_BR], out_size)
    hm_bl = _heatmap_to_color(heatmaps_np[IDX_BL], out_size)
    hm_grid = np.hstack([np.vstack([hm_tl, hm_tr]), np.vstack([hm_bl, hm_br])])
    cv2.imwrite(
        os.path.join(out_dir, "08_heatmap_corners_tl_tr_bl_br.jpg"),
        hm_grid,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    x_min_sq, y_min_sq = pred_px_sq.min(axis=0)
    x_max_sq, y_max_sq = pred_px_sq.max(axis=0)
    x_min_orig, y_min_orig = pred_px_orig.min(axis=0)
    x_max_orig, y_max_orig = pred_px_orig.max(axis=0)

    info = {
        "image_path": image_path,
        "input_shape": [int(src_h), int(src_w)],
        "model_path": model_path,
        "device": str(device),
        "out_size": int(out_size),
        "preprocess": {
            "mode": "letterbox",
            "scale": float(meta["scale"]),
            "new_w": int(meta["new_w"]),
            "new_h": int(meta["new_h"]),
            "pad_x": int(meta["pad_x"]),
            "pad_y": int(meta["pad_y"]),
        },
        "pred_kpts_norm_range": {
            "min": float(pred_kpts_np.min()),
            "max": float(pred_kpts_np.max()),
        },
        "heatmaps_shape": [
            int(heatmaps_np.shape[0]),
            int(heatmaps_np.shape[1]),
            int(heatmaps_np.shape[2]),
        ],
        "pred_kpts_bbox_square_px": {
            "x_min": float(x_min_sq),
            "y_min": float(y_min_sq),
            "x_max": float(x_max_sq),
            "y_max": float(y_max_sq),
        },
        "pred_kpts_bbox_orig_px": {
            "x_min": float(x_min_orig),
            "y_min": float(y_min_orig),
            "x_max": float(x_max_orig),
            "y_max": float(y_max_orig),
        },
        "pred_corners_orig_px": {
            "tl": [float(corners_orig[0, 0]), float(corners_orig[0, 1])],
            "tr": [float(corners_orig[1, 0]), float(corners_orig[1, 1])],
            "br": [float(corners_orig[2, 0]), float(corners_orig[2, 1])],
            "bl": [float(corners_orig[3, 0]), float(corners_orig[3, 1])],
        },
        "homography_canon_to_pred_square": H_can2pred.tolist(),
    }

    with open(os.path.join(out_dir, "00_debug_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("[Test-Stage2] done")
    print(f"[Test-Stage2] outputs: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test_wraped.jpg", help="input image path")
    parser.add_argument("--model", default=None, help="stage2 model path")
    parser.add_argument(
        "--out",
        default="train/inference_out/stage2_rectify_debug",
        help="output folder",
    )
    parser.add_argument("--out-size", type=int, default=OUT_SIZE)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    args = parser.parse_args()

    run_test(
        image_path=args.image,
        model_path=args.model,
        out_dir=args.out,
        out_size=args.out_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
