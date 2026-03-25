import argparse
import glob
import os
import json

import cv2
import numpy as np
from ultralytics import YOLO


CROP_PAD_PX = 20


def _find_stage1_model(model_path) -> str:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        return model_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(current_dir, "runs")
    run_names = [
        "color_qr_stage1_yolov8n_pose_960",
        "color_qr_stage1_yolov8s_pose_960",
        "color_qr_stage1_yolov8s_detect_960",
        "color_qr_stage1_rtdetr_l_detect_960",
        "color_qr_stage1_corner4_yolo26_pose_960",
        "color_qr_stage1_corner4_yolo11_pose",
        "color_qr_stage1_corner4",
    ]

    for run_name in run_names:
        best_cands = sorted(
            glob.glob(
                os.path.join(runs_dir, "**", run_name, "weights", "best.pt"),
                recursive=True,
            )
        )
        if best_cands:
            return best_cands[-1]

        last_cands = sorted(
            glob.glob(
                os.path.join(runs_dir, "**", run_name, "weights", "last.pt"),
                recursive=True,
            )
        )
        if last_cands:
            return last_cands[-1]

    raise RuntimeError(
        "stage1 model not found under train/runs. "
        "expected detect run such as color_qr_stage1_rtdetr_l_detect_960"
    )


def _draw_box(
    img: np.ndarray,
    xyxy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _square_box_xyxy(xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    nx1 = cx - side * 0.5
    ny1 = cy - side * 0.5
    nx2 = cx + side * 0.5
    ny2 = cy + side * 0.5

    # 平移回图内（保持正方形）
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0.0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0.0
    if nx2 > img_w:
        d = nx2 - img_w
        nx1 -= d
        nx2 = float(img_w)
    if ny2 > img_h:
        d = ny2 - img_h
        ny1 -= d
        ny2 = float(img_h)

    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(img_w), nx2)
    ny2 = min(float(img_h), ny2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def _detect_qr_box_classic(img: np.ndarray) -> np.ndarray:
    detector = cv2.QRCodeDetector()

    # 直接检测
    ok, points = detector.detect(img)
    if not ok or points is None:
        # 灰度增强后再试一次
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ok, points = detector.detect(gray)
    if not ok or points is None:
        return None

    pts = points.reshape(-1, 2).astype(np.float32)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _parse_int_list(text: str, fallback: int) -> list[int]:
    if text is None or text.strip() == "":
        return [fallback]
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals if vals else [fallback]


def _parse_float_list(text: str, fallback: float) -> list[float]:
    if text is None or text.strip() == "":
        return [fallback]
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals if vals else [fallback]


def _collect_candidates(
    model: YOLO,
    img: np.ndarray,
    imgsz_list: list[int],
    conf_list: list[float],
):
    h, w = img.shape[:2]
    img_area = float(max(1, h * w))
    candidates = []

    for cur_imgsz in imgsz_list:
        for cur_conf in conf_list:
            results = model.predict(img, imgsz=cur_imgsz, conf=cur_conf, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            res0 = results[0]
            n_det = len(res0.boxes)
            for det_idx in range(n_det):
                xyxy = res0.boxes.xyxy[det_idx].cpu().numpy().astype(np.float32)
                x1, y1, x2, y2 = xyxy.tolist()
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area = float(bw * bh)
                area_ratio = area / img_area

                box_conf = 0.0
                if res0.boxes is not None and len(res0.boxes) > det_idx:
                    box_conf = float(res0.boxes.conf[det_idx].item())

                # 兼顾检测置信度和几何面积：过小四边形通常不稳定
                score = box_conf + 0.25 * min(1.0, area_ratio * 4.0)

                candidates.append(
                    {
                        "imgsz": int(cur_imgsz),
                        "conf": float(cur_conf),
                        "det_idx": int(det_idx),
                        "box_conf": box_conf,
                        "area": area,
                        "area_ratio": area_ratio,
                        "score": score,
                        "xyxy": xyxy,
                    }
                )

    return candidates


def run_test(
    image_path: str,
    model_path,
    out_dir: str,
    imgsz: int,
    conf: float,
    rectify_size: int,
    imgsz_list: str,
    conf_list: str,
):
    model_path = _find_stage1_model(model_path)
    print(f"[Test-Stage1] use model: {model_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    model = YOLO(model_path)

    # 基础 conf/imgsz 优先，再尝试更低阈值和更高分辨率回退
    parsed_imgsz = _parse_int_list(imgsz_list, imgsz)
    parsed_conf = _parse_float_list(conf_list, conf)
    if imgsz not in parsed_imgsz:
        parsed_imgsz = [imgsz] + parsed_imgsz
    if conf not in parsed_conf:
        parsed_conf = [conf] + parsed_conf

    h, w = img.shape[:2]

    # 先尝试经典 QR 检测，失败后再回退 YOLO
    classic_box = _detect_qr_box_classic(img)
    used_method = "classic_qr_detector"
    if classic_box is not None:
        pred_box = _square_box_xyxy(classic_box, w, h)
        candidates = []
        best = {
            "imgsz": None,
            "conf": None,
            "det_idx": 0,
            "box_conf": 1.0,
            "area": float((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])),
            "area_ratio": float(
                ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]))
                / max(1, h * w)
            ),
            "score": 1.0,
        }
    else:
        used_method = "yolo_detect"
        candidates = _collect_candidates(model, img, parsed_imgsz, parsed_conf)
        if not candidates:
            raise RuntimeError(
                "classic QR detect and YOLO detect both failed. "
                "try better lighting / lower conf / larger imgsz-list"
            )
        best = max(candidates, key=lambda c: c["score"])
        pred_box = _square_box_xyxy(best["xyxy"], w, h)

    os.makedirs(out_dir, exist_ok=True)

    # 保存推理元信息，便于排查实拍失败原因
    debug_info = {
        "model": model_path,
        "method": used_method,
        "num_candidates": len(candidates),
        "selected": {
            "imgsz": best["imgsz"],
            "conf": best["conf"],
            "det_idx": best["det_idx"],
            "box_conf": best["box_conf"],
            "area": best["area"],
            "area_ratio": best["area_ratio"],
            "score": best["score"],
        },
    }
    with open(os.path.join(out_dir, "00_debug_info.json"), "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

    print(
        "[Test-Stage1] selected "
        f"imgsz={best['imgsz']} conf={best['conf']:.3f} "
        f"box_conf={best['box_conf']:.3f} area_ratio={best['area_ratio']:.4f}"
    )

    # 可视化原图预测框
    vis_src = img.copy()
    _draw_box(vis_src, pred_box, (0, 0, 255), thickness=3)
    cv2.imwrite(
        os.path.join(out_dir, "01_src_pred.jpg"),
        vis_src,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # 检测框裁剪：固定向框外扩 20 像素，给 Stage2 留白边上下文
    x1, y1, x2, y2 = pred_box.tolist()
    bw = x2 - x1
    bh = y2 - y1
    pad = float(CROP_PAD_PX)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh) + 2.0 * pad

    crop_x1 = max(0, int(round(cx - side * 0.5)))
    crop_y1 = max(0, int(round(cy - side * 0.5)))
    crop_x2 = min(w, int(round(cx + side * 0.5)))
    crop_y2 = min(h, int(round(cy + side * 0.5)))

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if cropped.size == 0:
        raise RuntimeError("detected crop is empty")

    rectified = cv2.resize(
        cropped, (rectify_size, rectify_size), interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(
        os.path.join(out_dir, "02_crop.jpg"),
        cropped,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(out_dir, "03_crop_resized.jpg"),
        rectified,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # 在最终图叠加边界框辅助观察
    vis_rect = rectified.copy()
    cv2.rectangle(
        vis_rect, (0, 0), (rectify_size - 1, rectify_size - 1), (255, 255, 0), 2
    )
    cv2.imwrite(
        os.path.join(out_dir, "04_crop_resized_with_border.jpg"),
        vis_rect,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    debug_info["selected"]["xyxy"] = [float(v) for v in pred_box.tolist()]
    debug_info["selected"]["crop_xyxy"] = [
        int(crop_x1),
        int(crop_y1),
        int(crop_x2),
        int(crop_y2),
    ]
    with open(os.path.join(out_dir, "00_debug_info.json"), "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

    print("[Test-Stage1] done")
    print(f"[Test-Stage1] outputs: {out_dir}")
    print(f"[Test-Stage1] crop_size={rectify_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test.png", help="input image path")
    parser.add_argument("--model", default=None, help="stage1 model path")
    parser.add_argument(
        "--out",
        default="train/inference_out/stage1_rectify_debug",
        help="output folder",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--imgsz-list",
        default="1280,1536,1920",
        help="fallback imgsz list, comma separated",
    )
    parser.add_argument(
        "--conf-list",
        default="0.25,0.10,0.05",
        help="fallback conf list, comma separated",
    )
    parser.add_argument("--rectify-size", type=int, default=716)
    args = parser.parse_args()

    run_test(
        image_path=args.image,
        model_path=args.model,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        rectify_size=args.rectify_size,
        imgsz_list=args.imgsz_list,
        conf_list=args.conf_list,
    )


if __name__ == "__main__":
    main()
