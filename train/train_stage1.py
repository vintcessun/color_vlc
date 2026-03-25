from ultralytics import YOLO
import os
import glob
import shutil
import yaml
import torch
from pathlib import Path

STAGE1_IMGSZ = 960
STAGE1_RUN_NAME = "color_qr_stage1_yolov8n_pose_960"
STAGE1_BASE_MODEL = "yolov8n-pose.pt"
STAGE1_EPOCHS = 800
STAGE1_ENABLE_PLOTS = True


def _force_val_plots_every_epoch(validator):
    validator.args.plots = True


def _save_epoch_val_visuals(trainer):
    save_dir = Path(trainer.save_dir)
    vis_dir = save_dir / "val_vis_per_epoch"
    vis_dir.mkdir(parents=True, exist_ok=True)

    epoch = int(trainer.epoch) + 1
    pred_candidates = sorted(save_dir.glob("val_batch*_pred.jpg"))
    if pred_candidates:
        src = pred_candidates[0]
        dst = vis_dir / f"epoch_{epoch:04d}_pred.jpg"
        shutil.copy2(src, dst)


def _load_prev_train_args(runs_dir: str) -> dict:
    """读取上一次同名 run 的 args.yaml，用于继承训练配置。"""
    arg_files = sorted(
        glob.glob(
            os.path.join(runs_dir, "**", STAGE1_RUN_NAME, "args.yaml"),
            recursive=True,
        )
    )
    if not arg_files:
        return {}

    try:
        with open(arg_files[-1], "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_ckpt_epoch(ckpt_path: str) -> int:
    """读取 checkpoint 已训练的最后 epoch（0-based），读取失败返回 -1。"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ep = ckpt.get("epoch", -1)
        return int(ep)
    except Exception:
        return -1


def train_stage1():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "stage1_dataset")
    runs_dir = os.path.join(current_dir, "runs")
    data_yaml_path = os.path.join(current_dir, "data_stage1.yaml")

    # 2) 写 data yaml
    data_yaml = f"""
path: {dataset_path}
train: images
val: images

kpt_shape: [4, 3]

names:
  0: color_qr
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml)

    # 3) 载入权重（可续训）
    last_candidates = sorted(
        glob.glob(
            os.path.join(runs_dir, "**", STAGE1_RUN_NAME, "weights", "last.pt"),
            recursive=True,
        )
    )
    best_candidates = sorted(
        glob.glob(
            os.path.join(runs_dir, "**", STAGE1_RUN_NAME, "weights", "best.pt"),
            recursive=True,
        )
    )
    last_pt = last_candidates[-1] if last_candidates else None
    best_pt = best_candidates[-1] if best_candidates else None

    # 自动检查并 resume 上一次中断的 epoch
    should_resume = False
    resume_last_epoch = -1
    if last_pt:
        resume_last_epoch = _read_ckpt_epoch(last_pt)
        # 仅当未达到目标 epochs 时才 resume；否则走 finetune（resume=False）
        should_resume = (resume_last_epoch + 1) < STAGE1_EPOCHS
        if should_resume:
            print(
                f"[Stage1] Found unfinished last checkpoint, RESUMING: {last_pt} "
                f"(last_epoch={resume_last_epoch + 1}, target={STAGE1_EPOCHS})"
            )
        else:
            print(
                f"[Stage1] Last checkpoint already reached target (or above), "
                f"continue from weights without resume: {last_pt} "
                f"(last_epoch={resume_last_epoch + 1}, target={STAGE1_EPOCHS})"
            )

    if should_resume:
        # 加载 last.pt
        model = YOLO(last_pt)
    elif last_pt:
        # 不 resume，只从 last 权重继续训练（允许修改 epochs）
        model = YOLO(last_pt)
    elif best_pt:
        print(f"[Stage1] Last checkpoint missing but found best, finetuning: {best_pt}")
        model = YOLO(best_pt)
    else:
        print(f"[Stage1] No checkpoint found, starting FRESH: {STAGE1_BASE_MODEL}")
        # Stage1 只做粗检测+裁剪，优先使用经典稳定的 YOLOv8 detect。
        model = YOLO(STAGE1_BASE_MODEL)

    if STAGE1_ENABLE_PLOTS:
        model.add_callback("on_val_start", _force_val_plots_every_epoch)
        model.add_callback("on_fit_epoch_end", _save_epoch_val_visuals)

    print(f"[Stage1] project dir: {runs_dir}")

    # 默认配置
    train_kwargs = dict(
        data=data_yaml_path,
        epochs=STAGE1_EPOCHS,
        imgsz=STAGE1_IMGSZ,
        batch=8,  # 提高到 16 优化训练效率
        project=runs_dir,
        name=STAGE1_RUN_NAME,
        exist_ok=True,
        workers=0,  # workers 至少为 8
        resume=should_resume,
        pose=20.0,  # 调高 pose 权重
        rect=False,  # 既然要全向旋转，不建议开启 rect 训练
        plots=STAGE1_ENABLE_PLOTS,
        augment=True,  # 开启增强
        overlap_mask=False,  # 关心 Keypoints，不关心分割
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.1,
        save_period=1,
    )

    # 继承上一轮训练配置（仅覆盖允许字段），最后强制 epochs/resume 生效
    prev_args = _load_prev_train_args(runs_dir)
    inheritable_keys = [
        "imgsz",
        "batch",
        "workers",
        "rect",
        "pose",
        "augment",
        "overlap_mask",
        "fliplr",
        "flipud",
        "mosaic",
        "save_period",
    ]
    for k in inheritable_keys:
        if k in prev_args and prev_args[k] is not None:
            train_kwargs[k] = prev_args[k]

    # 用户需求：只改变轮数；其余训练配置保持继承/原值
    train_kwargs["epochs"] = STAGE1_EPOCHS
    train_kwargs["resume"] = should_resume
    train_kwargs["plots"] = STAGE1_ENABLE_PLOTS

    # 4) 训练
    try:
        model.train(**train_kwargs)
    except AssertionError as e:
        msg = str(e)
        if train_kwargs.get("resume", False) and "nothing to resume" in msg:
            print(
                "[Stage1] Resume checkpoint already marked as finished by Ultralytics; "
                "fallback to resume=False and continue from last weights."
            )
            train_kwargs["resume"] = False

            # 重新构建 model，避免复用失败 trainer 的内部状态
            if last_pt:
                model = YOLO(last_pt)
            elif best_pt:
                model = YOLO(best_pt)
            else:
                model = YOLO(STAGE1_BASE_MODEL)

            if STAGE1_ENABLE_PLOTS:
                model.add_callback("on_val_start", _force_val_plots_every_epoch)
                model.add_callback("on_fit_epoch_end", _save_epoch_val_visuals)
            model.train(**train_kwargs)
        else:
            raise

    # 5) 导出
    onnx_ok = model.export(format="onnx")
    print(f"[Stage1] export onnx: {onnx_ok}")


if __name__ == "__main__":
    train_stage1()
