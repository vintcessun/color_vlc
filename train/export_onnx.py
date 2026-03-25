import torch
import os
import sys
from ultralytics import YOLO

# 添加当前目录到路径以便导入模型定义
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_stage2_new import ColorQRStage2New, OUT_SIZE


def export_stage1():
    s1_pt = "train/stage1.pt"
    if not os.path.exists(s1_pt):
        print(f"Error: Stage 1 weights not found at {s1_pt}")
        return

    print(f"Exporting Stage 1: {s1_pt}")
    model = YOLO(s1_pt)
    # 导出为 ONNX，支持动态 Batch
    onnx_path = model.export(
        format="onnx",
        imgsz=960,
        opset=12,
        dynamic=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    print(f"Stage 1 ONNX exported to: {onnx_path}")


def export_stage2():
    s2_pt = "train/stage2.pt"
    if not os.path.exists(s2_pt):
        print(f"Error: Stage 2 weights not found at {s2_pt}")
        return

    print(f"Exporting Stage 2: {s2_pt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 严格按照 model_stage2_new.py
    model = ColorQRStage2New(pretrained_locnet=False, out_size=800).to(device)

    state_dict = torch.load(s2_pt, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # 严格按照 OUT_SIZE = 800
    dummy_input = torch.randn(1, 3, 800, 800).to(device)
    onnx_path = "train/stage2.onnx"

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pred_pts", "rectified"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "pred_pts": {0: "batch_size"},
            "rectified": {0: "batch_size"},
        },
    )
    print(f"Stage 2 ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    export_stage2()
    export_stage1()
