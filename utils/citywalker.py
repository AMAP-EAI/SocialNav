#!/usr/bin/env python

import os
import jsonlines
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time
from PIL import Image
from safetensors.torch import load_file

from transformers import AutoProcessor

MODEL_PATH = "/mnt/nas-data-3/jiexing.gyn/ckpts/checkpoint-28500"
DATA_PATH = "/mnt/nas-data-3/jiexing.gyn/Data/debug/1031_teleop_range1_interval1_new.jsonl"

DEVICE = "cuda:0"

OUTPUT_DIR = os.path.join(MODEL_PATH, "infer_result_citywalker_qwen3_fast_step_5")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PRED_JSONL_PATH = os.path.join(OUTPUT_DIR, "pred_citywalker_qwen3.jsonl")
METRIC_CSV_PATH = os.path.join(OUTPUT_DIR, "metrics_citywalker_qwen3.csv")

TEST_CATEGORIES = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']

def init_metric_dict():
    metrics = {}
    cats = TEST_CATEGORIES[:] + ["mean", "overall"]
    for c in cats:
        metrics[c] = {
            "l1_loss": [],
            "arrived_accuracy": [],
            "angle_step1": [],
            "angle_step2": [],
            "angle_step3": [],
            "angle_step4": [],
            "angle_step5": [],
            "mean_angle": [],
        }
    return metrics

def compute_sample_metrics(pred_wp_abs, gt_wp_abs, pred_arrive_logit, gt_arrive):
    pred_t = torch.from_numpy(pred_wp_abs).unsqueeze(0)
    gt_t = torch.from_numpy(gt_wp_abs).unsqueeze(0)
    l1_like = F.mse_loss(pred_t, gt_t, reduction="none").sqrt()
    max_l1_like = float(l1_like.view(-1).max().item())

    pred_prob = torch.sigmoid(torch.tensor(pred_arrive_logit))
    pred_label = 1.0 if float(pred_prob) >= 0.5 else 0.0
    arrived_correct = 1.0 if int(pred_label) == int(gt_arrive[0]) else 0.0

    pred_flat = pred_t.view(-1, 2)
    gt_flat = gt_t.view(-1, 2)
    cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(cos_sim) * 180.0 / torch.pi
    angles_np = angles.detach().cpu().numpy()
    max_angle_deg = float(angles.max().item())
    return max_l1_like, arrived_correct, angles_np, max_angle_deg

SPECIAL_TOKEN2ID = {
    '<input_pos1>': 151657,
    '<input_pos2>': 151658,
    '<input_pos3>': 151659,
    '<input_pos4>': 151660,
    '<input_pos5>': 151661,
    '<input_target>': 151662,
    '<flow_matching_policy>': 151664,
    '<time>': 151665,
}

class Qwen3VLModel(object):
    def __init__(self, model_path, device="cuda:0", flow_steps=5):
        self.device = torch.device(device)

        from transformers import Qwen3VLForConditionalGeneration

        dtype = torch.bfloat16
        additional_model_kwargs = {
            "action_dim": 2,
            "action_chunk": 5,
            "flow_matching_policy": True,
            "num_flow_steps": flow_steps,
            "action_former": True,
            "query_action_layer": 4,
            "sigma": 0.0,
            "sde_mode": "cps",
        }

        print(f">>> 加载 Qwen3VL 模型：{model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=str(device),
            trust_remote_code=True,
            **additional_model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.cuda()
        for name in os.listdir(model_path):
            if name.endswith('safetensors'):
                safe_model_path = os.path.join(model_path, name)
                state_dict = load_file(safe_model_path)
                self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def infer_one(self, item):
        messages = item["messages"]
        user_content = messages[0]["content"]
        images_paths = item["images"]

        images = [Image.open(p).convert("RGB") for p in images_paths]

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_content})
        messages_for_model = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages_for_model, tokenize=False, add_generation_prompt=True
        ) + "<|im_end|>"

        inputs = self.processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_waypoints = torch.tensor(
            messages[1]["input_waypoints"], dtype=torch.bfloat16
        ).unsqueeze(0).to(self.device)
        inputs["input_waypoints"] = input_waypoints

        outputs = self.model(
            **inputs,
            train=False,
            train_branch="fm",
            num_samples=1,
            special_token2id=SPECIAL_TOKEN2ID,
        )

        if isinstance(outputs, tuple):
            wp_pred = outputs[0]
            arrive_pred = outputs[1] if len(outputs) > 1 else None
        else:
            wp_pred = outputs
            arrive_pred = None

        if isinstance(wp_pred, torch.Tensor):
            wp_pred = wp_pred.squeeze(0).detach().cpu().float().numpy()
        else:
            wp_pred = np.array(wp_pred, dtype=np.float32)

        if isinstance(arrive_pred, torch.Tensor):
            arrive_logit = float(arrive_pred.squeeze().detach().cpu().float().item())
        elif isinstance(arrive_pred, (bool, int, float)):
            arrive_logit = 10.0 if bool(arrive_pred) else -10.0
        else:
            arrive_logit = -10.0

        return wp_pred, arrive_logit

def main():
    print("===> [1/4] 加载模型:", MODEL_PATH)
    model = Qwen3VLModel(MODEL_PATH, device=DEVICE, flow_steps=5)

    print("===> [2/4] 加载测试数据:", DATA_PATH)
    data_lines = []
    with jsonlines.open(DATA_PATH, "r") as reader:
        for obj in reader:
            data_lines.append(obj)
    total = len(data_lines)
    print(f"===> 一共 {total} 条样本")

    print("===> [3/4] 推理 + 指标统计（无可视化） ...")
    metrics = init_metric_dict()
    wf = jsonlines.open(PRED_JSONL_PATH, mode="w")

    success = 0
    filtered_count = 0
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(data_lines),
        total=total,
        desc="推理进度",
        dynamic_ncols=True,
        mininterval=0.3,
        smoothing=0.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for idx, item in progress_bar:
        try:
            t0 = time.time()
            wp_pred_rel, arrive_logit = model.infer_one(item)
            t1 = time.time()

            msg1 = item["messages"][1]
            gt_waypoints = np.asarray(msg1["gt_waypoints"], dtype=np.float32)[:5]
            wp_pred_rel = wp_pred_rel[:5]

            step_scale = float(msg1["step_scale"])
            gt_arrive = msg1.get("arrive", [0.0])
            raw_categories = msg1.get("categories", [0] * len(TEST_CATEGORIES))
            categories = [int(round(x)) for x in raw_categories[:len(TEST_CATEGORIES)]]
            if len(categories) < len(TEST_CATEGORIES):
                categories += [0] * (len(TEST_CATEGORIES) - len(categories))

            wp_pred_abs = wp_pred_rel * step_scale
            gt_wp_abs = gt_waypoints * step_scale

            l1_val, acc_val, angles, max_angle = compute_sample_metrics(
                wp_pred_abs, gt_wp_abs, arrive_logit, gt_arrive
            )

            if gt_wp_abs.shape[0] >= 2:
                path_distance_m = float(np.linalg.norm(gt_wp_abs[-1] - gt_wp_abs[0]))
            else:
                path_distance_m = float(np.linalg.norm(gt_wp_abs[-1]))

            too_close = (path_distance_m < 1.0)
            too_obtuse = (max_angle >= 90.0)
            should_record = not (too_close or too_obtuse)
            if not should_record:
                filtered_count += 1

            if should_record:
                m = metrics["overall"]
                m["l1_loss"].append(l1_val)
                m["arrived_accuracy"].append(acc_val)
                m["mean_angle"].append(max_angle)
                for i, k in enumerate(
                    ["angle_step1", "angle_step2", "angle_step3", "angle_step4", "angle_step5"]
                ):
                    m[k].append(float(angles[i]))

                for ci, cname in enumerate(TEST_CATEGORIES):
                    if categories[ci] == 1:
                        mc = metrics[cname]
                        mc["l1_loss"].append(l1_val)
                        mc["arrived_accuracy"].append(acc_val)
                        mc["mean_angle"].append(max_angle)
                        for i, k in enumerate(
                            ["angle_step1", "angle_step2", "angle_step3", "angle_step4", "angle_step5"]
                        ):
                            mc[k].append(float(angles[i]))

            wf.write({
                "item": item,
                "pred": {
                    "wp_pred": wp_pred_rel.tolist(),
                    "arrive_pred_logit": arrive_logit,
                },
                "metrics_per_sample": {
                    "l1_like": l1_val,
                    "arrived_correct": acc_val,
                    "max_angle": max_angle,
                    "angles": angles.tolist(),
                    "path_distance_m": path_distance_m,
                    "filtered": (not should_record),
                    "inference_time": round(t1 - t0, 3),
                },
            })
            wf._fp.flush()

            success += 1
            progress_bar.set_postfix({
                "dist": f"{path_distance_m:.2f}m",
                "angle": f"{max_angle:.2f}°",
                "arr": int(acc_val),
                "t(s)": f"{t1 - t0:.2f}",
            })

        except Exception as e:
            print(f"[WARN] 第 {idx} 条推理失败: {e}")
            continue

    wf.close()
    total_time = time.time() - start_time

    print(f"\n===> [4/4] 完成。总用时 {total_time/60:.1f} 分钟，共处理 {success}/{total} 条样本。")
    print(f"有效样本(参与指标 mean_angle 等)：{len(metrics['overall']['mean_angle'])}，被过滤：{filtered_count}")

    for cname in TEST_CATEGORIES:
        metrics[cname]["count"] = len(metrics[cname]["l1_loss"])
    metrics["overall"]["count"] = len(metrics["overall"]["l1_loss"])
    metrics["mean"]["count"] = 0

    for cname in TEST_CATEGORIES:
        for k, v in metrics[cname].items():
            if k == "count":
                continue
            arr = np.asarray(v, dtype=np.float32)
            metrics[cname][k] = float(np.nanmean(arr)) if arr.size > 0 else float("nan")

    for k, v in metrics["overall"].items():
        if k == "count":
            continue
        arr = np.asarray(v, dtype=np.float32)
        metrics["overall"][k] = float(np.nanmean(arr)) if arr.size > 0 else float("nan")

    metric_names = [
        "l1_loss", "arrived_accuracy",
        "angle_step1", "angle_step2", "angle_step3", "angle_step4", "angle_step5",
        "mean_angle",
    ]
    for mk in metric_names:
        vals = [metrics[c][mk] for c in TEST_CATEGORIES]
        metrics["mean"][mk] = (
            float(np.nanmean(np.asarray(vals, dtype=np.float32))) if len(vals) > 0 else float("nan")
        )

    df = pd.DataFrame(metrics).reset_index().rename(columns={"index": "Metrics"})
    df.to_csv(METRIC_CSV_PATH, index=False)

    print(f"指标CSV: {METRIC_CSV_PATH}")
    print(f"逐条结果: {PRED_JSONL_PATH}")

if __name__ == "__main__":
    main()
