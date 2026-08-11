#!/usr/bin/env python

import argparse
import os
import jsonlines
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time
from PIL import Image

from transformers import AutoConfig, AutoProcessor

TEST_CATEGORIES = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']

# The waypoint placeholders are contiguous in the tokenizer: <input_pos1>..<input_pos5>
# followed by <input_target>. Their absolute ids differ per backbone, so they are
# always resolved from the tokenizer rather than hard-coded.
WAYPOINT_TOKENS = [f"<input_pos{i}>" for i in range(1, 6)] + ["<input_target>"]
EXTRA_SPECIAL_TOKENS = ["<flow_matching_policy>", "<time>"]


def parse_args():
    p = argparse.ArgumentParser(
        description="CityWalker benchmark evaluation for SocialNav (Qwen2-VL / Qwen2.5-VL / Qwen3-VL)."
    )
    p.add_argument("--model-path", required=True, help="Path to a SocialNav checkpoint directory.")
    p.add_argument("--data-path", required=True, help="Benchmark .jsonl file.")
    p.add_argument("--output-dir", default=None, help="Defaults to <model-path>/infer_result_citywalker.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flow-steps", type=int, default=5, help="Euler steps used by the action expert.")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    return p.parse_args()


def init_metric_dict(action_chunk):
    metrics = {}
    cats = TEST_CATEGORIES[:] + ["mean", "overall"]
    for c in cats:
        metrics[c] = {"l1_loss": [], "arrived_accuracy": []}
        for i in range(1, action_chunk + 1):
            metrics[c][f"angle_step{i}"] = []
        metrics[c]["mean_angle"] = []
    return metrics

def compute_sample_metrics(pred_wp_abs, gt_wp_abs, pred_arrive_logit, gt_arrive):
    pred_t = torch.from_numpy(pred_wp_abs).unsqueeze(0)
    gt_t = torch.from_numpy(gt_wp_abs).unsqueeze(0)
    l1_like = F.mse_loss(pred_t, gt_t, reduction="none").sqrt()
    max_l1_like = float(l1_like.view(-1).max().item())

    # The released checkpoints have no arrival head, so `pred_arrive_logit` is None
    # and the accuracy is reported as NaN rather than as a meaningless constant.
    if pred_arrive_logit is None:
        arrived_correct = float("nan")
    else:
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


# model_type -> (model class name, monkey patch helpers)
BACKBONES = {
    "qwen2_vl": ("Qwen2VLForConditionalGeneration", ["replace_qwen_2_with_mixed_modality_forward"]),
    "qwen2_5_vl": (
        "Qwen2_5_VLForConditionalGeneration",
        ["replace_qwen2_5_with_mixed_modality_forward", "replace_qwen2_5_vision"],
    ),
    "qwen3_vl": ("Qwen3VLForConditionalGeneration", ["replace_qwen3_with_mixed_modality_forward"]),
    "qwen3_vl_moe": ("Qwen3VLMoeForConditionalGeneration", ["replace_qwen3_vl_moe_with_mixed_modality_forward"]),
}


class SocialNavModel(object):
    """Loads a SocialNav checkpoint on any of the supported Qwen-VL backbones."""

    def __init__(self, model_path, device="cuda:0", flow_steps=5):
        import transformers

        from src.train import monkey_patch_forward, monkey_patch_vision

        self.device = torch.device(device)

        config = AutoConfig.from_pretrained(model_path)
        self.model_type = config.model_type
        if self.model_type not in BACKBONES:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. Expected one of: {', '.join(BACKBONES)}."
            )
        class_name, patches = BACKBONES[self.model_type]

        # Apply the same mixed-modality patches the training entrypoints use, so
        # that evaluation and training run the exact same backbone code path.
        for patch in patches:
            fn = getattr(monkey_patch_forward, patch, None) or getattr(monkey_patch_vision, patch)
            fn()

        additional_model_kwargs = {
            "action_dim": 2,
            "action_chunk": 5,
            "flow_matching_policy": True,
            "num_flow_steps": flow_steps,
            "action_former": True,
            "query_action_layer": 4,
            "sigma": 0.0,
            "ar_lambda_loss": 1.0,
            "min_value": -1.25,
            "max_value": 1.25,
            "sde_mode": "cps",
        }

        print(f">>> loading {class_name} from {model_path}")
        model_cls = getattr(transformers, class_name)
        # `from_pretrained` already remaps the checkpoint layout and loads the
        # action expert, so no manual state_dict reload is needed.
        self.model = model_cls.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=str(device),
            **additional_model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.special_token2id = self._resolve_special_tokens()
        self.action_chunk = self.model.action_chunk
        print(f">>> backbone={self.model_type} action_chunk={self.action_chunk} flow_steps={flow_steps}")
        print(f">>> <input_pos1> resolved to id {self.special_token2id['<input_pos1>']}")

    def _resolve_special_tokens(self):
        """Resolve the SocialNav placeholder ids from the checkpoint's tokenizer."""
        tokenizer = self.processor.tokenizer
        token2id = {}
        for token in WAYPOINT_TOKENS + EXTRA_SPECIAL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Token {token} is missing from the tokenizer of this checkpoint; "
                    "it is required to inject the history waypoints."
                )
            token2id[token] = token_id

        ids = [token2id[t] for t in WAYPOINT_TOKENS]
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError(f"Waypoint placeholder ids must be contiguous, got {ids}.")
        return token2id

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
            messages[1]["input_waypoints"], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        inputs["input_waypoints"] = input_waypoints

        outputs = self.model(
            **inputs,
            train=False,
            train_branch="fm",
            num_samples=1,
            special_token2id=self.special_token2id,
        )

        wp_pred, arrive_pred = outputs[0], outputs[1] if len(outputs) > 1 else None
        wp_pred = wp_pred.squeeze(0).detach().cpu().float().numpy()

        # `arrive` is a placeholder in the released checkpoints (no arrival head):
        # keep it as None so the metric is reported as NaN instead of a constant.
        arrive_logit = None
        if isinstance(arrive_pred, torch.Tensor) and arrive_pred.abs().sum() > 0:
            arrive_logit = float(arrive_pred.squeeze().detach().cpu().float().item())

        return wp_pred, arrive_logit

def main():
    args = parse_args()
    model_path = args.model_path
    output_dir = args.output_dir or os.path.join(model_path, "infer_result_citywalker")
    os.makedirs(output_dir, exist_ok=True)

    print("===> [1/4] loading model:", model_path)
    model = SocialNavModel(model_path, device=args.device, flow_steps=args.flow_steps)
    action_chunk = model.action_chunk
    angle_keys = [f"angle_step{i}" for i in range(1, action_chunk + 1)]

    pred_jsonl_path = os.path.join(output_dir, f"pred_citywalker_{model.model_type}.jsonl")
    metric_csv_path = os.path.join(output_dir, f"metrics_citywalker_{model.model_type}.csv")

    print("===> [2/4] loading benchmark data:", args.data_path)
    data_lines = []
    with jsonlines.open(args.data_path, "r") as reader:
        for obj in reader:
            data_lines.append(obj)
    if args.limit is not None:
        data_lines = data_lines[: args.limit]
    total = len(data_lines)
    print(f"===> {total} samples")

    print("===> [3/4] inference + metrics ...")
    metrics = init_metric_dict(action_chunk)
    wf = jsonlines.open(pred_jsonl_path, mode="w")

    success = 0
    filtered_count = 0
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(data_lines),
        total=total,
        desc="inference",
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
            gt_waypoints = np.asarray(msg1["gt_waypoints"], dtype=np.float32)[:action_chunk]
            wp_pred_rel = wp_pred_rel[:action_chunk]

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
                for i, k in enumerate(angle_keys):
                    m[k].append(float(angles[i]))

                for ci, cname in enumerate(TEST_CATEGORIES):
                    if categories[ci] == 1:
                        mc = metrics[cname]
                        mc["l1_loss"].append(l1_val)
                        mc["arrived_accuracy"].append(acc_val)
                        mc["mean_angle"].append(max_angle)
                        for i, k in enumerate(angle_keys):
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
                "t(s)": f"{t1 - t0:.2f}",
            })

        except Exception as e:
            print(f"[WARN] sample {idx} failed: {e}")
            continue

    wf.close()
    total_time = time.time() - start_time

    print(f"\n===> [4/4] done in {total_time/60:.1f} min, {success}/{total} samples processed.")
    print(f"kept for metrics: {len(metrics['overall']['mean_angle'])}, filtered out: {filtered_count}")

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

    metric_names = ["l1_loss", "arrived_accuracy"] + angle_keys + ["mean_angle"]
    for mk in metric_names:
        vals = [metrics[c][mk] for c in TEST_CATEGORIES]
        metrics["mean"][mk] = (
            float(np.nanmean(np.asarray(vals, dtype=np.float32))) if len(vals) > 0 else float("nan")
        )

    df = pd.DataFrame(metrics).reset_index().rename(columns={"index": "Metrics"})
    df.to_csv(metric_csv_path, index=False)

    print(f"metrics CSV: {metric_csv_path}")
    print(f"per-sample results: {pred_jsonl_path}")
    if np.isnan(metrics["overall"]["arrived_accuracy"]):
        print(
            "note: `arrived_accuracy` is NaN because the released checkpoints have no "
            "arrival head; the action expert only predicts the trajectory."
        )

if __name__ == "__main__":
    main()
