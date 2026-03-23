import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai


# gyn_add_start

import json
import jsonlines
from typing import Union, List, Any
import numpy as np

def load_json_data(data_path: Union[str, List, Any]):
    """
    加载 JSON 或 JSONL 文件，或直接接受数据列表，支持多个路径(list)的输入：
    - str: 路径，优先尝试按 JSON 读，失败则按 JSONL 读
    - list: 其中每一项可以是路径或已经加载的数据，最终展平成一个 list
    - 其它类型: 原样返回
    """
    # 1. 如果是字符串，当做单个文件路径
    if isinstance(data_path, str):
        data_path = data_path.strip()
        # 先尝试 JSON
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # 再尝试 JSONL
            data = []
            with jsonlines.open(data_path, 'r') as reader:
                for obj in reader:
                    data.append(obj)
            return data

    # 2. 如果是 list，递归处理每一个分项，并展平结果
    elif isinstance(data_path, list):
        all_data = []
        for path in data_path:
            result = load_json_data(path)
            if isinstance(result, list):
                all_data.extend(result)
            else:
                all_data.append(result)
        return all_data

    # 3. 其它情况（一般直接给数据了）
    else:
        return data_path

class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(GRPODataset, self).__init__()
        # if isinstance(data_path, str):
        #     list_data_dict = json.load(open(data_path, "r"))
        # else:
        #     list_data_dict = data_path

        list_data_dict = load_json_data(data_path)  # gyn

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

        self.processor.image_processor.do_resize = False

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        if "image" in sources:
            videos = None
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                # image_input = get_image_info(
                #         image_file, 
                #         self.image_min_pixel, 
                #         self.image_max_pixel, 
                #         self.image_resized_w, 
                #         self.image_resized_h, 
                #         self.image_patch_size
                #     )
                # images.append(image_input)
                # gyn
                try:
                    image_input = get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                        self.image_patch_size
                    )   
                except FileNotFoundError as e:
                    # 关键：遇到缺失图片，直接跳过当前样本
                    print(f"[WARN] skip sample {i} due to missing image: {image_file}. err={e}")
                    return None   
                images.append(image_input)
                              
        elif "video" in sources:
            is_video = True
            images=None

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata
                )
                videos.append(video_input)
        else:
            images=None
            videos=None

        sources_orig = sources

        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        user_input = conversations[0]
        gpt_response = conversations[1]

        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        user_message = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"

        user_prompt = system_message + user_message
        assistant_prompt = gpt_response['content']

        data_dict = dict(
            prompt=user_prompt,
            assistant=assistant_prompt,
            images=images,
            videos=videos,
            video_kwargs=video_kwargs if is_video else None,
        )

        # gyn add
        if "metadata" in sources_orig:
            if "input_waypoints" in sources_orig["metadata"]:
                data_dict["input_waypoints"] = torch.tensor(sources_orig["metadata"]["input_waypoints"], dtype=torch.float32)
            if "gt_waypoints" in sources_orig["metadata"]:
                data_dict["gt_waypoints"] = torch.tensor(sources_orig["metadata"]["gt_waypoints"], dtype=torch.float32)
            if "arrive" in sources_orig["metadata"]:
                data_dict["arrive"] = torch.tensor(sources_orig["metadata"]["arrive"][0], dtype=torch.float32)
            if "input_rotation_matrix" in sources_orig["metadata"]:
                data_dict["input_rotation_matrix"] = torch.tensor(sources_orig["metadata"]["input_rotation_matrix"], dtype=torch.float32)
            if "step_scale" in sources_orig["metadata"]:
                data_dict["step_scale"] = torch.tensor(sources_orig["metadata"]["step_scale"], dtype=torch.float32)



        return data_dict
    
def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )

    return dict(train_dataset=grpo_dataset,
                eval_dataset=None)