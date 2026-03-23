import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence


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

# gyn_add_end

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
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
        print('image_resized_width:', self.image_resized_w)     # gyn
        print('image_resized_height:', self.image_resized_h)
        self.input_waypoint_augment = data_args.input_waypoint_augment
        if self.input_waypoint_augment:
            print('using input_waypoint_augment!!!!!!')

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

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
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

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
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        # gyn: 保存原始 sources 以便读取 metadata
        sources_orig = sources

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if "Qwen2.5" in self.model_id:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt', 
                        **video_kwargs
                    )
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif "Qwen3" in self.model_id:

                    videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                    video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                    video_datas_for_turn = list(video_datas_for_turn)
                    video_metadatas_for_turn = list(video_metadatas_for_turn)

                    inputs = processor(
                        text=[user_input],
                        images=images,
                        videos=video_datas_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors='pt',
                        **video_kwargs,
                        video_metadata=video_metadatas_for_turn,
                    )
                else:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt'
                    )
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key and len(all_pixel_values) > 0 and len(all_image_grid_thw) > 0: # gyn changed
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
        else:
            print(f"[Warning] No valid image or video data found for sample {i}")
            return None  #gyn: 如果没有有效的图像或视频数据，返回 None

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        # gyn add
        if "metadata" in sources_orig:
            if "input_waypoints" in sources_orig["metadata"]:
                if self.input_waypoint_augment:
                    prob = np.random.uniform(0, 1)
                    input_waypoints = sources_orig["metadata"]["input_waypoints"]
                    disturb_num = np.array(input_waypoints).shape[0] - 2
                    if prob < 0.5:
                        data_dict["input_waypoints"] = torch.tensor(sources_orig["metadata"]["input_waypoints"], dtype=torch.float32)
                    elif prob < 0.7:
                        disturb = (np.random.randn(disturb_num, 2) * 0.2).tolist()+[[0, 0], [0, 0]]
                        input_waypoints = (np.array(input_waypoints) + np.array(disturb)).tolist()
                        data_dict["input_waypoints"] = torch.tensor(input_waypoints, dtype=torch.float32)
                    elif prob < 0.8:
                        disturb = (np.random.randn(disturb_num, 2) * 1).tolist()+[[0, 0], [0, 0]]
                        input_waypoints = (np.array(input_waypoints) + np.array(disturb)).tolist()
                        data_dict["input_waypoints"] = torch.tensor(input_waypoints, dtype=torch.float32)
                    else:
                        disturb = (np.random.randn(disturb_num, 2) * 0.02).tolist()+[input_waypoints[-2], input_waypoints[-1]]
                        data_dict["input_waypoints"] = torch.tensor(disturb, dtype=torch.float32)
                else:
                    data_dict["input_waypoints"] = torch.tensor(sources_orig["metadata"]["input_waypoints"], dtype=torch.float32)
            if "gt_waypoints" in sources_orig["metadata"]:
                data_dict["gt_waypoints"] = torch.tensor(sources_orig["metadata"]["gt_waypoints"], dtype=torch.float32)
            if "arrive" in sources_orig["metadata"]:
                data_dict["arrive"] = torch.tensor(sources_orig["metadata"]["arrive"][0], dtype=torch.float32)
            if "train_branch" in sources_orig["metadata"]:
                if sources_orig["metadata"]["train_branch"]!='fm':
                    data_dict["train_branch"] = 'ar'
                else:
                    data_dict["train_branch"] = 'fm'
            else:
                data_dict["train_branch"] = 'fm'


        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        # 添加额外参数的批次列表 gyn add
        batch_input_waypoints = []
        batch_gt_waypoints = []
        batch_arrive = []
        batch_branch_label = []

        # --- gyn: 过滤无效样本 (防止因为某个图片加载失败导致训练崩溃) ---
        valid_examples = [example for example in examples if example is not None]
        if len(valid_examples) == 0:
            return None

        
        for example in valid_examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
            
            # 收集额外参数 gyn
            if "input_waypoints" in keys:
                batch_input_waypoints.append(example["input_waypoints"])
            if "gt_waypoints" in keys:
                batch_gt_waypoints.append(example["gt_waypoints"])
            if "arrive" in keys:
                batch_arrive.append(example["arrive"])
            if "train_branch" in keys:
                batch_branch_label.append(example["train_branch"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        # 添加额外参数到批次字典中 gyn
        if len(batch_input_waypoints) > 0:
            data_dict["input_waypoints"] = torch.stack(batch_input_waypoints)
        if len(batch_gt_waypoints) > 0:
            data_dict["gt_waypoints"] = torch.stack(batch_gt_waypoints)
        if len(batch_arrive) > 0:
            data_dict["arrive"] = torch.stack(batch_arrive)
        if len(batch_branch_label) > 0:
            data_dict["train_branch"] = batch_branch_label # list

        return data_dict

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    # sft_dataset = SupervisedDataset(
    #     data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    # )
    list_data_dict = data_args.data_path.split(',')         # gyn change
    sft_dataset_list = []
    for data_path in list_data_dict:
        print('loading from: ', data_path)
        sft_dataset = SupervisedDataset(
            data_path=data_path, processor=processor, data_args=data_args, model_id=model_id
        )
        sft_dataset_list.append(sft_dataset)
    
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset_list,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
