from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class CLSArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0
    mlp_head_dim: Optional[int] = field(default=0)
    mlp_head_dropout: Optional[float] = field(default=0.0)
    
    loss_type : str = field(
        default="cross_entropy",
        metadata={"help": "Loss type to use. Should be one of `cross_entropy`, `focal_loss`, `class_balanced_cross_entropy`, or `class_balanced_focal_loss`."}
    )
    focal_alpha: Optional[str] = field(
        default=None,
        metadata={"help": "Focal Loss alpha value. If None use CrossEntropyLoss. ex '1.0,7.5'"}
    )
    focal_gamma: float = field(
        default=0.0,
        metadata={"help": "Focal Loss gamma value"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification."}
    )
    class_balanced_beta: float = field(
        default=0.999,
        metadata={"help": "Beta value for Class Balanced Loss. If 0.0, use standard CrossEntropyLoss."}
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change in the monitored quantity to qualify as an improvement."}
    )

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True

    ####### gyn add - start
    action_dim: int = 2
    action_chunk: int = 5
    flow_matching_policy: bool = True
    num_flow_steps: int = 3
    action_former: bool = True
    query_action_layer: int = 4
    sigma: float = 0.0
    ar_lambda_loss: float = 1.0
    ####### gyn add - end


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    flow_matching_lr: Optional[float] = None    #gyn
    action_former_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True

    ####### gyn add - start
    action_dim: int = 2
    action_chunk: int = 5
    flow_matching_policy: bool = True
    num_flow_steps: int = 3
    action_former: bool = True
    query_action_layer: int = 4
    sigma: float = 0.0
    min_value: float = -1.25
    max_value: float = 1.25
    ar_lambda_loss: float = 1.0
    gradient_checkpointing_use_reentrant: bool = True # 默认--gradient_checkpointing_kwargs '{"use_reentrant": true}'
    # Flow Matching模块训练控制
    freeze_flow_matching: bool = field(default=False, metadata={"help": "Freeze flow matching model (velocity_predictor)"})
    freeze_action_former: bool = field(default=False, metadata={"help": "Freeze action former (query attention layers)"})
    freeze_input_wp_encoder: bool = field(default=False, metadata={"help": "Freeze input waypoint encoder"})
    ####### gyn add - end

@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    flow_matching_lr: Optional[float] = None    #gyn
    action_former_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_loss: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )

@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    flow_matching_lr: Optional[float] = None    #gyn
    action_former_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512
    use_liger_loss: bool = True

    ####### gyn add - start
    action_dim: int = 2
    action_chunk: int = 5
    flow_matching_policy: bool = False
    num_flow_steps: int = 3
    query_action_layer: int = 4
    sigma: float = 0.0
    ar_lambda_loss: float = 1.0
    noise_level: float = 0.0
    # reward_weights: Optional[list] = None
    # 奖励函数配置
    reward_funcs: str = field(
        default="",
        metadata={"help": "Comma-separated list of reward functions to use. Available: accuracy, format, react_format, repetition, cosine"}
    )
    # 奖励函数参数配置
    reward_func_params: str = field(
        default="",
        metadata={"help": "JSON string for reward function parameters. Example: '{\"occ_check\": {\"dilation_pixels\": 2, \"radius_pixel\": 3}, \"repetition\": {\"repetition_n_grams\": 4, \"repetition_max_penalty\": -0.5}}'"}
    )
    add_noise_step_num: int = None # 控制增加扰动的训练时间步数
    min_value: float = -1.25
    max_value: float = 1.25
    sde_mode: str = 'cps'
    # Flow Matching模块训练控制
    freeze_flow_matching: bool = field(default=False, metadata={"help": "Freeze flow matching model (velocity_predictor)"})
    freeze_action_former: bool = field(default=False, metadata={"help": "Freeze action former (query attention layers)"})
    freeze_input_wp_encoder: bool = field(default=False, metadata={"help": "Freeze input waypoint encoder"})
    ####### gyn add - end


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_path: str= field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_image_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation image data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: Optional[int] = field(default=None, metadata={"help": "Frames per second for video data."})
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})
    input_waypoint_augment: bool = False # gyn add
    image_height_to_width_ratio: Optional[float] = field(default=None, metadata={"help": "Image aspect ratio for image data."}) # czy add  长/宽: height / width