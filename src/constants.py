IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

# SYSTEM_MESSAGE = "You are a helpful assistant."

# MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

# gyn_add_start

# --- ORIGINAL CONTENT ---
SYSTEM_MESSAGE = "You are a helpful assistant."
MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

# --- MIGRATION ADDITIONS ---
# New VLA-related special tokens for continuous action grounding
VLA_INPUT_POS_TOKEN_1 = "<input_pos1>"
VLA_INPUT_POS_TOKEN_2 = "<input_pos2>"
VLA_INPUT_POS_TOKEN_3 = "<input_pos3>"
VLA_INPUT_POS_TOKEN_4 = "<input_pos4>"
VLA_INPUT_POS_TOKEN_5 = "<input_pos5>"
VLA_INPUT_TARGET_TOKEN = "<input_target>"
VLA_TIME_TOKEN = "<time>"
VLA_FLOW_MATCHING_TOKEN = "<flow_matching_policy>"

# gyn_add_end_251216

