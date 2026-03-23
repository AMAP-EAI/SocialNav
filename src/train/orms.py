import os
import re
import json
from datetime import datetime
from typing import Dict, List, Union, Callable
from math_verify import parse, verify
import cv2
import ast
import numpy as np
from PIL import Image
from torch import nn
import torch
from scipy.spatial.distance import euclidean
from dtw import dtw
import math


def _resolve_data_file(env_key: str) -> str:
    path = os.environ.get(env_key)
    if not path:
        raise FileNotFoundError(
            f"Set environment variable {env_key} to the file path "
            f"(grayscale occupancy image for NAV_OCCMAP_PATH; .npy distance map for NAV_DT_MAP_PATH)."
        )
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{env_key}={path!r} is not a file.")
    return path


TOP_LEFT_X = 1151.67
TOP_LEFT_Y = -991.29
IMAGE_WIDTH = 7648
IMAGE_HEIGHT = 7059
WIDTH_COORDINATE_RANGE = 1911.85
HEIGHT_COORDINATE_RANGE = 1764.88

_occ_map_arr = None
_distance_map_arr = None


def get_occ_map():
    global _occ_map_arr
    if _occ_map_arr is None:
        occ_path = _resolve_data_file("NAV_OCCMAP_PATH")
        img = Image.open(occ_path).convert("L")
        _occ_map_arr = np.array(img) > 240
    return _occ_map_arr


def get_distance_map():
    global _distance_map_arr
    if _distance_map_arr is None:
        dt_path = _resolve_data_file("NAV_DT_MAP_PATH")
        _distance_map_arr = np.load(dt_path)
    return _distance_map_arr

class BaseRewardFunction(nn.Module):
    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError

class FormatReward(BaseRewardFunction):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

class AccuracyReward(BaseRewardFunction):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, assistant, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        contents = [completion[0]["content"] for completion in completions]
        solution = [a['content'] for a in assistant]
        rewards = []
        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(answer_parsed, gold_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards

def to_numpy(obj):
    obj = obj.to(torch.float32)
    """Convert tensors (with optional grad) or arrays to numpy."""
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            return obj.detach().cpu().numpy()
        else:
            return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    elif isinstance(obj, (float, int)):
        return np.array([obj])
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
def world_to_pixel(x_real, y_real):
    """World (x, y) to pixel coordinates."""
    x_pixel = IMAGE_WIDTH * (TOP_LEFT_X - x_real) / WIDTH_COORDINATE_RANGE
    y_pixel = IMAGE_HEIGHT * (y_real - TOP_LEFT_Y) / HEIGHT_COORDINATE_RANGE
    return [int(round(x_pixel)), int(round(y_pixel))]

def pixel_to_world(x_pixel, y_pixel):
    """Pixel to world (x, y)."""
    x_real = TOP_LEFT_X - (x_pixel / IMAGE_WIDTH) * WIDTH_COORDINATE_RANGE
    y_real = TOP_LEFT_Y + (y_pixel / IMAGE_HEIGHT) * HEIGHT_COORDINATE_RANGE
    return [x_real, y_real]

def transform_back_positions(positions, current_pose_array, step_scale, data_mode='prue'):
    """Map model positions back to the original frame using pose and scale."""
    positions = np.array(positions)
    if positions.ndim != 2:
        raise ValueError("positions must be 2D")
    
    if positions.shape[1] == 2:
        positions = np.hstack([positions, np.zeros((positions.shape[0], 1))])
    elif positions.shape[1] != 3:
        raise ValueError("positions last dim must be 2 or 3")

    positions = positions * step_scale

    if data_mode == 'coord_layout_yz':
        positions[:, 0] *= -1
        positions[:, [1, 2]] = positions[:, [0, 1]]
    else:
        positions[:, 0] *= -1
        positions[:, [0, 1]] = positions[:, [1, 0]]

    homogeneous_positions = np.hstack([positions, np.ones((positions.shape[0], 1))])

    original_homogeneous = np.einsum('ij,nj->ni', np.array(current_pose_array), homogeneous_positions)

    original_positions = original_homogeneous[:, :3]

    return original_positions

class OccMapReward(BaseRewardFunction):
    def __init__(self, dilation_pixels: int = 0, radius_pixel: int = 2):
        self.occmap = None
        self.weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        self.dilation_pixels=dilation_pixels
        self.radius_pixel=radius_pixel
        self.load_occmap()

        
    def load_occmap(self):
        self.occmap = get_occ_map()
        if self.dilation_pixels:
            binary_occmap = self.occmap.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_pixels, self.dilation_pixels))
            dilated_binary = cv2.dilate(~binary_occmap, kernel, iterations=1)
            self.occmap = ~dilated_binary > 240
    
            
    def is_point_in_free_space(self, x: float, y: float) -> bool:
        u, v = world_to_pixel(x, y)

        if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
            return self.occmap[v, u]
        else:
            return False

    def is_line_safe(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        u0, v0 = world_to_pixel(x0, y0)
        u1, v1 = world_to_pixel(x1, y1)

        if not (0 <= u0 < IMAGE_WIDTH and 0 <= v0 < IMAGE_HEIGHT and
                0 <= u1 < IMAGE_WIDTH and 0 <= v1 < IMAGE_HEIGHT):
            return False

        height, width = self.occmap.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.line(mask, (u0, v0), (u1, v1), color=1, thickness=1)

        collision_mask = np.logical_and(mask.astype(bool), ~self.occmap)

        return not np.any(collision_mask)

    def is_circle_safe(self, x: float, y: float, radius_pixel: int) -> bool:
        u_center, v_center = world_to_pixel(x, y)
        radius_pixel = int(radius_pixel)

        height, width = self.occmap.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.circle(mask, (u_center, v_center), radius=radius_pixel, color=1, thickness=-1)

        collision_mask = np.logical_and(mask.astype(bool), ~self.occmap)

        return not np.any(collision_mask)

    def check_line_segment(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        u0, v0 = world_to_pixel(x0, y0)
        u1, v1 = world_to_pixel(x1, y1)

        if not (0 <= u0 < IMAGE_WIDTH and 0 <= v0 < IMAGE_HEIGHT and
                0 <= u1 < IMAGE_WIDTH and 0 <= v1 < IMAGE_HEIGHT):
            return False

        height, width = self.occmap.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.line(mask, (u0, v0), (u1, v1), color=1, thickness=1)

        intersection = np.logical_and(mask.astype(bool), self.occmap)

        return not np.any(intersection)

    def is_circle_in_free_space(self, x: float, y: float, radius_pixel: int) -> bool:
        u_center, v_center = world_to_pixel(x, y)

        height, width = self.occmap.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.circle(mask, (u_center, v_center), radius=radius_pixel, color=1, thickness=-1)

        intersection = np.logical_and(mask.astype(bool), self.occmap)

        return not np.any(intersection)

    def plot_coordinates_on_occmap(self, world_coords, color=(0, 255, 0), label=None, save_path=None, radius_pixel=3):
        occmap_gray = np.array(self.occmap).astype(np.uint8) * 255

        occmap_for_plot = cv2.cvtColor(occmap_gray, cv2.COLOR_GRAY2BGR)

        image_coords = []
        for idx, (x, y) in enumerate(world_coords):
            u, v = world_to_pixel(x, y)
            if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                image_coords.append((u, v))
            else:
                image_coords.append(None)

        for idx, (u, v) in enumerate(image_coords):
            if u is not None and v is not None:
                if idx == 0:
                    point_color = (255, 0, 0)
                else:
                    if self.is_point_in_free_space(world_coords[idx][0], world_coords[idx][1]):
                        point_color = color
                    else:
                        point_color = (0, 0, 255)
                cv2.circle(occmap_for_plot, (u, v), radius=radius_pixel, color=point_color, thickness=-1)

        if save_path:
            cv2.imwrite(save_path, occmap_for_plot)
        else:
            cv2.imshow('OccMap with Coordinates', occmap_for_plot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def __call__(self, completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        gt_waypoints_list = kwargs['gt_waypoints']
        input_rotation_matrix_list = kwargs['input_rotation_matrix']
        step_scale_list = kwargs['step_scale']

        rewards = []
        for i, (coords, gt_waypoints, input_rotation_matrix, step_scale) in enumerate(zip(contents, gt_waypoints_list, input_rotation_matrix_list, step_scale_list)):
            coords = to_numpy(coords)
            input_rotation_matrix = to_numpy(input_rotation_matrix)
            step_scale = to_numpy(step_scale)
            world_coords = transform_back_positions(coords, input_rotation_matrix, step_scale)
            world_coords = world_coords[:, [0,1]].tolist()

            sum_reward = 1.0
            total_penalty = 0.0
            for _, (x, y) in enumerate(world_coords):
                if not self.is_circle_safe(x, y, radius_pixel=self.radius_pixel):
                    total_penalty = 1.0
                    break

            reward = max(sum_reward - total_penalty, 0.0)
            rewards.append(reward)
        return rewards
 

class SmoothnessReward(BaseRewardFunction):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def __call__(self, completions, **kwargs):
        coords_list = [completion[0]["content"] for completion in completions]
        rewards = []
        for coords in coords_list:
            coords = to_numpy(coords)
            points = coords[:, :2] if coords.shape[1] >= 2 else coords
            step_dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
            std = np.std(step_dists)
            reward = np.exp(-std/self.alpha)
            rewards.append(reward)
        return rewards



class SimilarityToExpertReward(BaseRewardFunction):
    def __init__(self, metric: str = 'euclidean', max_diff: float = 1.0, weight: float = 0.7,
                 direction_weight: float = 0.3):
        self.metric = metric
        self.max_diff = max_diff
        self.weight = weight
        self.direction_weight = direction_weight

    def __call__(self, completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        gt_waypoints_list = kwargs['gt_waypoints']
        input_rotation_matrix_list = kwargs['input_rotation_matrix']
        step_scale_list = kwargs['step_scale']

        rewards = []
        gt_world_coords=None
        for i, (coords, gt_waypoints, input_rotation_matrix, step_scale) in enumerate(zip(contents, gt_waypoints_list, input_rotation_matrix_list, step_scale_list)):
            coords = to_numpy(coords)
            input_rotation_matrix = to_numpy(input_rotation_matrix)
            step_scale = to_numpy(step_scale)
            world_coords = transform_back_positions(coords, input_rotation_matrix, step_scale)
            world_coords = world_coords[:, [0,1]]

            if gt_world_coords is None:
                gt_waypoints = to_numpy(gt_waypoints)
                gt_world_coords = transform_back_positions(gt_waypoints, input_rotation_matrix, step_scale)
                gt_world_coords = gt_world_coords[:, [0,1]]

            avg_distance = self._euclidean_distance(world_coords, gt_world_coords)

            distance_reward = np.exp(-avg_distance / self.max_diff)
            distance_reward = np.clip(distance_reward, 0.0, 1.0)

            direction_similarity = self._direction_similarity(world_coords, gt_world_coords)

            direction_reward = (direction_similarity + 1.0) / 2.0
            direction_reward = np.clip(direction_reward, 0.0, 1.0)

            reward = self.weight * distance_reward + self.direction_weight * direction_reward

            rewards.append(reward)

        return rewards

    def _dtw_distance(self, pred, gt):
        alignment = dtw(pred, gt, dist_method=euclidean)
        return alignment.distance

    def _euclidean_distance(self, pred, gt):
        return np.mean([np.linalg.norm(p - g) for p, g in zip(pred, gt)])

    def _hausdorff_distance(self, pred, gt):
        def _point_to_set_distance(p, set_points):
            return np.min([np.linalg.norm(p - q) for q in set_points])

        return max(
            np.max([_point_to_set_distance(p, gt) for p in pred]),
            np.max([_point_to_set_distance(g, pred) for g in gt])
        )

    def _direction_similarity(self, pred, gt):
        if len(pred) < 2 or len(gt) < 2:
            return 0.0

        pred_directions = []
        for i in range(1, len(pred)):
            vec = pred[i] - pred[i-1]
            if np.linalg.norm(vec) == 0:
                pred_directions.append([0.0, 0.0])
            else:
                pred_directions.append(vec / np.linalg.norm(vec))

        gt_directions = []
        for i in range(1, len(gt)):
            vec = gt[i] - gt[i-1]
            if np.linalg.norm(vec) == 0:
                gt_directions.append([0.0, 0.0])
            else:
                gt_directions.append(vec / np.linalg.norm(vec))

        min_len = min(len(pred_directions), len(gt_directions))
        pred_directions = pred_directions[:min_len]
        gt_directions = gt_directions[:min_len]

        total_angle_diff = 0.0
        for pd, gd in zip(pred_directions, gt_directions):
            dot_product = np.dot(pd, gd)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = np.arccos(dot_product)
            total_angle_diff += angle_diff

        avg_angle_diff = total_angle_diff / len(pred_directions)
        similarity = np.cos(avg_angle_diff)
        return similarity


class DtMapReward(BaseRewardFunction):
    def __init__(self, alpha=0.5, beta=2.0):
        self.alpha = alpha
        self.beta = beta
        self.load_dtmap()
        self.pixel_length = WIDTH_COORDINATE_RANGE / IMAGE_WIDTH

    def load_dtmap(self):
        self.distance_map = get_distance_map()

    def is_point_in_free_space(self, x: float, y: float) -> bool:
        u, v = world_to_pixel(x, y)

        if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
            return get_occ_map()[v, u]
        else:
            return False

    def plot_coordinates_on_occmap(self, world_coords, color=(0, 255, 0), label=None, save_path=None, radius_pixel=3):
        occmap_gray = np.array(get_occ_map()).astype(np.uint8) * 255

        occmap_for_plot = cv2.cvtColor(occmap_gray, cv2.COLOR_GRAY2BGR)

        image_coords = []
        for idx, (x, y) in enumerate(world_coords):
            u, v = world_to_pixel(x, y)
            if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                image_coords.append((u, v))
            else:
                image_coords.append(None)

        for idx, (u, v) in enumerate(image_coords):
            if u is not None and v is not None:
                if idx == 0:
                    point_color = (255, 0, 0)
                else:
                    if self.is_point_in_free_space(world_coords[idx][0], world_coords[idx][1]):
                        point_color = color
                    else:
                        point_color = (0, 0, 255)
                cv2.circle(occmap_for_plot, (u, v), radius=radius_pixel, color=point_color, thickness=-1)

        if save_path:
            cv2.imwrite(save_path, occmap_for_plot)
        else:
            cv2.imshow('OccMap with Coordinates', occmap_for_plot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_distance_to_obstacle(self, x: float, y: float) -> float:
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0

        try:
            u, v = world_to_pixel(x, y)
        except (ValueError, TypeError):
            return 0.0

        if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
            return self.distance_map[v, u]
        else:
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        gt_waypoints_list = kwargs['gt_waypoints']
        input_rotation_matrix_list = kwargs['input_rotation_matrix']
        step_scale_list = kwargs['step_scale']

        rewards = []
        gt_world_coords=None
        for i, (coords, gt_waypoints, input_rotation_matrix, step_scale) in enumerate(zip(contents, gt_waypoints_list, input_rotation_matrix_list, step_scale_list)):
            coords = to_numpy(coords)
            input_rotation_matrix = to_numpy(input_rotation_matrix)
            step_scale = to_numpy(step_scale)
            world_coords = transform_back_positions(coords, input_rotation_matrix, step_scale)
            world_coords = world_coords[:, [0,1]].tolist()

            if gt_world_coords is None:
                gt_waypoints = to_numpy(gt_waypoints)
                gt_world_coords = transform_back_positions(gt_waypoints, input_rotation_matrix, step_scale)
                gt_world_coords = gt_world_coords[:, [0,1]].tolist()
                gt_distance_list = []
                for j, (x, y) in enumerate(gt_world_coords):
                    distance = self.get_distance_to_obstacle(x, y)
                    distance_in_meter = distance * self.pixel_length
                    gt_distance_list.append(distance_in_meter)
                gt_avg_distance = sum(gt_distance_list) / len(gt_world_coords)

            distance_list = []
            for j, (x, y) in enumerate(world_coords):
                distance = self.get_distance_to_obstacle(x, y)
                distance_in_meter = distance * self.pixel_length
                distance_list.append(distance_in_meter)
            total_distance = sum(distance_list)
            avg_distance = total_distance/len(world_coords)
            min_distance = min(distance_list)
            reward = self.beta*torch.sigmoid(torch.tensor((avg_distance-gt_avg_distance)/self.alpha))
            reward = np.clip(reward, 0, 1.2)
            if min_distance < 0.3:
                reward -= 1.0
            elif min_distance < 0.5:
                reward -= 0.5

            rewards.append(reward)
        return rewards
def get_reward_function(name: str, **kwargs) -> BaseRewardFunction:
    """Get a reward function by name with optional parameters."""
    if name not in reward_functions:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(reward_functions.keys())}")
    
    reward_class = reward_functions[name]
    return reward_class(**kwargs)

def load_reward_funcs_from_registry(names: List[str], **kwargs) -> List[BaseRewardFunction]:
    """Load reward functions from the registry by names."""
    reward_funcs = []
    for name in names:
        reward_func = get_reward_function(name, **kwargs)
        reward_funcs.append(reward_func)
    return reward_funcs

class LengthReward(BaseRewardFunction):
    def __init__(self, alpha=5.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        gt_waypoints_list = kwargs['gt_waypoints']
        input_rotation_matrix_list = kwargs['input_rotation_matrix']
        step_scale_list = kwargs['step_scale']

        rewards = []
        gt_length = None

        for i, (coords, gt_waypoints, input_rotation_matrix, step_scale) in enumerate(zip(contents, gt_waypoints_list, input_rotation_matrix_list, step_scale_list)):
            coords = to_numpy(coords)
            input_rotation_matrix = to_numpy(input_rotation_matrix)
            step_scale = to_numpy(step_scale)
            cur_vy = np.array([input_rotation_matrix[0,-1], input_rotation_matrix[1,-1]])
            world_coords = transform_back_positions(coords, input_rotation_matrix, step_scale)
            world_coords_xy = world_coords[:, [0, 1]]

            pred_length = np.sqrt((world_coords_xy[-1, 0]-cur_vy[0])**2 + (world_coords_xy[-1, 1]-cur_vy[1])**2)

            if gt_length is None:
                gt_waypoints_np = to_numpy(gt_waypoints)
                gt_world_coords = transform_back_positions(gt_waypoints_np, input_rotation_matrix, step_scale)
                gt_world_coords_xy = gt_world_coords[:, [0, 1]]
                gt_length = np.sqrt((gt_world_coords_xy[-1, 0]-cur_vy[0])**2 + (gt_world_coords_xy[-1, 1]-cur_vy[1])**2)

            length_diff = pred_length - gt_length

            reward = self.beta * torch.sigmoid(torch.tensor(length_diff / self.alpha)).item()
            reward = np.clip(reward, 0, 1.2)

            if pred_length < gt_length * 0.5:
                reward -= 0.5

            rewards.append(reward)
        return rewards

reward_functions = {
    'accuracy': AccuracyReward,
    'format': FormatReward,
    'occ_check': OccMapReward,
    'smoothness': SmoothnessReward,
    'similarity': SimilarityToExpertReward,
    'dt_compute': DtMapReward,
    'walk_length': LengthReward,
}