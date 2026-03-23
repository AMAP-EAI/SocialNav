import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

from torch.utils.data import Sampler
from itertools import accumulate
import torch.distributed as dist
from torch.utils.data import ConcatDataset

class FixedProportionSampler(Sampler): # gyn add
    """
    支持 DDP 的非流式采样器，保证：
    - 每个 epoch 按数据量比例从各数据集采样
    - 每个 batch 来自同一数据集（pure batch）
    - 所有 batch 全局 shuffle
    - DDP 下各 rank 无重复
    """
    def __init__(
        self,
        datasets,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.lengths = [len(ds) for ds in datasets]
        self.offsets = list(accumulate([0] + self.lengths))  # 0, len0, len0+len1, ...

        if drop_last:
            self.batches_per_dataset = [l // batch_size for l in self.lengths]
        else:
            self.batches_per_dataset = [(l + batch_size - 1) // batch_size for l in self.lengths]

        self.epoch = 0  # 默认 epoch

    def __iter__(self):
        generator = torch.Generator()
        seed = self.seed + self.epoch
        generator.manual_seed(seed)

        all_batch_indices = []

        # 1. 为每个数据集生成 batch
        for i, dataset_len in enumerate(self.lengths):
            n_batches = self.batches_per_dataset[i]
            if n_batches == 0:
                continue

            indices = torch.randperm(dataset_len, generator=generator).tolist()

            # 填充不足 batch 的情况（仅当 drop_last=False）
            required = n_batches * self.batch_size
            if len(indices) < required:
                extended = []
                while len(extended) < required:
                    extended.extend(indices)
                indices = extended[:required]

            # 切成 batch 并加偏移
            for j in range(n_batches):
                start = j * self.batch_size
                end = start + self.batch_size
                batch = [self.offsets[i] + indices[k] for k in range(start, end)]
                all_batch_indices.append(batch)

        # 2. 全局 shuffle
        if self.shuffle:
            order = torch.randperm(len(all_batch_indices), generator=generator).tolist()
            all_batch_indices = [all_batch_indices[i] for i in order]

        # 3. DDP 分片：每个 rank 只取一部分，避免重复
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            all_batch_indices = all_batch_indices[rank::world_size]

            # 重要：如果分片后为空，至少返回一个 batch（避免 DataLoader 报错）
            if len(all_batch_indices) == 0 and len(all_batch_indices) > 0:
                all_batch_indices = [all_batch_indices[0]]

        return iter(all_batch_indices)

    def __len__(self):
        total = sum(self.batches_per_dataset)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            return total // world_size
        return total

    def set_epoch(self, epoch: int):
        """由 Trainer 自动调用"""
        self.epoch = epoch

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            flow_matching_parameters = []   # gyn
            action_former_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            # Flow Matching Model (velocity_predictor, time_encoder, etc.) gyn
            if self.args.flow_matching_lr is not None:
                lr_mapper["flow_matching"] = self.args.flow_matching_lr
                flow_matching_parameters = [
                    name for name, _ in opt_model.named_parameters() 
                    if "flow_matching_model" in name
                ]

            # Action Former (query_action, query_multihead_attn, input_wp_encoder) gyn
            if self.args.action_former_lr is not None:
                lr_mapper["action_former"] = self.args.action_former_lr
                action_former_parameters = [
                    name for name, _ in opt_model.named_parameters() 
                    if any(keyword in name for keyword in [
                        "query_action", "query_multihead_attn", 
                        "query_multihead_multi_attn", "input_wp_encoder"
                    ])
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )

                if flow_matching_parameters:    # gyn
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in flow_matching_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.flow_matching_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in flow_matching_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.flow_matching_lr,
                            },
                        ]
                    )
                    print(f"✅ Flow Matching: using custom learning rate {self.args.flow_matching_lr}")
                
                if action_former_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in action_former_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.action_former_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in action_former_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.action_former_lr,
                            },
                        ]
                    )
                    print(f"✅ Action Former: using custom learning rate {self.args.action_former_lr}")
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            # Skip filesystem writes on non-saving ranks
            if not self.args.should_save:
                return

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenSFTTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs, num_items_in_batch):
        
    #     loss = super().training_step(model, inputs, num_items_in_batch)

    #     for name, p in model.named_parameters():
    #         if 'visual' in name and 'lora_' in name:
    #             g = p.grad
    #             if g is None:
    #                 print(f"[NONE] {name}")
    #             else:
    #                 print(f"[GRAD] {name} | norm={g.norm().item():.3e}")
    
    #     return loss
    def get_train_dataloader(self): # gyn add
        print('loading train dataloader!')
        if isinstance(self.train_dataset, list):
            batch_size = self._train_batch_size
            datasets = self.train_dataset
            concat_dataset = ConcatDataset(datasets)
            print('concat_dataset size: ', len(concat_dataset))
            sampler = FixedProportionSampler(
                datasets=datasets,
                batch_size=batch_size,
                # shuffle=self.args.train_dataloader_shuffle,
                shuffle=True, # 默认打乱
                # shuffle=False, # 默认不打乱
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            print('loading train sampler!')
            # czy add: 添加日志以验证每个 batch 来自哪个数据集
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(f"Rank {rank}: Using FixedProportionSampler with {len(datasets)} datasets")
            for i, ds in enumerate(datasets):
                print(f"Rank {rank}: Dataset {i} size: {len(ds)}")

            def custom_collate_fn(batch):
                # 过滤掉 None 值
                batch = [item for item in batch if item is not None]
                if not batch:
                    print("[Warning] Empty batch after filtering None values, returning empty batch")
                    return self.data_collator([])  # 返回一个空批次
                return self.data_collator(batch)

            return torch.utils.data.DataLoader(
                concat_dataset,
                batch_sampler=sampler,
                # collate_fn=self.data_collator,
                collate_fn=custom_collate_fn,  # 使用自定义的 collate_fn
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor,
            )
        else:
            return super().get_train_dataloader()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels") if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
