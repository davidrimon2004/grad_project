import os
import shutil
import subprocess
import sys
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

# In modern transformers, validate_quantization_for_training blocks training on
# quantized models unless wrapped in PEFT. In Video-LLaVA Stage 1, the mm_projector
# adapter is trained directly with a frozen quantized backbone.
try:
    import transformers.trainer
    import transformers.trainer_utils
    transformers.trainer.validate_quantization_for_training = lambda m: None
    transformers.trainer_utils.validate_quantization_for_training = lambda m: None
except Exception:
    pass

# Transformers 4.35+ compatibility helpers
try:
    from transformers.trainer_pt_utils import is_sagemaker_mp_enabled
except ImportError:
    try:
        from transformers.utils.import_utils import is_sagemaker_mp_enabled
    except ImportError:
        def is_sagemaker_mp_enabled():
            return False

try:
    from transformers.trainer_pt_utils import get_parameter_names
except ImportError:
    try:
        from transformers.pytorch_utils import get_parameter_names
    except ImportError:
        def get_parameter_names(model, forbidden_layer_types):
            decay_parameters = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                decay_parameters.append(name)
            return decay_parameters

try:
    from transformers.trainer_pt_utils import has_length
except ImportError:
    def has_length(dataset):
        try:
            return len(dataset) is not None
        except TypeError:
            return False

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except ImportError:
    try:
        from transformers.trainer import ALL_LAYERNORM_LAYERS
    except ImportError:
        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]

try:
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    try:
        from transformers.trainer import ShardedDDPOption
    except ImportError:
        class ShardedDDPOption:
            SIMPLE = "simple"
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def _maybe_async_upload_checkpoint_to_gdrive(
    checkpoint_dir: str,
    run_dir: str,
    ckpt_name: Optional[str] = None,
    exclude_checkpoints: bool = False,
) -> None:
    """
    Spawns a non-blocking background rclone upload for a freshly saved checkpoint.
    Controlled by env vars:
      - GDRIVE_AUTO_UPLOAD (default "1"; set "0" to disable)
      - GDRIVE_REMOTE      (default "gdrive:Video-LLaVA/checkpoints")
    No-ops silently if rclone is missing or the remote is not configured yet.
    """
    if os.environ.get("GDRIVE_AUTO_UPLOAD", "1") == "0":
        return
    remote_base = os.environ.get("GDRIVE_REMOTE", "gdrive:Video-LLaVA/checkpoints").rstrip("/")
    rclone_bin = shutil.which("rclone") or os.path.expanduser("~/.local/bin/rclone")
    if not rclone_bin or not os.path.isfile(rclone_bin):
        return
    watcher_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "gdrive_checkpoint_watcher.py")
    )
    run_name = os.path.basename(os.path.abspath(run_dir))
    if ckpt_name is None:
        ckpt_name = os.path.basename(os.path.abspath(checkpoint_dir))
    dst_remote = f"{remote_base}/{run_name}/{ckpt_name}"
    log_path = os.path.join(run_dir, "gdrive_upload.log")
    try:
        os.makedirs(run_dir, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")
        if os.path.isfile(watcher_script):
            cmd = [sys.executable, "-u", watcher_script, "--once", checkpoint_dir, "--remote", dst_remote]
            if exclude_checkpoints:
                cmd.append("--exclude-checkpoints")
        else:
            cmd = [
                rclone_bin, "copy", checkpoint_dir, dst_remote,
                "--transfers", "2", "--retries", "5",
                "--exclude", "global_step*/**",
                "--exclude", "optimizer.pt",
                "--exclude", "rng_state*.pth",
                "--exclude", "zero_to_fp32.py",
                "--exclude", "latest",
                "--exclude", "*.tmp",
                "--exclude", "gdrive_upload.log",
            ]
            if exclude_checkpoints:
                cmd.extend(["--exclude", "checkpoint-*/**"])
        subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
        logger.info(f"[GDriveSync] Triggered background upload: {checkpoint_dir} -> {dst_remote}")
    except Exception as exc:
        logger.warning(f"[GDriveSync] Failed to spawn background upload: {exc}")


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_non_lora_trainable_state_maybe_zero_3(named_params, always_include=("mm_projector", "early_fusion")):
    """Collects non-LoRA trainable weights (plus mm_projector/early_fusion even if frozen) under ZeRO-2/3."""
    to_return = {
        k: t for k, t in named_params
        if "lora_" not in k and (t.requires_grad or any(kw in k for kw in always_include))
    }
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_lora_state_maybe_zero_3(named_params, bias="none"):
    """Collects PEFT LoRA weights safely under ZeRO-2/3."""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError(f"Unsupported lora_bias='{bias}'")
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, tokenizer=None, processing_class=None, **kwargs):
        proc = processing_class if processing_class is not None else tokenizer
        try:
            super().__init__(*args, processing_class=proc, **kwargs)
        except TypeError:
            super().__init__(*args, tokenizer=proc, **kwargs)
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.tokenizer = proc

    def _get_train_sampler(self, dataset=None, *args, **kwargs) -> Optional[torch.utils.data.Sampler]:
        train_ds = dataset if dataset is not None else self.train_dataset
        if train_ds is None or not has_length(train_ds):
            return None

        if self.args.group_by_modality_length:
            lengths = getattr(train_ds, "modality_lengths", None)
            if lengths is None and hasattr(self.train_dataset, "modality_lengths"):
                lengths = self.train_dataset.modality_lengths
            if lengths is not None:
                return LengthGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                    lengths=lengths,
                    group_by_modality=True,
                )

        try:
            return super()._get_train_sampler(dataset, *args, **kwargs)
        except TypeError:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if getattr(self, "sharded_ddp", None) == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if ("mm_projector" in name or "early_fusion" in name)
                ]
                raw_groups = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
                optimizer_grouped_parameters = [g for g in raw_groups if len(g["params"]) > 0]
                if getattr(self.args, "local_rank", -1) in [0, -1]:
                    base_tensors = [
                        (n, p) for n, p in opt_model.named_parameters()
                        if p.requires_grad and n not in projector_parameters
                    ]
                    proj_tensors = [
                        (n, p) for n, p in opt_model.named_parameters()
                        if p.requires_grad and n in projector_parameters
                    ]
                    ef_tensors = [n for n, _ in proj_tensors if "early_fusion" in n]
                    base_elems = sum(p.numel() for _, p in base_tensors)
                    proj_elems = sum(p.numel() for _, p in proj_tensors)
                    logger.info(
                        f"[Optimizer] base group (lr={self.args.learning_rate}): "
                        f"{len(base_tensors)} tensors ({base_elems:,} params) | "
                        f"mm_projector+early_fusion group (lr={self.args.mm_projector_lr}): "
                        f"{len(proj_tensors)} tensors ({proj_elems:,} params, {len(ef_tensors)} early_fusion)"
                    )
            else:
                raw_groups = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                optimizer_grouped_parameters = [g for g in raw_groups if len(g["params"]) > 0]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if getattr(self, "sharded_ddp", None) == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
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

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if hasattr(self, "_get_output_dir"):
                try:
                    run_dir = self._get_output_dir(trial=trial)
                except Exception:
                    run_dir = self.args.output_dir
            else:
                run_dir = self.args.output_dir

            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Always include mm_projector + early_fusion so each downloaded checkpoint
            # is a 100% self-contained drop-in mm_projector.bin for benchmarking/eval.
            keys_to_match = ['mm_projector', 'vision_resampler', 'early_fusion']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if is_dist else 0

            if getattr(self.args, "local_rank", -1) in [0, -1]:
                os.makedirs(output_dir, exist_ok=True)
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                if self.lr_scheduler is not None:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if is_dist:
                torch.distributed.barrier()

            # Save trainable-only (fusion) optimizer state per rank (tiny; avoids saving 7B frozen model)
            if self.optimizer is not None:
                os.makedirs(output_dir, exist_ok=True)
                try:
                    opt_state = self.optimizer.state_dict()
                    torch.save(opt_state, os.path.join(output_dir, f"fusion_optimizer_rank{rank}.pt"))
                except Exception as exc:
                    logger.warning(f"[Checkpoint] Failed to save fusion optimizer state on rank {rank}: {exc}")

            if is_dist:
                torch.distributed.barrier()

            if getattr(self.args, "local_rank", -1) in [0, -1]:
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
                _maybe_async_upload_checkpoint_to_gdrive(output_dir, run_dir)
        else:
            try:
                super()._save_checkpoint(model, trial, metrics=metrics)
            except TypeError:
                super()._save_checkpoint(model, trial)

            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if hasattr(self, "_get_output_dir"):
                try:
                    run_dir = self._get_output_dir(trial=trial)
                except Exception:
                    run_dir = self.args.output_dir
            else:
                run_dir = self.args.output_dir
            output_dir = os.path.join(run_dir, checkpoint_folder)

            is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
            if getattr(self.args, "lora_enable", False):
                # Gather non-LoRA trainable params + mm_projector + early_fusion across ALL ranks (ZeRO-3 safe)
                non_lora_state_dict = get_non_lora_trainable_state_maybe_zero_3(self.model.named_parameters())

                # Defensive rescue: verify LoRA adapter file exists and is non-empty after HF save
                if is_dist:
                    torch.distributed.barrier()
                adapter_sft = os.path.join(output_dir, "adapter_model.safetensors")
                adapter_bin = os.path.join(output_dir, "adapter_model.bin")
                has_adapter = (
                    (os.path.isfile(adapter_sft) and os.path.getsize(adapter_sft) >= 1024)
                    or (os.path.isfile(adapter_bin) and os.path.getsize(adapter_bin) >= 1024)
                )
                peft_state = None
                if not has_adapter:
                    peft_state = get_peft_lora_state_maybe_zero_3(
                        self.model.named_parameters(),
                        getattr(self.args, "lora_bias", "none"),
                    )

                if getattr(self.args, "local_rank", -1) in [0, -1]:
                    os.makedirs(output_dir, exist_ok=True)
                    if peft_state is not None:
                        self.model.save_pretrained(output_dir, state_dict=peft_state)
                    self.model.config.save_pretrained(output_dir)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))
                    logger.info(
                        f"[Checkpoint] Saved Stage 2 LoRA config.json + non_lora_trainables.bin "
                        f"({len(non_lora_state_dict)} tensors) to {output_dir}"
                    )

                if is_dist:
                    torch.distributed.barrier()

            if getattr(self.args, "local_rank", -1) in [0, -1] and os.path.isdir(output_dir):
                _maybe_async_upload_checkpoint_to_gdrive(output_dir, run_dir)

    def log(self, logs, *args, **kwargs):
        if isinstance(logs, dict) and "loss" in logs:
            try:
                base_model = self.model.get_model() if hasattr(self.model, "get_model") else self.model
                gate_lin = getattr(base_model, "early_fusion_gate_linear", None)
                if gate_lin is not None and hasattr(gate_lin, "weight") and gate_lin.weight.numel() > 0:
                    w = gate_lin.weight.detach().float()
                    logs["gate_v2_w_norm"] = round(float(w.norm().item()), 6)
                    logs["gate_v2_max_abs"] = round(float(w.abs().max().item()), 6)
                    if gate_lin.bias is not None and gate_lin.bias.numel() == 1:
                        logs["gate_v2_bias"] = round(float(gate_lin.bias.detach().float().item()), 6)
            except Exception:
                pass
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            try:
                super()._save(output_dir, state_dict=state_dict)
            except TypeError:
                super()._save(output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            scheduler_file = os.path.join(checkpoint, "scheduler.pt")
            if os.path.isfile(scheduler_file) and self.lr_scheduler is not None:
                sched_state = torch.load(scheduler_file, map_location="cpu")
                self.lr_scheduler.load_state_dict(sched_state)
                logger.info(f"[Resume] Loaded lr_scheduler state from {scheduler_file}")
            elif self.lr_scheduler is not None:
                # Legacy checkpoint without scheduler.pt (e.g. checkpoint-4750):
                # fast-forward lr_scheduler to trainer_state.json's global_step so cosine LR
                # continues smoothly instead of restarting warmup from step 0.
                resume_step = 0
                ts_file = os.path.join(checkpoint, "trainer_state.json")
                if os.path.isfile(ts_file):
                    try:
                        import json as _json
                        with open(ts_file, "r", encoding="utf-8") as _f:
                            resume_step = int(_json.load(_f).get("global_step", 0))
                    except Exception as exc:
                        logger.warning(f"[Resume] Failed to parse {ts_file}: {exc}")
                if resume_step > 0:
                    for _ in range(resume_step):
                        self.lr_scheduler.step()
                    cur_lr = self.lr_scheduler.get_last_lr() if hasattr(self.lr_scheduler, "get_last_lr") else None
                    logger.info(f"[Resume] Fast-forwarded lr_scheduler by {resume_step} steps (no scheduler.pt); lr={cur_lr}")
                else:
                    logger.warning(f"[Resume] No scheduler.pt found at {checkpoint}; lr_scheduler will start from step 0")

            if self.optimizer is not None:
                is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
                world_size = torch.distributed.get_world_size() if is_dist else 1
                rank = torch.distributed.get_rank() if is_dist else 0
                if hasattr(self.optimizer, "dp_world_size"):
                    rank_files = [os.path.join(checkpoint, f"fusion_optimizer_rank{r}.pt") for r in range(world_size)]
                    if all(os.path.isfile(f) for f in rank_files):
                        state_list = [torch.load(f, map_location="cpu") for f in rank_files]
                        self.optimizer.load_state_dict(state_list, load_optimizer_states=True)
                        logger.info(f"[Resume] Loaded ZeRO-3 fusion optimizer states from {checkpoint} ({world_size} ranks)")
                    else:
                        logger.warning(f"[Resume] Missing fusion_optimizer_rank*.pt at {checkpoint}; skipping optimizer state")
                else:
                    opt_file = os.path.join(checkpoint, f"fusion_optimizer_rank{rank}.pt")
                    if os.path.isfile(opt_file):
                        self.optimizer.load_state_dict(torch.load(opt_file, map_location="cpu"))
                        logger.info(f"[Resume] Loaded fusion optimizer state from {opt_file}")
                    else:
                        logger.warning(f"[Resume] Missing {opt_file}; skipping optimizer state")
            return
        return super()._load_optimizer_and_scheduler(checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if resume_from_checkpoint is None:
            return
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            logger.info(f"Adapter-only checkpoint: skipping full-model weight loading from {resume_from_checkpoint}")
            return
        try:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        except TypeError:
            return super()._load_from_checkpoint(resume_from_checkpoint)
