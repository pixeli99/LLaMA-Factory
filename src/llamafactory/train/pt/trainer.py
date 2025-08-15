# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomTrainer(Trainer):
    r"""Trainer for pre-training with optional GIDD/DLM-style objectives.

    Notes:
    - Mirrors the GIDD/DLM training path available in the SFT trainer, so that
      users can enable `--use_gidd` even in `--stage pt`.
    - For pretraining datasets we generally do not have prompt/response label
      masks. When `labels` are present (from `DataCollatorForLanguageModeling`),
      they will typically contain no `IGNORE_INDEX`; in that case all tokens are
      treated as learnable targets (i.e., `src_mask` = False everywhere).
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:  # keep a reference for older transformers
            self.processing_class = kwargs.get("tokenizer")  # type: ignore[attr-defined]

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Enable GIDD loss in PT stage to match SFT capabilities.
        if getattr(self.finetuning_args, "use_gidd", False):
            labels = inputs.get("labels")
            src_mask = labels == IGNORE_INDEX if isinstance(labels, torch.Tensor) else None
            return self.gidd_forward(model, inputs["input_ids"], src_mask)

        return super().compute_loss(model, inputs, *args, **kwargs)

    # ===================== GIDD helpers (training loss only) =====================
    def _gidd_mixing_schedule(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pu = torch.as_tensor(self.finetuning_args.gidd_pu, device=t.device, dtype=t.dtype)
        gamma = torch.as_tensor(self.finetuning_args.gidd_gamma, device=t.device, dtype=t.dtype)
        eps = torch.as_tensor(self.finetuning_args.gidd_eps, device=t.device, dtype=t.dtype)

        B = (2.0 ** gamma) * pu / torch.clamp(1.0 - pu, min=eps)
        c_t = B * torch.pow(t, gamma / 2.0) * torch.pow(1.0 - t, gamma / 2.0)
        C_t = 1.0 + c_t
        alpha_t = (1.0 - t) / C_t
        beta_t = 1.0 - alpha_t
        return alpha_t, beta_t, c_t, C_t, B

    def _gidd_build_pi_t(self, t: torch.Tensor, vocab_size: int, mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t, beta_t, c_t, C_t, B = self._gidd_mixing_schedule(t)
        device = t.device
        V = vocab_size
        m = torch.zeros(V, device=device, dtype=t.dtype)
        m[mask_token_id] = 1.0
        u = torch.ones(V, device=device, dtype=t.dtype)
        u[mask_token_id] = 0.0
        u = u / max(1, V - 1)
        beta_pi = (t / C_t) * m + (c_t / C_t) * u
        pi_t = beta_pi / torch.clamp(beta_t, min=self.finetuning_args.gidd_eps)
        return alpha_t, beta_t, pi_t, B

    def _gidd_sample_z_t(self, x_ids: torch.Tensor, t: torch.Tensor, vocab_size: int, mask_id: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        gamma = self.finetuning_args.gidd_gamma
        pu = self.finetuning_args.gidd_pu
        eps = self.finetuning_args.gidd_eps
        B = (2.0 ** gamma) * pu / max(eps, (1.0 - pu))
        c_t = B * (t ** (gamma / 2.0)) * ((1.0 - t) ** (gamma / 2.0))
        C_t = 1.0 + c_t
        alpha_t = (1.0 - t) / C_t
        p_mask = t / C_t
        p_unif = c_t / C_t

        u = torch.rand_like(x_ids.float())
        z = x_ids.clone()
        stay = (u < alpha_t)
        u2 = torch.rand_like(x_ids.float())
        mask_region = (~stay) & (u2 < p_mask / (p_mask + p_unif + eps))
        z[mask_region] = mask_id
        unif_region = (~stay) & (~mask_region)
        if unif_region.any():
            r = torch.randint(low=0, high=vocab_size - 1, size=(int(unif_region.sum().item()),), device=x_ids.device)
            r = r + (r >= mask_id).long()
            z[unif_region] = r
        return z, (alpha_t, p_mask, p_unif, C_t)

    def gidd_forward(
        self,
        model: "torch.nn.Module",
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        sampling_eps: float = 1e-4,
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)

        t = (1 - 2 * sampling_eps) * torch.rand(x.shape[0], device=x.device) + sampling_eps
        t_b = t.view(-1, 1)

        mask_token_id = getattr(getattr(self, "processing_class", None), "mask_token_id", None)
        if mask_token_id is None:
            pad_id = getattr(getattr(self, "processing_class", None), "pad_token_id", None)
            mask_token_id = pad_id if pad_id is not None else 0
        vocab_size = int(model.config.vocab_size)

        x_t, (alpha_t, p_mask, p_unif, C_t) = self._gidd_sample_z_t(x, t_b, vocab_size, mask_token_id)
        x_t = torch.where(src_mask, x, x_t)

        logits = model(input_ids=x_t, attention_mask=None).logits
        logits = logits[:, :-1, :]
        x_theta = logits.log_softmax(dim=-1).exp()
        x = x[:, 1:]
        x_t = x_t[:, 1:]
        src_mask = src_mask[:, 1:]

        alpha_s, beta_s, pi_t_vec, B = self._gidd_build_pi_t(t_b, vocab_size, mask_token_id)
        x_onehot = F.one_hot(x, num_classes=vocab_size).float()
        alpha_b = alpha_s.view(-1, 1, 1)
        beta_b = beta_s.view(-1, 1, 1)
        pi_vec = pi_t_vec
        qx_t = (alpha_b * x_onehot) + (beta_b * pi_vec)
        qx_t = qx_t.clamp_min(self.finetuning_args.gidd_eps)
        qtheta_t = (alpha_b * x_theta) + (beta_b * pi_vec)
        qtheta_t = qtheta_t.clamp_min(self.finetuning_args.gidd_eps)

        kl = (qx_t * (qx_t.log() - qtheta_t.log())).sum(dim=-1)
        z_onehot = F.one_hot(x_t, num_classes=vocab_size).float()
        p = (qx_t * z_onehot).sum(dim=-1).clamp_min(self.finetuning_args.gidd_eps)
        q = (qtheta_t * z_onehot).sum(dim=-1).clamp_min(self.finetuning_args.gidd_eps)
        ratio = p / q
        is_term = ratio - ratio.log() - 1.0

        lam = (alpha_s / torch.clamp(1.0 - alpha_s, min=self.finetuning_args.gidd_eps)).log().view(-1, 1)
        ind_mask = (x_t == mask_token_id).float()
        ind_clean = (x_t == x).float()
        wmax = self.finetuning_args.gidd_wmax
        dyn = wmax * (1.0 + ind_mask + ((B / vocab_size) * torch.exp(-lam / 2.0) - 1.0) * ind_clean)

        valid_mask = (~src_mask).float()
        loss_tokens = valid_mask * dyn * (kl + is_term)
        loss = loss_tokens.sum(dim=-1).mean()
        return loss
