# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

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
        # GIDD training path (superset; falls back to MDM when p_u=0)
        if getattr(self.finetuning_args, "use_gidd", False):
            src_mask = inputs["labels"] == IGNORE_INDEX
            final_loss = self.gidd_forward(
                model,
                inputs["input_ids"],
                src_mask,
            )
            return final_loss

        # DLM training path
        if getattr(self.finetuning_args, "use_dlm", False):
            src_mask = inputs["labels"] == IGNORE_INDEX
            final_loss, _, _ = self.diffusion_forward(
                model,
                inputs["input_ids"],
                src_mask,
            )
            return final_loss

        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        # GIDD evaluation path
        if getattr(self.finetuning_args, "use_gidd", False):
            if self.args.predict_with_generate:
                labels = inputs.pop("labels", None)
            else:
                labels = inputs.get("labels")

            src_mask = labels == IGNORE_INDEX if labels is not None else None
            loss = self.gidd_forward(model, inputs["input_ids"], src_mask)
            return loss, None, None

        # DLM evaluation path
        if getattr(self.finetuning_args, "use_dlm", False):
            if self.args.predict_with_generate:
                labels = inputs.pop("labels", None)
            else:
                labels = inputs.get("labels")

            src_mask = labels == IGNORE_INDEX if labels is not None else None
            loss, _, _ = self.diffusion_forward(model, inputs["input_ids"], src_mask)
            return loss, None, None

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    # ===================== DLM specific helpers =====================
    def transition(self, x_0: torch.Tensor, sigma: torch.Tensor, maskable_mask: torch.Tensor) -> torch.Tensor:
        move_chance = sigma
        move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
        mask_token_id = getattr(self.processing_class, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = self.processing_class.pad_token_id
        x_t = torch.where(move_indices, torch.as_tensor(mask_token_id, device=x_0.device, dtype=x_0.dtype), x_0)
        return x_t

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
        # m one-hot and uniform excluding mask
        m = torch.zeros(V, device=device, dtype=t.dtype)
        m[mask_token_id] = 1.0
        u = torch.ones(V, device=device, dtype=t.dtype)
        u[mask_token_id] = 0.0
        u = u / max(1, V - 1)
        beta_pi = (t / C_t) * m + (c_t / C_t) * u
        pi_t = beta_pi / torch.clamp(beta_t, min=self.finetuning_args.gidd_eps)
        # Ensure pi_t broadcasts over sequence length: (B, 1, V)
        pi_t = pi_t.unsqueeze(-2)
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
        # Minimal, stable vocab size inference:
        base = getattr(model, "module", model)
        get_out = getattr(base, "get_output_embeddings", None)
        if callable(get_out) and getattr(get_out(), "num_embeddings", None):
            vocab_size = int(get_out().num_embeddings)
        else:
            get_in = getattr(base, "get_input_embeddings", None)
            if callable(get_in) and getattr(get_in(), "num_embeddings", None):
                vocab_size = int(get_in().num_embeddings)
            else:
                vocab_size = int(getattr(getattr(base, "config", {}), "vocab_size", 0) or 32000)

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

    def diffusion_forward(
        self,
        model: "torch.nn.Module",
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        sampling_eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if src_mask is None:
            src_mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)

        # Sample noise level
        t = (1 - sampling_eps) * torch.rand(x.shape[0], device=x.device) + sampling_eps

        # Compute sigma and dsigma
        sigma = t
        dsigma = torch.reciprocal(t)

        # Apply noise to input
        x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)

        # Forward pass (intentionally no attention mask)
        logits = model(input_ids=x_t, attention_mask=None).logits

        # Apply mask for loss computation
        mask_token_id = getattr(self.processing_class, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = self.processing_class.pad_token_id
        loss_mask = x_t == mask_token_id

        # Shift loss
        logits = logits[:, :-1]
        loss_mask = loss_mask[:, 1:]
        x_target = x[:, 1:]

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            x_target.reshape(-1),
            reduction="none",
        ).float().reshape(batch_size, -1)

        loss = loss.masked_fill(~loss_mask, 0)
        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum().clamp_min(1)
        unweighted_loss = loss.sum() / loss_mask.sum().clamp_min(1)

        return final_loss, unweighted_loss, x_t

    def _get_dynamic_ratio(self) -> float:
        if not hasattr(self.state, "global_step") or not hasattr(self.state, "max_steps"):
            logger.warning("State not initialized, using default ratio")
            return 0.9

        progress = self.state.global_step / self.state.max_steps
        if progress < 1 / 5:
            return progress * 5
        else:
            return 1.0

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
