#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GIDD Inference for LLaMA-Factory trained models
Implements GIDD (Generalized Importance-weighted Denoising Diffusion) inference
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Add LLaMA-Factory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.llamafactory.model import load_model, load_tokenizer
from src.llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments


@dataclass
class GIDDConfig:
    """Configuration for GIDD inference"""
    gidd_pu: float = 0.5
    gidd_gamma: float = 2.0
    gidd_eps: float = 1e-3
    t_eps: float = 1e-3
    steps: int = 512
    temperature: float = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GIDDModelWrapper:
    """Wrapper to make LLaMA-Factory models compatible with GIDD interface"""
    
    def __init__(self, model: PreTrainedModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        
    def __call__(self, input_ids: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning logits"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            return outputs.logits


class GIDDNoiseSchedule:
    """GIDD noise schedule implementation"""
    
    def __init__(self, tokenizer, config: GIDDConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = len(tokenizer)
        
        # Get special token IDs
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if self.mask_token_id is None:
            # Fallback to pad token if mask token not available
            self.mask_token_id = tokenizer.pad_token_id
            
    def mixing_schedule(self, t: torch.Tensor):
        """Compute GIDD mixing schedule parameters"""
        pu = self.config.gidd_pu
        gamma = self.config.gidd_gamma
        eps = self.config.gidd_eps
        
        B = (2.0 ** gamma) * pu / max(eps, 1.0 - pu)
        c_t = B * torch.pow(t, gamma / 2.0) * torch.pow(1.0 - t, gamma / 2.0)
        C_t = 1.0 + c_t
        alpha_t = (1.0 - t) / C_t
        beta_t = t / C_t + c_t / C_t
        
        return alpha_t, beta_t, c_t, C_t, B
    
    def build_pi_t(self, t: torch.Tensor):
        """Build prior distribution pi_t"""
        alpha_t, beta_t, c_t, C_t, B = self.mixing_schedule(t)
        device = t.device
        V = self.vocab_size
        
        # Create mask and uniform distributions
        m = torch.zeros(V, device=device, dtype=t.dtype)
        m[self.mask_token_id] = 1.0
        
        u = torch.ones(V, device=device, dtype=t.dtype)
        u[self.mask_token_id] = 0.0
        u = u / max(1, V - 1)
        
        # Compute pi_t
        beta_pi = (t / C_t) * m + (c_t / C_t) * u
        pi_t = beta_pi / torch.clamp(beta_t, min=self.config.gidd_eps)
        
        # Ensure proper shape for broadcasting
        pi_t = pi_t.unsqueeze(-2)  # (B, 1, V)
        
        return alpha_t, beta_t, pi_t, beta_pi
    
    def sample_prior(self, shape: tuple) -> torch.Tensor:
        """Sample from prior (all mask tokens)"""
        return torch.full(shape, self.mask_token_id, dtype=torch.long)
    
    def get_alpha_betapi(self, t: torch.Tensor):
        """Get alpha and beta*pi for given timestep"""
        alpha_t, beta_t, pi_t, beta_pi = self.build_pi_t(t)
        return alpha_t.view(-1, 1, 1), beta_pi.unsqueeze(-2)


class GIDDSampler:
    """GIDD Sampler for iterative denoising"""
    
    def __init__(self, model_wrapper: GIDDModelWrapper, noise_schedule: GIDDNoiseSchedule, config: GIDDConfig):
        self.model = model_wrapper
        self.noise_schedule = noise_schedule
        self.config = config
        self.tokenizer = model_wrapper.tokenizer
        
    def sample_tokens(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with temperature and top-p/top-k filtering"""
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
            
        # Apply top-p filtering
        if self.config.top_p is not None and self.config.top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(mask, -float('inf'))
            
        # Apply top-k filtering
        if self.config.top_k is not None:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float('inf'))
            
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        
        if self.config.temperature > 0:
            samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            samples = samples.view(probs.shape[:-1])
            confidence = torch.gather(probs, -1, samples.unsqueeze(-1)).squeeze(-1)
        else:
            confidence, samples = probs.max(dim=-1)
            
        return samples, confidence
    
    def denoising_step(self, z_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Single denoising step from timestep t to s"""
        # Get model predictions
        logits = self.model(z_t, t)
        
        # Mask out the mask token in predictions
        logits[..., self.noise_schedule.mask_token_id] = -1e6
        
        # Get noise schedule parameters
        alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)
        
        # Compute transition probabilities
        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s
        
        # Compute q(z_s|z_t, x)
        probs = F.softmax(logits, dim=-1)
        q_s = self.noise_schedule.get_alpha_betapi(s)[0] * probs + self.noise_schedule.get_alpha_betapi(s)[1]
        q_t = self.noise_schedule.get_alpha_betapi(t)[0] * probs + self.noise_schedule.get_alpha_betapi(t)[1]
        
        # One-hot encoding of z_t
        vz_t = F.one_hot(z_t, num_classes=self.noise_schedule.vocab_size).float()
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        
        # Compute transition kernel
        beta_pi_ts_at_zt = beta_pi_ts.expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt).squeeze(-1)
        
        # Compute posterior q(z_s|z_t)
        q_st = q_ts * q_s / torch.clamp(q_zt.unsqueeze(-1), min=self.config.gidd_eps)
        
        # Normalize
        q_st = q_st / q_st.sum(dim=-1, keepdim=True)
        
        # Sample z_s
        z_s = torch.multinomial(q_st.view(-1, q_st.size(-1)), 1)
        z_s = z_s.view(z_t.shape)
        
        return z_s
    
    @torch.no_grad()
    def generate(self, 
                 num_samples: int = 1,
                 prompts: Optional[List[str]] = None,
                 prompt_ids: Optional[torch.Tensor] = None,
                 show_progress: bool = True) -> List[str]:
        """Generate samples using GIDD"""
        device = self.config.device
        
        # Handle prompts
        if prompts is not None:
            # Tokenize prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            prompt_ids = inputs["input_ids"].to(device)
            num_samples = len(prompts)
        elif prompt_ids is not None:
            prompt_ids = prompt_ids.to(device)
            num_samples = prompt_ids.shape[0]
        
        # Initialize with mask tokens
        max_length = self.config.max_length
        if prompt_ids is not None:
            prompt_length = prompt_ids.shape[1]
            z_t = torch.full((num_samples, max_length), self.noise_schedule.mask_token_id, 
                           dtype=torch.long, device=device)
            z_t[:, :prompt_length] = prompt_ids
        else:
            z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device)
        
        # Setup timesteps
        steps = self.config.steps
        ts = torch.linspace(1 - self.config.t_eps, self.config.t_eps, steps + 1, device=device)
        
        # Iterative denoising
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(steps), desc="GIDD Sampling")
        else:
            iterator = range(steps)
            
        for i in iterator:
            t = ts[i].unsqueeze(0).expand(num_samples, 1)
            s = ts[i + 1].unsqueeze(0).expand(num_samples, 1)
            
            # Only update mask tokens
            mask_indices = (z_t == self.noise_schedule.mask_token_id)
            
            if mask_indices.any():
                # Perform denoising step
                z_t_new = self.denoising_step(z_t, t, s)
                
                # Only update masked positions
                z_t = torch.where(mask_indices, z_t_new, z_t)
            
            # Check for early stopping (no more masks)
            if not mask_indices.any():
                break
        
        # Decode to text
        texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
        return texts


class GIDDPipeline:
    """Complete GIDD inference pipeline for LLaMA-Factory models"""
    
    def __init__(self, model_path: str, config: Optional[GIDDConfig] = None, device: Optional[str] = None):
        """
        Initialize GIDD pipeline
        
        Args:
            model_path: Path to the model checkpoint
            config: GIDD configuration
            device: Device to run inference on
        """
        self.config = config or GIDDConfig()
        if device:
            self.config.device = device
            
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.load_model(model_path)
        
        # Setup GIDD components
        self.model_wrapper = GIDDModelWrapper(self.model, self.tokenizer)
        self.noise_schedule = GIDDNoiseSchedule(self.tokenizer, self.config)
        self.sampler = GIDDSampler(self.model_wrapper, self.noise_schedule, self.config)
        
    def load_model(self, model_path: str):
        """Load LLaMA-Factory trained model"""
        # Try loading as LLaMA-Factory model first
        try:
            # Create minimal arguments for loading
            model_args = ModelArguments(model_name_or_path=model_path)
            finetuning_args = FinetuningArguments()
            
            # Load tokenizer
            tokenizer_module = load_tokenizer(model_args)
            self.tokenizer = tokenizer_module["tokenizer"]
            
            # Ensure mask token exists
            if not hasattr(self.tokenizer, 'mask_token_id') or self.tokenizer.mask_token_id is None:
                # Add mask token if not present
                self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            
            # Load model
            self.model = load_model(
                self.tokenizer, 
                model_args, 
                finetuning_args, 
                is_trainable=False
            )
            self.model.to(self.config.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Failed to load as LLaMA-Factory model: {e}")
            print("Falling back to standard transformers loading...")
            
            # Fallback to standard transformers loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Ensure mask token
            if not hasattr(self.tokenizer, 'mask_token_id') or self.tokenizer.mask_token_id is None:
                self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.eval()
    
    def generate(self, 
                 prompts: Optional[Union[str, List[str]]] = None,
                 num_samples: int = 1,
                 **kwargs) -> List[str]:
        """
        Generate text using GIDD
        
        Args:
            prompts: Input prompts (optional)
            num_samples: Number of samples to generate
            **kwargs: Additional configuration overrides
            
        Returns:
            List of generated texts
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Generate
        return self.sampler.generate(
            num_samples=num_samples,
            prompts=prompts,
            show_progress=True
        )
    
    def __call__(self, *args, **kwargs):
        """Convenience method for generation"""
        return self.generate(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GIDD Inference for LLaMA-Factory models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=512, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gidd_pu", type=float, default=0.5, help="GIDD p_uniform parameter")
    parser.add_argument("--gidd_gamma", type=float, default=2.0, help="GIDD gamma parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    config = GIDDConfig(
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        gidd_pu=args.gidd_pu,
        gidd_gamma=args.gidd_gamma,
        device=args.device
    )
    
    # Initialize pipeline
    pipeline = GIDDPipeline(args.model_path, config)
    
    # Generate
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        results = pipeline.generate(args.prompt, num_samples=args.num_samples)
    else:
        print("\nGenerating unconditional samples...")
        results = pipeline.generate(num_samples=args.num_samples)
    
    # Print results
    print("\n" + "="*50)
    print("Generated Samples:")
    print("="*50)
    for i, text in enumerate(results, 1):
        print(f"\n[Sample {i}]")
        print(text)
        print("-"*50)