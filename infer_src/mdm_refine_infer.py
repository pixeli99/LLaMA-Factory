#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MDM with Refinement Inference for LLaMA-Factory models
Based on generation_utils.py and demo_infer.py
"""

import sys
import time
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer


@dataclass
class MDMRefineConfig:
    """Configuration for MDM with refinement inference"""
    # MDM generation parameters
    max_new_tokens: int = 512
    steps: int = 512
    temperature: float = 0.2
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    alg: str = "entropy"  # 'origin', 'entropy', 'maskgit_plus', 'topk_margin'
    alg_temp: Optional[float] = 0.0
    eps: float = 1e-3
    
    # Refinement parameters
    refine_steps: int = 50
    refine_temperature: float = 0.1
    refine_threshold: float = 0.8  # Only refine tokens with confidence below this
    
    # Output control
    output_history: bool = True
    return_dict_in_generate: bool = True
    show_refinement_log: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MDMRefineModelOutput:
    """Output class for MDM with refinement"""
    def __init__(self, 
                 sequences: torch.LongTensor,
                 history: Optional[List[torch.LongTensor]] = None,
                 refinement_log: Optional[Dict] = None):
        self.sequences = sequences
        self.history = history
        self.refinement_log = refinement_log


class MDMRefinePipeline:
    """MDM generation with self-correction refinement"""
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[MDMRefineConfig] = None,
                 trust_remote_code: bool = True):
        """
        Initialize pipeline
        
        Args:
            model_path: Path to model checkpoint
            config: Configuration for generation
            trust_remote_code: Whether to trust remote code
        """
        self.config = config or MDMRefineConfig()
        
        # Load model and tokenizer (following demo_infer.py)
        print(f"Loading model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            trust_remote_code=trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        self.model = self.model.to(self.config.device).eval()
        
        # Get special token IDs
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.pad_token_id
            print(f"Warning: No mask_token_id found, using pad_token_id={self.mask_token_id}")
            
    def top_p_logits(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p filtering to logits (from generation_utils.py)"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits
    
    def top_k_logits(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits (from generation_utils.py)"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits
    
    def sample_tokens(self, logits: torch.Tensor, 
                     temperature: float = 0.0,
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None,
                     margin_confidence: bool = False,
                     neg_entropy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits (from generation_utils.py)"""
        
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
            
        probs = torch.softmax(logits, dim=-1)
        
        if temperature > 0:
            try:
                x0 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                confidence, x0 = probs.max(dim=-1)
        else:
            confidence, x0 = probs.max(dim=-1)
        
        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0]
            top2_probs = sorted_probs[:, 1]
            confidence = top1_probs - top2_probs
        
        if neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)
        
        return confidence, x0
    
    @torch.no_grad()
    def mdm_generate(self,
                    input_ids: torch.LongTensor,
                    attention_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.LongTensor, List]:
        """
        MDM generation phase (based on generation_utils.py _sample method)
        """
        device = input_ids.device
        max_length = input_ids.shape[1] + self.config.max_new_tokens
        
        # Initialize history
        histories = [] if self.config.output_history else None
        
        # Pad input_ids to max_length with mask tokens
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=self.mask_token_id)
        
        # Handle attention mask (following generation_utils.py lines 390-403)
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            
            # Create 2D attention mask
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"
        
        # Setup timesteps
        timesteps = torch.linspace(1, self.config.eps, self.config.steps + 1, device=device)
        
        # MDM generation loop
        for i in range(self.config.steps):
            mask_index = (x == self.mask_token_id)
            
            # Stop if no masks remain
            if not mask_index.any():
                print(f"Early stopping at step {i} - no masks remaining")
                break
            
            # Get model predictions
            logits = self.model(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Only process masked positions
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            
            # Apply sampling algorithm
            if self.config.alg == 'origin':
                p_transfer = 1 - s / t if i < self.config.steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + self.mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
                _, x0[transfer_index_t_s] = self.sample_tokens(
                    mask_logits[transfer_index_t_s],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k
                )
                x[mask_index] = x0.clone()
                
            else:
                # Handle entropy, maskgit_plus, topk_margin algorithms
                if self.config.alg == 'maskgit_plus':
                    confidence, x0 = self.sample_tokens(
                        mask_logits,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k
                    )
                elif self.config.alg == 'topk_margin':
                    confidence, x0 = self.sample_tokens(
                        mask_logits,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        margin_confidence=True
                    )
                elif self.config.alg == 'entropy':
                    confidence, x0 = self.sample_tokens(
                        mask_logits,
                        self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        neg_entropy=True
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {self.config.alg}")
                
                # Calculate number of tokens to transfer
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < self.config.steps - 1 else int(num_mask_token)
                
                # Build full confidence tensor
                full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                # Select tokens to transfer
                if number_transfer_tokens > 0:
                    if self.config.alg_temp is None or self.config.alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / self.config.alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    
                    # Update selected positions
                    x_ = torch.zeros_like(x, device=device, dtype=torch.long) + self.mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]
            
            # Record history
            if histories is not None:
                histories.append(x.clone())
        
        return x, histories
    
    @torch.no_grad()
    def refine(self, 
              x: torch.LongTensor,
              num_steps: Optional[int] = None) -> Tuple[torch.LongTensor, Dict]:
        """
        Refinement phase - improve generated text
        
        Args:
            x: Generated sequences
            num_steps: Number of refinement steps
            
        Returns:
            Refined sequences and refinement log
        """
        num_steps = num_steps or self.config.refine_steps
        device = x.device
        
        # Initialize refinement log
        refinement_log = {
            'total_refinements': 0,
            'refined_positions': [],
            'refined_tokens': [],
            'confidence_improvements': []
        }
        
        print(f"\nStarting refinement phase with {num_steps} steps...")
        
        for step in range(num_steps):
            # Get model predictions
            logits = self.model(x, attention_mask="full", tok_idx=None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Mask out special tokens
            logits[..., self.mask_token_id] = -1e6
            if self.tokenizer.pad_token_id is not None:
                logits[..., self.tokenizer.pad_token_id] = -1e6
            
            # Calculate confidence for all positions
            probs = F.softmax(logits / self.config.refine_temperature, dim=-1)
            current_token_probs = torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1)
            
            # Find low-confidence positions (excluding special tokens)
            is_special = (x == self.tokenizer.pad_token_id) | (x == self.tokenizer.eos_token_id)
            if self.tokenizer.bos_token_id is not None:
                is_special = is_special | (x == self.tokenizer.bos_token_id)
            
            low_confidence_mask = (current_token_probs < self.config.refine_threshold) & (~is_special)
            
            if not low_confidence_mask.any():
                print(f"Refinement converged at step {step} - all tokens above threshold")
                break
            
            # Sample new tokens for low-confidence positions
            confidence, new_tokens = self.sample_tokens(
                logits,
                temperature=self.config.refine_temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                neg_entropy=True  # Use entropy-based confidence
            )
            
            # Only refine positions where new token has higher confidence
            new_token_probs = torch.gather(probs, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
            should_refine = low_confidence_mask & (new_token_probs > current_token_probs)
            
            if should_refine.any():
                # Log refinements
                refined_positions = torch.where(should_refine)
                for batch_idx, pos_idx in zip(refined_positions[0].tolist(), refined_positions[1].tolist()):
                    old_token = x[batch_idx, pos_idx].item()
                    new_token = new_tokens[batch_idx, pos_idx].item()
                    old_word = self.tokenizer.decode([old_token])
                    new_word = self.tokenizer.decode([new_token])
                    old_conf = current_token_probs[batch_idx, pos_idx].item()
                    new_conf = new_token_probs[batch_idx, pos_idx].item()
                    
                    refinement_log['refined_positions'].append((batch_idx, pos_idx))
                    refinement_log['refined_tokens'].append({
                        'position': pos_idx,
                        'old': old_word,
                        'new': new_word,
                        'old_confidence': old_conf,
                        'new_confidence': new_conf,
                        'improvement': new_conf - old_conf
                    })
                    refinement_log['confidence_improvements'].append(new_conf - old_conf)
                
                # Apply refinements
                x = torch.where(should_refine, new_tokens, x)
                refinement_log['total_refinements'] += should_refine.sum().item()
                
                if self.config.show_refinement_log:
                    print(f"Step {step}: Refined {should_refine.sum().item()} tokens")
        
        # Summary statistics
        if refinement_log['confidence_improvements']:
            avg_improvement = sum(refinement_log['confidence_improvements']) / len(refinement_log['confidence_improvements'])
            refinement_log['avg_confidence_improvement'] = avg_improvement
            
            if self.config.show_refinement_log:
                print(f"\nRefinement Summary:")
                print(f"  Total tokens refined: {refinement_log['total_refinements']}")
                print(f"  Average confidence improvement: {avg_improvement:.4f}")
                print(f"\nTop 5 refinements by improvement:")
                sorted_refinements = sorted(
                    refinement_log['refined_tokens'], 
                    key=lambda x: x['improvement'], 
                    reverse=True
                )[:5]
                for ref in sorted_refinements:
                    print(f"  Position {ref['position']}: '{ref['old']}' -> '{ref['new']}' "
                          f"(+{ref['improvement']:.4f})")
        
        return x, refinement_log
    
    @torch.no_grad()
    def generate(self,
                prompt: Optional[Union[str, List[str]]] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                **kwargs) -> Union[List[str], MDMRefineModelOutput]:
        """
        Full generation pipeline with MDM + refinement
        
        Args:
            prompt: Text prompt(s)
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask
            **kwargs: Override config parameters
            
        Returns:
            Generated text or full output object
        """
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Handle input
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            
            # Apply chat template if needed
            messages = [{"role": "user", "content": p} for p in prompt]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True
            )
            input_ids = inputs.input_ids.to(self.config.device)
            attention_mask = inputs.attention_mask.to(self.config.device)
            
        elif input_ids is None:
            raise ValueError("Either prompt or input_ids must be provided")
        else:
            input_ids = input_ids.to(self.config.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.config.device)
        
        # Phase 1: MDM generation
        print("Phase 1: MDM Generation")
        start_time = time.time()
        x, histories = self.mdm_generate(input_ids, attention_mask)
        mdm_time = time.time() - start_time
        print(f"MDM generation completed in {mdm_time:.2f}s")
        
        # Phase 2: Refinement
        print("\nPhase 2: Self-Correction Refinement")
        start_time = time.time()
        x_refined, refinement_log = self.refine(x, self.config.refine_steps)
        refine_time = time.time() - start_time
        print(f"Refinement completed in {refine_time:.2f}s")
        
        # Add timing info to log
        refinement_log['mdm_time'] = mdm_time
        refinement_log['refine_time'] = refine_time
        
        # Return results
        if self.config.return_dict_in_generate:
            return MDMRefineModelOutput(
                sequences=x_refined,
                history=histories,
                refinement_log=refinement_log
            )
        else:
            # Decode and return text
            texts = []
            for i, seq in enumerate(x_refined):
                # Find where the input ends (for proper decoding)
                input_len = input_ids.shape[1] if prompt is not None else 0
                generated = seq[input_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                texts.append(text)
            
            return texts[0] if len(texts) == 1 else texts
    
    def __call__(self, *args, **kwargs):
        """Convenience method for generation"""
        return self.generate(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MDM with Refinement Inference")
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B",
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, 
                       default="Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset.",
                       help="Input prompt")
    parser.add_argument("--steps", type=int, default=512, help="Number of MDM steps")
    parser.add_argument("--refine_steps", type=int, default=50, help="Number of refinement steps")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--alg", type=str, default="entropy", 
                       choices=['origin', 'entropy', 'maskgit_plus', 'topk_margin'],
                       help="Sampling algorithm")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    config = MDMRefineConfig(
        steps=args.steps,
        refine_steps=args.refine_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
        device=args.device
    )
    
    # Initialize pipeline
    print("Initializing MDM+Refine Pipeline...")
    pipeline = MDMRefinePipeline(args.model_path, config)
    
    # Generate
    print(f"\nPrompt: {args.prompt}\n")
    print("="*80)
    
    result = pipeline.generate(
        prompt=args.prompt,
        return_dict_in_generate=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("Generated Text:")
    print("="*80)
    
    # Decode the full sequence
    generated_text = pipeline.tokenizer.decode(result.sequences[0], skip_special_tokens=True)
    print(generated_text)
    
    # Print refinement summary
    if result.refinement_log and result.refinement_log['total_refinements'] > 0:
        print("\n" + "="*80)
        print("Refinement Analysis:")
        print("="*80)
        log = result.refinement_log
        print(f"Total refinements: {log['total_refinements']}")
        if 'avg_confidence_improvement' in log:
            print(f"Average confidence improvement: {log['avg_confidence_improvement']:.4f}")
        print(f"MDM generation time: {log['mdm_time']:.2f}s")
        print(f"Refinement time: {log['refine_time']:.2f}s")