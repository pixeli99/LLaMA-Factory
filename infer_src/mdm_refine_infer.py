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
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def red(text): return f"{Colors.FAIL}{text}{Colors.ENDC}"
    
    @staticmethod
    def green(text): return f"{Colors.OKGREEN}{text}{Colors.ENDC}"
    
    @staticmethod
    def blue(text): return f"{Colors.OKBLUE}{text}{Colors.ENDC}"
    
    @staticmethod
    def yellow(text): return f"{Colors.WARNING}{text}{Colors.ENDC}"
    
    @staticmethod
    def cyan(text): return f"{Colors.OKCYAN}{text}{Colors.ENDC}"
    
    @staticmethod
    def bold(text): return f"{Colors.BOLD}{text}{Colors.ENDC}"


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
    refine_threshold: float = 0.8  # Only refine tokens with confidence below this (for old method)
    refine_mode: str = "gidd"  # "gidd" for single-point greedy, "batch" for multi-point
    refine_patience: int = 32  # Early stopping patience for GIDD mode
    refine_t0: float = 0.01  # Time parameter for GIDD self-correction
    
    # Output control
    output_history: bool = True
    return_dict_in_generate: bool = True
    show_refinement_log: bool = True
    show_step_by_step: bool = True  # Show each refinement step
    show_mdm_progress: bool = True  # Show MDM generation progress
    
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
        
        # Track MDM progress
        if self.config.show_mdm_progress:
            from tqdm import tqdm
            iterator = tqdm(range(self.config.steps), desc="MDM Generation", ncols=100)
        else:
            iterator = range(self.config.steps)
        
        # MDM generation loop
        for i in iterator:
            mask_index = (x == self.mask_token_id)
            
            # Stop if no masks remain
            if not mask_index.any():
                if self.config.show_mdm_progress:
                    print(f"\n{Colors.green('✓')} Early stopping at step {i} - no masks remaining")
                break
            
            # Show progress
            if self.config.show_mdm_progress and i % 50 == 0 and i > 0:
                masks_remaining = mask_index.sum().item()
                print(f"\n  Step {i}: {masks_remaining} masks remaining")
            
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
    def refine_gidd(self, 
                   x: torch.LongTensor,
                   num_steps: Optional[int] = None,
                   temperature: Optional[float] = None,
                   patience: Optional[int] = None) -> Tuple[torch.LongTensor, Dict]:
        """
        GIDD-style self-correction: single-point greedy refinement
        - Each step only modifies 1 token (most conservative)
        - Uses self-accuracy for early stopping with patience
        - Based on the GIDD paper's self-correction implementation
        
        Args:
            x: Generated sequences
            num_steps: Number of refinement steps
            temperature: Temperature for sampling (default from config)
            patience: Early stopping patience (default from config)
            
        Returns:
            Refined sequences and refinement log
        """
        num_steps = num_steps or self.config.refine_steps
        temperature = temperature or self.config.refine_temperature
        patience = patience or self.config.refine_patience
        device = x.device
        
        # Initialize refinement log
        refinement_log = {
            'total_refinements': 0,
            'refined_positions': [],
            'refined_tokens': [],
            'confidence_improvements': [],
            'self_accuracy_history': [],
            'step_logs': []
        }
        
        print(f"\n{Colors.bold('='*80)}")
        print(f"{Colors.bold('Starting GIDD-Style Refinement (Single-Point Greedy)')}")
        print(f"{Colors.bold('='*80)}")
        print(f"Settings: max_steps={num_steps}, patience={patience}, temp={temperature:.2f}\n")
        
        # Keep track of best self-accuracy for early stopping
        best_self_acc = -1.0
        stall_count = 0
        
        # Store original for comparison
        original_x = x.clone()
        
        for step in range(num_steps):
            # Get model predictions
            logits = self.model(x, attention_mask="full", tok_idx=None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Mask out special tokens
            logits[..., self.mask_token_id] = -1e6
            if self.tokenizer.pad_token_id is not None:
                logits[..., self.tokenizer.pad_token_id] = -1e6
            if self.tokenizer.eos_token_id is not None:
                logits[..., self.tokenizer.eos_token_id] = -1e6
            if self.tokenizer.bos_token_id is not None:
                logits[..., self.tokenizer.bos_token_id] = -1e6
            
            # Calculate probabilities and argmax
            probs = F.softmax(logits / temperature, dim=-1)
            z_argmax = probs.argmax(dim=-1)
            
            # Calculate self-accuracy (how many tokens match argmax)
            self_acc = (z_argmax == x).float().mean().item()
            refinement_log['self_accuracy_history'].append(self_acc)
            
            # Check for early stopping with patience
            if self_acc > best_self_acc + 1e-12:
                best_self_acc = self_acc
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= patience:
                    if self.config.show_refinement_log:
                        print(f"\n{Colors.yellow('⚠')} Early stopping at step {step} - patience exhausted (self-acc: {self_acc:.4f})")
                    break
            
            # Sample candidate tokens for each position
            # Using multinomial sampling for diversity
            candidates = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x)
            
            # Find positions where candidate differs from current
            diff_mask = (candidates != x)
            
            # Skip special tokens
            is_special = torch.zeros_like(x, dtype=torch.bool)
            if self.tokenizer.pad_token_id is not None:
                is_special |= (x == self.tokenizer.pad_token_id)
            if self.tokenizer.eos_token_id is not None:
                is_special |= (x == self.tokenizer.eos_token_id)
            if self.tokenizer.bos_token_id is not None:
                is_special |= (x == self.tokenizer.bos_token_id)
            
            diff_mask = diff_mask & (~is_special)
            
            if not diff_mask.any():
                if self.config.show_refinement_log:
                    print(f"\n{Colors.green('✓')} Converged at step {step} - no changes proposed")
                break
            
            # Calculate scores for candidates (probability of the candidate token)
            candidate_probs = probs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
            
            # Optional: Calculate improvement scores (p(candidate) - p(current))
            current_probs = probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            improvement_scores = candidate_probs - current_probs
            
            # Only consider positions with positive improvement
            scores = torch.where(diff_mask & (improvement_scores > 0), candidate_probs, torch.tensor(-1.0, device=device))
            
            # Find the single best position to change (global maximum)
            if scores.max() <= 0:
                if self.config.show_refinement_log:
                    print(f"\n{Colors.yellow('⚠')} No improvements found at step {step}")
                continue
            
            # Flatten to find global best position
            flat_scores = scores.view(-1)
            flat_idx = flat_scores.argmax()
            batch_idx = flat_idx // scores.size(1)
            pos_idx = flat_idx % scores.size(1)
            
            # Get token information for logging
            old_token = x[batch_idx, pos_idx].item()
            new_token = candidates[batch_idx, pos_idx].item()
            old_word = self.tokenizer.decode([old_token])
            new_word = self.tokenizer.decode([new_token])
            old_conf = current_probs[batch_idx, pos_idx].item()
            new_conf = candidate_probs[batch_idx, pos_idx].item()
            improvement = improvement_scores[batch_idx, pos_idx].item()
            
            # Apply the single change
            x[batch_idx, pos_idx] = new_token
            
            # Log the refinement
            refinement_info = {
                'step': step,
                'position': pos_idx.item() if hasattr(pos_idx, 'item') else pos_idx,
                'old': old_word,
                'new': new_word,
                'old_confidence': old_conf,
                'new_confidence': new_conf,
                'improvement': improvement,
                'self_accuracy': self_acc
            }
            
            refinement_log['refined_positions'].append((batch_idx.item() if hasattr(batch_idx, 'item') else batch_idx, 
                                                        pos_idx.item() if hasattr(pos_idx, 'item') else pos_idx))
            refinement_log['refined_tokens'].append(refinement_info)
            refinement_log['confidence_improvements'].append(improvement)
            refinement_log['total_refinements'] += 1
            
            # Show step-by-step progress
            if self.config.show_step_by_step:
                old_display = Colors.red(repr(old_word))
                new_display = Colors.green(repr(new_word))
                conf_change = Colors.yellow(f"+{improvement:.4f}")
                print(f"{Colors.cyan(f'Step {step+1}/{num_steps}')}: "
                      f"Position {pos_idx}: {old_display} → {new_display} "
                      f"(conf: {old_conf:.3f} → {new_conf:.3f} {conf_change}) "
                      f"[self-acc: {self_acc:.4f}]")
        
        # Final summary
        if refinement_log['total_refinements'] > 0:
            final_self_acc = refinement_log['self_accuracy_history'][-1] if refinement_log['self_accuracy_history'] else 0
            initial_self_acc = refinement_log['self_accuracy_history'][0] if refinement_log['self_accuracy_history'] else 0
            
            if self.config.show_refinement_log:
                print(f"\n{Colors.bold('='*80)}")
                print(f"{Colors.bold('GIDD Refinement Summary')}")
                print(f"{Colors.bold('='*80)}")
                print(f"  {Colors.cyan('Total refinements:')} {Colors.bold(str(refinement_log['total_refinements']))}")
                print(f"  {Colors.cyan('Self-accuracy:')} {initial_self_acc:.4f} → {final_self_acc:.4f} "
                      f"({Colors.green(f'+{final_self_acc - initial_self_acc:.4f}')})")
                
                if refinement_log['confidence_improvements']:
                    avg_improvement = sum(refinement_log['confidence_improvements']) / len(refinement_log['confidence_improvements'])
                    print(f"  {Colors.cyan('Average confidence gain:')} {Colors.bold(f'{avg_improvement:.4f}')}")
                
                # Show top refinements
                if len(refinement_log['refined_tokens']) > 0:
                    print(f"\n{Colors.bold('Top refinements by improvement:')}")
                    sorted_refinements = sorted(
                        refinement_log['refined_tokens'],
                        key=lambda x: x['improvement'],
                        reverse=True
                    )[:5]
                    
                    for i, ref in enumerate(sorted_refinements, 1):
                        old_display = Colors.red(repr(ref['old']))
                        new_display = Colors.green(repr(ref['new']))
                        improvement = Colors.yellow(f"+{ref['improvement']:.4f}")
                        print(f"  {i}. Position {ref['position']:3d}: {old_display} → {new_display} ({improvement})")
        else:
            if self.config.show_refinement_log:
                print(f"\n{Colors.yellow('No refinements made - text was already optimal or converged')}")
        
        return x, refinement_log
    
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
            'confidence_improvements': [],
            'step_logs': []  # Track each step's changes
        }
        
        print(f"\n{Colors.bold('='*80)}")
        print(f"{Colors.bold('Starting Refinement Phase')}")
        print(f"{Colors.bold('='*80)}")
        print(f"Settings: {num_steps} steps, threshold={self.config.refine_threshold:.2f}, temp={self.config.refine_temperature:.2f}\n")
        
        # Keep track of original text for comparison
        original_x = x.clone()
        
        for step in range(num_steps):
            step_log = {
                'step': step,
                'refinements': []
            }
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
                step_refinements = []
                
                for batch_idx, pos_idx in zip(refined_positions[0].tolist(), refined_positions[1].tolist()):
                    old_token = x[batch_idx, pos_idx].item()
                    new_token = new_tokens[batch_idx, pos_idx].item()
                    old_word = self.tokenizer.decode([old_token])
                    new_word = self.tokenizer.decode([new_token])
                    old_conf = current_token_probs[batch_idx, pos_idx].item()
                    new_conf = new_token_probs[batch_idx, pos_idx].item()
                    
                    refinement_info = {
                        'position': pos_idx,
                        'old': old_word,
                        'new': new_word,
                        'old_confidence': old_conf,
                        'new_confidence': new_conf,
                        'improvement': new_conf - old_conf
                    }
                    
                    refinement_log['refined_positions'].append((batch_idx, pos_idx))
                    refinement_log['refined_tokens'].append(refinement_info)
                    refinement_log['confidence_improvements'].append(new_conf - old_conf)
                    step_refinements.append(refinement_info)
                    step_log['refinements'].append(refinement_info)
                
                # Apply refinements
                x = torch.where(should_refine, new_tokens, x)
                refinement_log['total_refinements'] += should_refine.sum().item()
                refinement_log['step_logs'].append(step_log)
                
                # Show step-by-step refinements
                if self.config.show_step_by_step:
                    print(f"\n{Colors.cyan(f'Step {step+1}/{num_steps}')}: Refined {Colors.bold(str(should_refine.sum().item()))} tokens")
                    
                    # Show each refinement
                    for ref in step_refinements[:5]:  # Show max 5 per step
                        old_display = Colors.red(repr(ref['old']))
                        new_display = Colors.green(repr(ref['new']))
                        conf_change = Colors.yellow(f"+{ref['improvement']:.3f}")
                        print(f"  Position {ref['position']:3d}: {old_display} → {new_display} (conf: {ref['old_confidence']:.3f} → {ref['new_confidence']:.3f} {conf_change})")
                    
                    if len(step_refinements) > 5:
                        print(f"  ... and {len(step_refinements) - 5} more refinements")
                    
                    # Show current text snippet with highlights
                    if self.config.show_step_by_step and step % 10 == 0:
                        self._show_text_with_highlights(x[0], refined_positions[1].tolist()[:20])  # Show first 20 positions
        
        # Summary statistics
        if refinement_log['confidence_improvements']:
            avg_improvement = sum(refinement_log['confidence_improvements']) / len(refinement_log['confidence_improvements'])
            refinement_log['avg_confidence_improvement'] = avg_improvement
            
            if self.config.show_refinement_log:
                print(f"\n{Colors.bold('='*80)}")
                print(f"{Colors.bold('Refinement Summary')}")
                print(f"{Colors.bold('='*80)}")
                print(f"  {Colors.cyan('Total tokens refined:')} {Colors.bold(str(refinement_log['total_refinements']))}")
                print(f"  {Colors.cyan('Average confidence improvement:')} {Colors.bold(f'{avg_improvement:.4f}')}")
                
                # Show distribution of improvements
                improvements = refinement_log['confidence_improvements']
                if improvements:
                    min_imp = min(improvements)
                    max_imp = max(improvements)
                    print(f"  {Colors.cyan('Improvement range:')} {min_imp:.4f} to {max_imp:.4f}")
                
                print(f"\n{Colors.bold('Top 10 refinements by improvement:')}")
                sorted_refinements = sorted(
                    refinement_log['refined_tokens'], 
                    key=lambda x: x['improvement'], 
                    reverse=True
                )[:10]
                
                for i, ref in enumerate(sorted_refinements, 1):
                    old_display = Colors.red(repr(ref['old']))
                    new_display = Colors.green(repr(ref['new']))
                    improvement = Colors.yellow(f"+{ref['improvement']:.4f}")
                    print(f"  {i:2d}. Position {ref['position']:3d}: {old_display} → {new_display} ({improvement})")
        else:
            if self.config.show_refinement_log:
                print(f"\n{Colors.yellow('No refinements were made - text was already optimal!')}")
        
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
        print(f"\n{Colors.bold('='*80)}")
        print(f"{Colors.bold(Colors.cyan('Phase 1: MDM Generation'))}")
        print(f"{Colors.bold('='*80)}")
        start_time = time.time()
        x, histories = self.mdm_generate(input_ids, attention_mask)
        mdm_time = time.time() - start_time
        print(f"\n{Colors.green('✓')} MDM generation completed in {Colors.bold(f'{mdm_time:.2f}s')}")
        
        # Store original for comparison
        x_after_mdm = x.clone()
        
        # Phase 2: Refinement
        print(f"\n{Colors.bold('='*80)}")
        refine_mode_display = "GIDD-Style (Single-Point)" if self.config.refine_mode == "gidd" else "Batch (Multi-Point)"
        print(f"{Colors.bold(Colors.cyan(f'Phase 2: Self-Correction Refinement ({refine_mode_display})'))}")
        print(f"{Colors.bold('='*80)}")
        start_time = time.time()
        
        # Choose refinement method based on config
        if self.config.refine_mode == "gidd":
            x_refined, refinement_log = self.refine_gidd(x, self.config.refine_steps)
        else:
            x_refined, refinement_log = self.refine(x, self.config.refine_steps)
            
        refine_time = time.time() - start_time
        print(f"\n{Colors.green('✓')} Refinement completed in {Colors.bold(f'{refine_time:.2f}s')}")
        
        # Show before/after comparison if requested
        if self.config.show_refinement_log and refinement_log['total_refinements'] > 0:
            self.show_before_after_comparison(x_after_mdm[0], x_refined[0])
        
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
    
    def _show_text_with_highlights(self, token_ids: torch.LongTensor, highlighted_positions: List[int], context_window: int = 50):
        """Show text with highlighted positions"""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        
        # Build text with highlights
        output_parts = []
        for i, token in enumerate(tokens[:context_window]):
            if i in highlighted_positions:
                output_parts.append(Colors.yellow(f"[{token}]"))
            else:
                output_parts.append(token)
        
        text = self.tokenizer.convert_tokens_to_string(output_parts)
        print(f"\n  {Colors.bold('Current text snippet (highlighted = recently refined):')}")
        print(f"  {text}...")
    
    def show_before_after_comparison(self, original_ids: torch.LongTensor, refined_ids: torch.LongTensor, max_length: int = 200):
        """Show before/after comparison with colored diff"""
        original_tokens = self.tokenizer.convert_ids_to_tokens(original_ids[:max_length].tolist())
        refined_tokens = self.tokenizer.convert_ids_to_tokens(refined_ids[:max_length].tolist())
        
        print(f"\n{Colors.bold('Before/After Comparison:')}")
        print(f"{Colors.bold('-'*80)}")
        
        # Find differences
        diff_positions = []
        for i, (orig, ref) in enumerate(zip(original_tokens, refined_tokens)):
            if orig != ref:
                diff_positions.append(i)
        
        # Show original
        print(f"{Colors.cyan('Original:')}")
        orig_parts = []
        for i, token in enumerate(original_tokens):
            if i in diff_positions:
                orig_parts.append(Colors.red(token))
            else:
                orig_parts.append(token)
        print(self.tokenizer.convert_tokens_to_string(orig_parts))
        
        # Show refined
        print(f"\n{Colors.cyan('Refined:')}")
        ref_parts = []
        for i, token in enumerate(refined_tokens):
            if i in diff_positions:
                ref_parts.append(Colors.green(token))
            else:
                ref_parts.append(token)
        print(self.tokenizer.convert_tokens_to_string(ref_parts))
        
        print(f"\n{Colors.yellow(f'Total differences: {len(diff_positions)} tokens')}")
    
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
    parser.add_argument("--refine_mode", type=str, default="gidd", 
                       choices=['gidd', 'batch'],
                       help="Refinement mode: 'gidd' for single-point, 'batch' for multi-point")
    parser.add_argument("--refine_patience", type=int, default=32, 
                       help="Early stopping patience for GIDD mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--refine_temperature", type=float, default=0.1, help="Refinement temperature")
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
        refine_mode=args.refine_mode,
        refine_patience=args.refine_patience,
        temperature=args.temperature,
        refine_temperature=args.refine_temperature,
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
    print(f"\n{Colors.bold('='*80)}")
    print(f"{Colors.bold(Colors.cyan('Final Generated Text:'))}")
    print(f"{Colors.bold('='*80)}")
    
    # Decode the full sequence
    generated_text = pipeline.tokenizer.decode(result.sequences[0], skip_special_tokens=True)
    print(generated_text)
    
    # Print timing summary
    print(f"\n{Colors.bold('='*80)}")
    print(f"{Colors.bold('Performance Metrics:')}")
    print(f"{Colors.bold('='*80)}")
    log = result.refinement_log
    print(f"  {Colors.cyan('MDM generation time:')} {Colors.bold(f'{log['mdm_time']:.2f}s')}")
    print(f"  {Colors.cyan('Refinement time:')} {Colors.bold(f'{log['refine_time']:.2f}s')}")
    print(f"  {Colors.cyan('Total time:')} {Colors.bold(f'{log['mdm_time'] + log['refine_time']:.2f}s')}")
    
    if log['total_refinements'] > 0:
        print(f"\n  {Colors.cyan('Refinement statistics:')}")
        print(f"    • Total refinements: {Colors.bold(str(log['total_refinements']))}")
        if 'avg_confidence_improvement' in log:
            print(f"    • Average confidence gain: {Colors.bold(f'{log['avg_confidence_improvement']:.4f}')}")
        
        # Show example refinements
        if log['refined_tokens']:
            print(f"\n  {Colors.cyan('Example refinements:')}")
            for ref in log['refined_tokens'][:3]:
                print(f"    • {Colors.red(repr(ref['old']))} → {Colors.green(repr(ref['new']))} "
                      f"({Colors.yellow(f'+{ref['improvement']:.3f}')})")