# diff_tome_packing_dynamic.py
# Speech-friendly ToMe (token merging) for SSL speech encoders, with:
# - local window matching (no long-range merges)
# - shared matching projection space (stabilize routing across layers)
# - importance-aware merge score
# - (option A) strict keep_ratio via Top-K merges (fixed budget)
# - (option B) dynamic budget: network decides which pairs to merge AND how many,
#              via hard-concrete gating + FLOPs/length penalty in loss
# - optional packing to truly shorten sequence (real attention speedup)

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers: local matching
# -----------------------------
def greedy_bipartite_match(sim: torch.Tensor, max_pairs: int) -> List[Tuple[int, int]]:
    """Greedy bipartite matching by similarity, conflict-free.

    sim: [nA, nB]
    return: list of (iA, iB)
    """
    nA, nB = sim.shape
    if max_pairs <= 0 or nA == 0 or nB == 0:
        return []

    flat = sim.flatten()
    _, idxs = torch.sort(flat, descending=True)

    usedA = torch.zeros(nA, dtype=torch.bool, device=sim.device)
    usedB = torch.zeros(nB, dtype=torch.bool, device=sim.device)

    pairs = []
    for idx in idxs:
        if len(pairs) >= max_pairs:
            break
        iA = int((idx // nB).item())
        iB = int((idx % nB).item())
        if (not usedA[iA]) and (not usedB[iB]):
            usedA[iA] = True
            usedB[iB] = True
            pairs.append((iA, iB))
    return pairs


def _valid_length_from_mask(attn_mask_1d: torch.Tensor) -> int:
    # attn_mask_1d: [T] bool
    if attn_mask_1d.numel() == 0:
        return 0
    return int(attn_mask_1d.long().sum().item())


def build_local_candidate_pairs(
    x: torch.Tensor,
    match_x: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    window_size: int,
    candidate_ratio: float,
    sim_threshold: float,
) -> List[Tuple[int, int, float]]:
    """Build candidate pairs within local windows (speech-friendly).

    - Only merge within local window
    - Similarity computed in shared matching space (match_x)
    - Optional similarity threshold to avoid merging across boundaries

    Return:
      candidates: list of (i, j, sim_ij) global indices with similarity score
    """
    T = x.shape[0]
    valid_T = T if attn_mask is None else _valid_length_from_mask(attn_mask)

    candidates: List[Tuple[int, int, float]] = []
    if valid_T < 2:
        return candidates

    for start in range(0, valid_T, window_size):
        end = min(start + window_size, valid_T)
        wlen = end - start
        if wlen < 2:
            continue

        idx = torch.arange(start, end, device=x.device)
        A_idx = idx[0::2]
        B_idx = idx[1::2]
        if B_idx.numel() == 0:
            continue

        A = match_x[A_idx]                 # [nA, Cm]
        B = match_x[B_idx]                 # [nB, Cm]
        sim = A @ B.transpose(0, 1)         # cosine if match_x normalized

        if sim_threshold is not None:
            sim = sim.masked_fill(sim < sim_threshold, float("-inf"))

        max_pairs = int(candidate_ratio * min(A_idx.numel(), B_idx.numel()))
        if max_pairs <= 0:
            continue

        local_pairs = greedy_bipartite_match(sim, max_pairs)
        for iA, iB in local_pairs:
            i = int(A_idx[iA].item())
            j = int(B_idx[iB].item())
            s = float(sim[iA, iB].item())
            if math.isfinite(s):
                candidates.append((i, j, s))

    return candidates


def topk_select_merges(scores: torch.Tensor, merges_target: int) -> torch.Tensor:
    """Strict budget selection: choose exactly top-K merges (hard 0/1 mask)."""
    P = scores.numel()
    K = max(0, min(int(merges_target), P))
    z = torch.zeros_like(scores)
    if K == 0:
        return z
    idx = torch.topk(scores, k=K, largest=True).indices
    z[idx] = 1.0
    return z


# -----------------------------
# Hard-Concrete gate for per-pair decisions
# -----------------------------
def hard_concrete_sample(
    log_alpha: torch.Tensor,
    temperature: float = 2/3,
    limit_l: float = -0.1,
    limit_r: float = 1.1,
    eps: float = 1e-6,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample differentiable z in [0,1] and its L0 probability (expected non-zero).

    This is the scalar Hard-Concrete used in L0 regularization (Louizos et al.).
    """
    if training:
        u = torch.rand_like(log_alpha)
        s = torch.sigmoid((torch.log(u + eps) - torch.log(1 - u + eps) + log_alpha) / temperature)
    else:
        s = torch.sigmoid(log_alpha)

    s_bar = s * (limit_r - limit_l) + limit_l
    z = torch.clamp(s_bar, min=0.0, max=1.0)

    # Handle edge case: if limit_l == 0, use a small epsilon to avoid log(0)
    if abs(limit_l) < 1e-8:
        limit_l_adj = -1e-6  # Small negative value to avoid log(0)
    else:
        limit_l_adj = limit_l
    
    if abs(limit_r) < 1e-8:
        limit_r_adj = 1e-6  # Small positive value
    else:
        limit_r_adj = limit_r
    
    temp_term = temperature * math.log(-limit_l_adj / limit_r_adj)
    l0_prob = torch.sigmoid(log_alpha - temp_term)  # P(z>0)
    return z, l0_prob


# -----------------------------
# Packing: real shorten
# -----------------------------
def pack_after_merge(
    x: torch.Tensor,                          # [T, C]
    pairs: List[Tuple[int, int]],
    z_hard: torch.Tensor,                     # [P] {0,1}
    merge_alpha: torch.Tensor,                # [P] in [0,1] (weight toward i vs j)
    attn_mask: Optional[torch.Tensor] = None  # [T] bool
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """Perform merges and pack into shorter sequence (keep time order).

    Convention: if merge, keep i and drop j.
    """
    T, C = x.shape
    valid_T = T if attn_mask is None else _valid_length_from_mask(attn_mask)

    drop = set()
    merged_content: Dict[int, torch.Tensor] = {}

    for p, (i, j) in enumerate(pairs):
        if z_hard[p].item() < 0.5:
            continue
        if j in drop:
            continue
        if i >= valid_T or j >= valid_T:
            continue
        drop.add(j)

        a = float(merge_alpha[p].item())
        xi = x[i]  # [C]
        xj = x[j]  # [C]
        merged = a * xi + (1.0 - a) * xj
        merged_content[i] = merged

    out_tokens = []
    orig2new: Dict[int, int] = {}
    new_idx = 0
    for t in range(valid_T):
        if t in drop:
            continue
        out_tokens.append(merged_content.get(t, x[t]))
        orig2new[t] = new_idx
        new_idx += 1

    out = torch.stack(out_tokens, dim=0) if len(out_tokens) > 0 else x[:0]
    out_mask = torch.ones(out.shape[0], device=x.device, dtype=torch.bool)
    return out, out_mask, orig2new


# -----------------------------
# Main module
# -----------------------------
class SpeechToMePackingBlock(nn.Module):
    """Speech-friendly ToMe merging (+ optional packing) with two modes:

    1) Fixed budget (keep_ratio): exactly Top-K merges.
    2) Dynamic budget (no keep_ratio): learn per-pair gates, and rely on an external
       FLOPs/length penalty to decide "merge how many".

    Practical training tip:
      - use dynamic_budget + pack_in_train=False (soft, fixed-length) to let task loss
        shape the gates (fully differentiable).
      - then switch pack_in_train=True for a short fine-tune to get real compute speedup.

    Args:
      keep_ratio: if not None => fixed budget (Top-K). If None => dynamic budget.
      pack_in_train: if False, training uses "soft merge without packing" (keeps length);
                    if True, training also packs (faster but gates become non-diff due to packing).
    """
    def __init__(
        self,
        dim: int,
        match_dim: int = 128,
        window_size: int = 16,
        candidate_ratio: float = 0.8,
        keep_ratio: Optional[float] = 0.7,
        sim_threshold: float = 0.0,
        score_sim_weight: float = 1.0,
        score_imp_weight: float = 0.5,
        alpha_scale: float = 5.0,
        # dynamic budget
        pack_in_train: bool = False,
        gate_temperature: float = 2/3,
        gate_limit_l: float = -0.1,
        gate_limit_r: float = 1.1,
        gate_mlp_hidden: int = 64,
        gate_init_bias: float = -2.0,  # negative => start with fewer merges
        l0_reg_weight: float = 0.0,    # optional: encourage fewer active gates
    ):
        super().__init__()
        self.dim = dim
        self.match_dim = match_dim

        self.window_size = window_size
        self.candidate_ratio = candidate_ratio
        self.keep_ratio = keep_ratio
        self.sim_threshold = sim_threshold

        self.score_sim_weight = score_sim_weight
        self.score_imp_weight = score_imp_weight
        self.alpha_scale = alpha_scale

        self.pack_in_train = pack_in_train
        self.gate_temperature = gate_temperature
        self.gate_limit_l = gate_limit_l
        self.gate_limit_r = gate_limit_r
        self.l0_reg_weight = l0_reg_weight

        # Shared matching projection (layer-wise stable routing space)
        self.match_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, match_dim, bias=False),
        )

        # Importance predictor (scalar per token): "don't merge away important frames"
        self.imp_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        # Gate predictor for dynamic budget (pair-wise)
        # Input features (per pair): [sim, imp_i, imp_j, imp_pair, rel_dist]
        self.gate_mlp = nn.Sequential(
            nn.Linear(5, gate_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gate_mlp_hidden, 1),
        )
        # a learnable bias to shift overall merge tendency
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_init_bias)))

    def _dynamic_gate(self, sim_ij: torch.Tensor, imp_i: torch.Tensor, imp_j: torch.Tensor,
                      i_idx: torch.Tensor, j_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return z_soft in [0,1], z_hard in {0,1}, and l0_prob for regularization."""
        imp_pair = 0.5 * (imp_i + imp_j)
        rel_dist = (j_idx - i_idx).to(sim_ij.dtype) / float(max(1, self.window_size))

        pair_feat = torch.stack([sim_ij, imp_i, imp_j, imp_pair, rel_dist], dim=-1)  # [P,5]
        log_alpha = self.gate_mlp(pair_feat).squeeze(-1) + self.gate_bias  # [P]

        z_soft, l0_prob = hard_concrete_sample(
            log_alpha,
            temperature=self.gate_temperature,
            limit_l=self.gate_limit_l,
            limit_r=self.gate_limit_r,
            training=self.training,
        )
        z_hard = (z_soft > 0.5).to(z_soft.dtype)
        return z_soft, z_hard, l0_prob

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, C]
        attn_mask: Optional[torch.Tensor] = None  # [B, T] bool, optional
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
          x_out:       [B, T', C] packed (or same-length if soft mode)
          new_mask_1d: [B, T'] bool valid tokens (or original if soft mode)
          info:        dict of stats (expected/hard lengths etc.).
                      If l0_reg_weight>0, info['reg_loss'] is a scalar tensor.
        """
        B, T, C = x.shape
        assert C == self.dim

        if attn_mask is not None:
            assert attn_mask.shape == (B, T)
            attn_mask = attn_mask.bool()

        match_x = F.normalize(self.match_proj(x), dim=-1)     # [B, T, match_dim]
        imp = torch.sigmoid(self.imp_net(x)).squeeze(-1)      # [B, T] in [0,1]

        packed_list = []
        mask_list = []
        kept_lens_hard = []
        kept_lens_exp = []
        merge_counts_hard = []
        merge_counts_exp = []
        l0_regs = []

        # Decide behavior in training
        do_pack = (not self.training) or self.pack_in_train

        for b in range(B):
            mb = None if attn_mask is None else attn_mask[b]
            valid_T = T if mb is None else _valid_length_from_mask(mb)
            xb = x[b]
            impb = imp[b]
            matchb = match_x[b]

            candidates = build_local_candidate_pairs(
                x=xb,
                match_x=matchb,
                attn_mask=mb,
                window_size=self.window_size,
                candidate_ratio=self.candidate_ratio,
                sim_threshold=self.sim_threshold,
            )

            if len(candidates) == 0 or valid_T < 2:
                out = xb[:valid_T] if do_pack else xb  # in soft mode keep full length
                out_mask = torch.ones(out.shape[0], device=x.device, dtype=torch.bool) if do_pack else (mb if mb is not None else torch.ones(T, device=x.device, dtype=torch.bool))
                packed_list.append(out)
                mask_list.append(out_mask)
                kept_lens_hard.append(int(out.shape[0] if do_pack else valid_T))
                kept_lens_exp.append(float(valid_T))
                merge_counts_hard.append(0)
                merge_counts_exp.append(0.0)
                continue

            i_idx = torch.tensor([c[0] for c in candidates], device=x.device, dtype=torch.long)
            j_idx = torch.tensor([c[1] for c in candidates], device=x.device, dtype=torch.long)
            sim_ij = torch.tensor([c[2] for c in candidates], device=x.device, dtype=x.dtype)  # [P]

            imp_i = impb[i_idx]
            imp_j = impb[j_idx]
            imp_pair = 0.5 * (imp_i + imp_j)

            scores = self.score_sim_weight * sim_ij + self.score_imp_weight * imp_pair  # [P]

            # Decide merges: fixed budget OR dynamic budget
            if self.keep_ratio is not None:
                T_target = int(math.ceil(self.keep_ratio * valid_T))
                merges_target = max(0, valid_T - T_target)
                merges_target = min(merges_target, len(candidates))
                z_hard = topk_select_merges(scores, merges_target)  # [P] {0,1}
                z_soft = z_hard.detach()  # no expectation in fixed mode
                l0_prob = z_soft.detach()
            else:
                z_soft, z_hard, l0_prob = self._dynamic_gate(sim_ij, imp_i, imp_j, i_idx, j_idx)

            # Merge alpha: choose how to mix i vs j (still deterministic from sim)
            alpha = torch.sigmoid(self.alpha_scale * sim_ij).clamp(0.05, 0.95)  # [P]

            # Expected stats (differentiable)
            exp_merges = float(z_soft.sum().detach().item())
            exp_len = float(valid_T) - float(z_soft.sum().detach().item())

            # Optional L0 regularizer (push fewer merges). Usually you'll use FLOPs penalty instead.
            if self.keep_ratio is None and self.l0_reg_weight > 0:
                l0_regs.append(l0_prob.mean())

            if do_pack:
                # Hard pack => real compute reduction (non-diff w.r.t. gate decisions)
                pairs_ij = [(int(i_idx[k].item()), int(j_idx[k].item())) for k in range(len(candidates))]
                out, out_mask, _ = pack_after_merge(
                    x=xb,
                    pairs=pairs_ij,
                    z_hard=z_hard,
                    merge_alpha=alpha,
                    attn_mask=mb
                )
                packed_list.append(out)
                mask_list.append(out_mask)
                kept_lens_hard.append(int(out.shape[0]))
            else:
                # Soft merge w/o packing (fully differentiable):
                # update kept i positions, and "suppress" j positions (but keep same length).
                out = xb.clone()
                # merged content for i
                merged = alpha.unsqueeze(-1) * xb[i_idx] + (1.0 - alpha).unsqueeze(-1) * xb[j_idx]
                out[i_idx] = z_soft.unsqueeze(-1) * merged + (1.0 - z_soft).unsqueeze(-1) * xb[i_idx]
                out[j_idx] = (1.0 - z_soft).unsqueeze(-1) * xb[j_idx]
                packed_list.append(out)
                mask_list.append(mb if mb is not None else torch.ones(T, device=x.device, dtype=torch.bool))
                kept_lens_hard.append(int(valid_T))  # still same length at runtime

            kept_lens_exp.append(exp_len)
            merge_counts_hard.append(int(z_hard.sum().item()))
            merge_counts_exp.append(float(exp_merges))

        # pad to batch max
        T_pad = max(t.shape[0] for t in packed_list) if len(packed_list) > 0 else 0
        x_out = x.new_zeros((B, T_pad, C))
        new_mask = torch.zeros((B, T_pad), device=x.device, dtype=torch.bool)
        for b in range(B):
            L = packed_list[b].shape[0]
            if L > 0:
                x_out[b, :L] = packed_list[b]
                new_mask[b, :L] = mask_list[b][:L]

        # L0 reg loss (optional)
        reg_loss = x.new_tensor(0.0)
        if len(l0_regs) > 0:
            reg_loss = self.l0_reg_weight * torch.stack(l0_regs).mean()

        info = {
            "reg_loss": reg_loss,
            "T_in": int(T),
            "T_pad": int(T_pad),
            "kept_lens_hard": kept_lens_hard,
            "kept_lens_exp": kept_lens_exp,
            "merge_counts_hard": merge_counts_hard,
            "merge_counts_exp": merge_counts_exp,
            "avg_keep_ratio_hard": float(sum(kept_lens_hard) / (B * max(1, T))),
            "avg_keep_ratio_exp": float(sum(kept_lens_exp) / (B * max(1, T))),
            "window_size": self.window_size,
            "candidate_ratio": self.candidate_ratio,
            "keep_ratio_target": self.keep_ratio,
            "dynamic_budget": (self.keep_ratio is None),
            "pack_in_train": self.pack_in_train,
        }
        return x_out, new_mask, info


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 101, 64
    x = torch.randn(B, T, C)

    # original valid-length mask (True=valid)
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    attn_mask[0, 90:] = False

    # Dynamic budget, soft mode in train (no packing), then hard pack in eval
    block = SpeechToMePackingBlock(
        dim=C,
        match_dim=64,
        window_size=16,
        candidate_ratio=0.8,
        keep_ratio=None,          # dynamic budget
        sim_threshold=0.0,
        pack_in_train=False,
        gate_init_bias=-2.0,
    )

    block.train()
    y_soft, m_soft, info = block(x, attn_mask=attn_mask)
    reg = info.get('reg_loss', None)
    print("[train/soft] y:", y_soft.shape, "mask sum:", m_soft.sum(dim=1).tolist(), "info:", {k: info[k] for k in ["avg_keep_ratio_exp","avg_keep_ratio_hard","dynamic_budget"]})

    block.eval()
    y_hard, m_hard, info = block(x, attn_mask=attn_mask)
    reg = info.get('reg_loss', None)
    print("[eval/pack]  y:", y_hard.shape, "mask sum:", m_hard.sum(dim=1).tolist(), "info:", {k: info[k] for k in ["avg_keep_ratio_hard","dynamic_budget"]})
