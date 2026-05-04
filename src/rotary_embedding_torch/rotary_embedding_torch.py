from __future__ import annotations

from collections.abc import Callable
from math import pi
from types import EllipsisType
from typing import cast, Literal
from warnings import warn

import torch
from einops import rearrange, repeat
from torch import einsum, is_tensor, nn, tensor, Tensor
from torch.amp.autocast_mode import autocast
from torch.nn import Module

from torch_einops_kit import broadcast_cat as broadcast_cat, default, exists, slice_at_dim

from hunterMakesPy import raiseIfNone

def rotate_half(x: Tensor) -> Tensor:
    """Compute a quarter turn of each adjacent coordinate pair in `Tensor` `x`.

    You can use `rotate_half` inside `apply_rotary_emb` [1] when rotary position embeddings [2] need
    the 90°-rotated version of `x`. `rotate_half` interprets the last axis as adjacent coordinate
    pairs, such as `(x₀, x₁)`, `(x₂, x₃)`, and so on. For each coordinate pair, `rotate_half`
    returns `(-x₁, x₀)`. Despite the name `rotate_half`, `rotate_half` performs a quarter turn, not a
    half turn. The outer axes, shape, and `dtype` stay unchanged.

    Parameters
    ----------
    x : Tensor
        Input `Tensor` whose last axis has even length. `rotate_half` interprets each adjacent
        2-element slice of the last axis as one coordinate pair.

    Returns
    -------
    rotated : Tensor
        `Tensor` with the same shape and `dtype` as `x`. For each adjacent coordinate pair `(a, b)`
        in the last axis, `rotated` contains `(-b, a)`.

    See Also
    --------
    apply_rotary_emb : Combine cosine and sine rotary terms using `rotate_half`.

    Mathematics
    -----------
    quarter turn : equation
    ```
        Let  d ≜ `x.shape[-1]`,  y ≜ `rotated`

        R(π/2) ≜ [[0, −1], [1, 0]]
        (y₂ⱼ, y₂ⱼ₊₁) = R(π/2) · (x₂ⱼ, x₂ⱼ₊₁)   ∀ j ∈ {0, …, d/2 − 1}
    ```

    last-axis reshaping : transformation
    ```
        Let  m ≜ d / 2

        x ∈ ℝ^{…, 2m}
        x ↦ x̃ ∈ ℝ^{…, m, 2}
        (a, b) ↦ (−b, a)
        x̃ ↦ y ∈ ℝ^{…, 2m}
    ```

    PyTorch
    -------
    last-axis reshaping : transformation
        `rotate_half` reshapes the last axis from `(..., 2m)` to `(..., m, 2)`, swaps each coordinate
        pair from `(a, b)` to `(-b, a)`, and flattens the last two axes.

    References
    ----------
    [1] rotary_embedding_torch.apply_rotary_emb

    [2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://doi.org/10.48550/arXiv.2104.09864
    """
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: Tensor, t: Tensor, start_index: int = 0, scale: Tensor | float = 1.0, seq_dim: int = -2, freqs_seq_dim: int | None = None
) -> Tensor:
    """Apply `freqs` rotary angles to `Tensor` `t` at `start_index`.

    You can use `apply_rotary_emb` to apply the rotary angles in angle `Tensor` `freqs` to one
    feature block of input `Tensor` `t`. `apply_rotary_emb` combines cosine terms from `freqs` with
    sine terms built from `rotate_half` [1] to implement rotary position embeddings [2]. `scale` is
    usually `1.0`, but `scale` can also be a broadcastable `Tensor` when XPos length-extrapolation
    scaling is required [3].

    Parameters
    ----------
    freqs : Tensor
        Rotary-angle `Tensor` `freqs` whose last axis matches the number of rotated features.
    t : Tensor
        Input `Tensor` `t` that receives the rotary transformation.
    start_index : int = 0
        First feature index of the rotated block inside `Tensor` `t`.
    scale : Tensor | float = 1.0
        Broadcastable scale factor multiplied into the rotated block. XPos commonly supplies scale
        `Tensor` `scale` here [3].
    seq_dim : int = -2
        Axis of `Tensor` `t` that stores sequence position.
    freqs_seq_dim : int | None = None
        Axis of `Tensor` `freqs` that stores sequence position when `freqs` contains more positions
        than `t`. If `freqs_seq_dim` is `None` and suffix trimming is required, `apply_rotary_emb`
        uses axis `0`.

    Returns
    -------
    rotated : Tensor
        Output `Tensor` with the same shape and `dtype` as `t`.

    Raises
    ------
    ValueError
        Raised when `freqs.shape[-1]` exceeds `t.shape[-1]`.

    See Also
    --------
    rotate_half : Compute the quarter-turn companion used in the sine term.
    apply_learned_rotations : Expand learned rotary angles before applying them.
    RotaryEmbedding.forward : Generate the rotary angles commonly passed as `freqs`.

    Mathematics
    -----------
    rotated block : equation
    ```
        Let  Ω ≜ `freqs`,  s ≜ `scale`,  J ≜ `rotate_half()`
             n ≜ `start_index`,  m ≜ n + `freqs.shape[-1]`
             t = [tˡ ‖ tᵐ ‖ tʳ],  y ≜ `out`

        y = [tˡ ‖ tᵐ ⊙ cos Ω ⊙ s + J(tᵐ) ⊙ sin Ω ⊙ s ‖ tʳ]
    ```

    sequence trimming : equation
    ```
        Let  Ω₀ ≜ `freqs`,  L ≜ `t.shape[seq_dim]`
             σ ≜ `freqs_seq_dim`,  N ≜ |Ω₀|_σ

        N > L  ⟹  Ω = Ω₀_[N − L, N)
        N ≤ L  ⟹  Ω = Ω₀
    ```

    PyTorch
    -------
    autocast behavior : rule
        `apply_rotary_emb` runs with CUDA automatic mixed precision disabled, computes the rotation,
        and casts the result back to `t.dtype` before returning.

    References
    ----------
    [1] rotary_embedding_torch.rotate_half

    [2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://doi.org/10.48550/arXiv.2104.09864
    [3] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
        Chaudhary, V., Song, X., & Wei, F. (2022).
        A Length-Extrapolatable Transformer.
        https://doi.org/10.48550/arXiv.2212.10554
    """
    dtype: torch.dtype = t.dtype

    if freqs.ndim == 2 or t.ndim == 3:
        if not exists(freqs_seq_dim):
            freqs_seq_dim = 0
        seq_len: int = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim: int = freqs.shape[-1]
    end_index: int = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        message: str = f'I received `{rot_dim = }` rotation dimensions, but `t` only has `{t.shape[-1] = }` features. The rotation \
            dimension must not exceed the feature dimension.'
        raise ValueError(message)

    # NOTE Split t into three parts: left, middle (to be transformed), and right
    t_left: Tensor = t[..., :start_index]
    t_middle: Tensor = t[..., start_index:end_index]
    t_right: Tensor = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed: Tensor = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out: Tensor = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

def apply_learned_rotations(rotations: Tensor, t: Tensor, start_index: int = 0, freq_ranges: Tensor | None = None) -> Tensor:
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)

class RotaryEmbedding(Module):

    def __init__(
        self,
        dim: int,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: int | float = 10000,
        max_freq: int | float = 10,
        num_freqs: int = 1,
        *,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: int | float = 512,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = True,
        cache_max_seq_len: int = 8192,
    ) -> None:
        super().__init__()

        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        if dim > 2:
            theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for: Literal['lang', 'pixel', 'constant'] = freqs_for

        if exists(custom_freqs):
            freqs: Tensor = custom_freqs
        elif self.freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif self.freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            if self.freqs_for != "lang":
                message: str = f'I received `{freqs_for = }` and {custom_freqs = }, so I used the default value of `freqs_for`.'
                warn(message, RuntimeWarning, stacklevel=0)
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        self.cache_if_possible: bool = cache_if_possible
        self.cache_max_seq_len: int = cache_max_seq_len

        self.register_buffer("cached_freqs", torch.zeros(self.cache_max_seq_len, dim), persistent=False)
        self.cached_freqs_seq_len: int = 0

        self.learned_freq: bool = learned_freq

        self.freqs = nn.Parameter(freqs, requires_grad=self.learned_freq)

        # dummy for device

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        # default sequence dimension

        self.seq_before_head_dim: bool = seq_before_head_dim
        self.default_seq_dim: int = -3 if seq_before_head_dim else -2

        # interpolation factors

        if interpolate_factor < 1.0:
            message: str = f'I received `{interpolate_factor = }`, but `interpolate_factor` must be greater than or equal to 1.0.'
            raise ValueError(message)
        self.interpolate_factor: float = interpolate_factor

        # xpos

        self.use_xpos: bool = use_xpos

        if not self.use_xpos:
            return

        scale: Tensor = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

        self.scale_base: int | float = xpos_scale_base

        self.register_buffer("cached_scales", torch.zeros(self.cache_max_seq_len, dim), persistent=False)
        self.cached_scales_seq_len: int = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb: Callable[..., Tensor] = staticmethod(apply_rotary_emb)

    @property
    def device(self) -> torch.device:
        return cast(Tensor, self.dummy).device

    def get_seq_pos(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None, offset: int = 0) -> Tensor:
        device = default(device, self.device)
        dtype = default(dtype, cast(Tensor, self.cached_freqs).dtype)

        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t: Tensor, seq_dim: int | None = None, offset: int = 0, scale: Tensor | float | None = None) -> Tensor:
        seq_dim = default(seq_dim, self.default_seq_dim)

        if self.use_xpos and not exists(scale):
            message: str = 'I did not receive a value for `scale`, but `use_xpos` is enabled. Call `rotate_queries_and_keys` instead and \
                pass both queries and keys for length-extrapolatable rotary embeddings.'
            raise ValueError(message)

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq: Tensor = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs: Tensor = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q: Tensor, k: Tensor, seq_dim: int | None = None, offset: int = 0) -> tuple[Tensor, Tensor]:
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        if q_len > k_len:
            message: str = f'I received `{q_len = }` query positions and `{k_len = }` key positions, but the query length must not exceed \
                the key length.'
            raise ValueError(message)

        q_scale: Tensor | float = 1.0
        k_scale: Tensor | float = 1.0

        if self.use_xpos:
            seq: Tensor = self.get_seq_pos(k_len, dtype=dtype, device=device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q: Tensor = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k: Tensor = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q: Tensor, k: Tensor, seq_dim: int | None = None) -> tuple[Tensor, Tensor]:
        seq_dim = default(seq_dim, self.default_seq_dim)

        if not self.use_xpos:
            message: str = 'I observed `use_xpos = False`, but `rotate_queries_and_keys` requires `use_xpos = True`. Initialize \
                `RotaryEmbedding` with `use_xpos=True` to use this method.'
            raise ValueError(message)
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq: Tensor = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs: Tensor = self.forward(seq, seq_len=seq_len)
        scale: Tensor = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q: Tensor = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k: Tensor = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: int | None = None, offset: int = 0) -> Tensor:
        if not self.use_xpos:
            message: str = 'I observed `use_xpos = False`, but `get_scale` requires `use_xpos = True`. Initialize `RotaryEmbedding` with \
                `use_xpos=True` to use this method.'
            raise ValueError(message)

        should_cache: bool = self.cache_if_possible and exists(seq_len) and ((offset + seq_len) <= self.cache_max_seq_len)

        if should_cache and exists(self.cached_scales) and ((raiseIfNone(seq_len) + offset) <= self.cached_scales_seq_len):
            return cast(Tensor, self.cached_scales)[offset : (offset + raiseIfNone(seq_len))]

        power: Tensor = (t - len(t) // 2) / self.scale_base
        scale: Tensor = cast(Tensor, self.scale) ** rearrange(power, "n -> n 1")
        scale = repeat(scale, "n d -> n (d r)", r=2)

        if should_cache and offset == 0:
            cast(Tensor, self.cached_scales)[:seq_len] = scale.detach()
            self.cached_scales_seq_len = raiseIfNone(seq_len)

        return scale

    def get_axial_freqs(self, *dims: int, offsets: (tuple[int | float, ...] | Tensor | None) = None) -> Tensor:
        Colon: slice = slice(None)
        all_freqs: list[Tensor] = []

        # handle offset

        if exists(offsets):
            if not is_tensor(offsets):
                offsets = tensor(offsets)

            if len(offsets) != len(dims):
                message: str = f'I received `{len(offsets) = }` offsets and `{len(dims) = }` axis dimensions, \
                    but the number of offsets must equal the number of axes.'
                raise ValueError(message)

        # get frequencies for each axis

        for ind, dim in enumerate(dims):
            offset = 0
            if exists(offsets):
                offset = offsets[ind]

            if self.freqs_for == "pixel":
                pos: Tensor = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            pos = pos + offset

            freqs: Tensor = self.forward(pos, seq_len=dim)

            all_axis: list[slice | None] = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice: tuple[EllipsisType | slice | None, ...] = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        return broadcast_cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len: int | None = None, offset: int = 0) -> Tensor:
        should_cache: bool = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
            and (offset + seq_len) <= self.cache_max_seq_len
        )

        if should_cache and exists(self.cached_freqs) and (offset + raiseIfNone(seq_len)) <= self.cached_freqs_seq_len:
            return cast(Tensor, self.cached_freqs)[offset : (offset + raiseIfNone(seq_len))].detach()

        freqs = self.freqs

        freqs: Tensor = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache and offset == 0:
            cast(Tensor, self.cached_freqs)[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = raiseIfNone(seq_len)

        return freqs

"""
Some or all of the logic in this module may be protected by the following.

MIT License

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
