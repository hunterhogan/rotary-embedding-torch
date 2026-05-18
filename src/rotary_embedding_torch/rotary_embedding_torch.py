# pyright: reportUntypedFunctionDecorator=false
# ruff: noqa D100
from __future__ import annotations

from collections.abc import Callable
from einops import rearrange, repeat
from hunterMakesPy import raiseIfNone
from math import pi
from torch import einsum, is_tensor, nn, tensor, Tensor
from torch.amp.autocast_mode import autocast
from torch.nn import Module
from torch_einops_kit import broadcast_cat as broadcast_cat, default, exists, slice_at_dim
from types import EllipsisType
from typing import cast, Literal
from warnings import warn
import torch

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
    """Apply learned `rotations` to `Tensor` `t` at `start_index`.

    (AI generated docstring)

    You can use `apply_learned_rotations` when learned-angle `Tensor` `rotations` stores one scalar
    per coordinate pair instead of the repeated layout expected by `apply_rotary_emb` [1]. If
    frequency-range `Tensor` `freq_ranges` is present, `apply_learned_rotations` expands each learned
    angle across `freq_ranges` before repeating each angle across both elements of each coordinate
    pair. The resulting repeated layout matches the rotary coordinate structure used in rotary
    position embeddings [2].

    Parameters
    ----------
    rotations : Tensor
        Learned-angle `Tensor` `rotations` before pairwise repetition.
    t : Tensor
        Input `Tensor` `t` that receives the rotary transformation.
    start_index : int = 0
        First feature index of the rotated block inside `Tensor` `t`.
    freq_ranges : Tensor | None = None
        Optional frequency-range `Tensor` `freq_ranges` multiplied into `rotations` before pairwise
        repetition.

    Returns
    -------
    rotated : Tensor
        Output `Tensor` with the same shape and `dtype` as `t`.

    See Also
    --------
    apply_rotary_emb : Apply repeated rotary angles to a feature block.

    Mathematics
    -----------
    phase expansion : equation
    ```
        Let  ρ ≜ `rotations`,  γ ≜ `freq_ranges`,  Ω ≜ repeated angles
             y ≜ `rotated`

        γ ≠ `None`  ⟹  ω = vec(ρ ⊗ γ)
        γ = `None`  ⟹  ω = ρ

        Ω = repeat₂(ω)
        y = `apply_rotary_emb(Ω, t, start_index = start_index)`
    ```

    PyTorch
    -------
    last-axis repetition : rule
        `apply_learned_rotations` expands `rotations` with `einsum` when `freq_ranges` is present and
        repeats the last axis with `repeat(..., r=2)` so that each learned angle matches one
        coordinate pair in `t`.

    References
    ----------
    [1] rotary_embedding_torch.apply_rotary_emb

    [2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://doi.org/10.48550/arXiv.2104.09864
    """
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)

class RotaryEmbedding(Module):
    """Generate rotary-angle tensors and optional XPos scales for attention tensors.

    (AI generated docstring)

    You can use `RotaryEmbedding` to turn position numbers into rotary-angle `Tensor` values for
    query `Tensor` `q`, key `Tensor` `k`, or any other feature `Tensor` `t` used by rotary position
    embeddings [1]. `RotaryEmbedding` also supports axial rotary embeddings, XPos scaling
    (position-dependent query and key rescaling for longer sequences) [2], position interpolation
    (compressing position numbers during longer-context fine-tuning) [3], and NTK-aware rescaling (a
    neural-tangent-kernel-inspired adjustment of `theta` for longer contexts) [4].

    Attributes
    ----------
    freqs_for : Literal['lang', 'pixel', 'constant']
        Frequency-family selector that determines how parameter `freqs` is initialized.
    cache_if_possible : bool
        Whether reusable phase and scale tensors may be stored in internal caches.
    cache_max_seq_len : int
        Maximum cached sequence length for buffer `cached_freqs` and buffer `cached_scales`.
    cached_freqs : Tensor
        Non-persistent buffer that stores previously generated rotary-angle tensors.
    cached_freqs_seq_len : int
        Length of the valid prefix stored in `cached_freqs`.
    learned_freq : bool
        Whether parameter `freqs` participates in gradient updates.
    freqs : nn.Parameter
        Base-frequency parameter used to generate rotary-angle tensors.
    dummy : Tensor
        Zero-valued buffer used to expose the current module device.
    seq_before_head_dim : bool
        Whether the sequence axis precedes the head axis in attention tensors.
    default_seq_dim : int
        Default sequence axis derived from `seq_before_head_dim`.
    interpolate_factor : float
        Factor that compresses position numbers before angle generation.
    use_xpos : bool
        Whether XPos scaling is enabled.
    scale : Tensor
        Per-coordinate-pair XPos base values. Present only when `use_xpos = True`.
    scale_base : int | float
        Denominator used in the XPos exponent. Present only when `use_xpos = True`.
    cached_scales : Tensor
        Non-persistent buffer that stores previously generated XPos scale tensors. Present only when
        `use_xpos = True`.
    cached_scales_seq_len : int
        Length of the valid prefix stored in `cached_scales`. Present only when `use_xpos = True`.
    apply_rotary_emb : Callable[..., Tensor]
        Convenience attribute that exposes `rotary_embedding_torch.apply_rotary_emb`. Present only
        when `use_xpos = True`.
    device : torch.device
        Device reported by registered buffer `dummy`.

    Mathematics
    -----------
    frequency family : equation
    ```
        Let  ω ≜ `freqs`,  d ≜ rotary dimension

        `freqs_for` = 'lang'     ⟹  ωⱼ = θ^(−2j / d)
        `freqs_for` = 'pixel'    ⟹  ωⱼ = π · linspace(1, max_freq / 2, d / 2)ⱼ
        `freqs_for` = 'constant' ⟹  ωⱼ = 1
    ```

    position map : equation
    ```
        Let  p ≜ position values returned by `get_seq_pos`

        pₘ = (m + offset) / `interpolate_factor`
    ```

    xpos scale : equation
    ```
        Let  b ≜ `scale`,  s ≜ scale values returned by `get_scale`
             L ≜ sequence length used by `get_scale`

        bⱼ = (2j + 0.4d) / (1.4d)
        sₘ,2ⱼ = sₘ,2ⱼ₊₁ = bⱼ^((m − ⌊L / 2⌋) / `scale_base`)
    ```

    rescaled base : equation
    ```
        Let  θ₀ ≜ input `theta`,  θ ≜ stored base

        d > 2  ⟹  θ = θ₀ · `theta_rescale_factor`^(d / (d − 2))
    ```

    PyTorch
    -------
    stored state : rule
        `RotaryEmbedding` stores base frequencies in parameter `freqs`, stores reusable angle tables
        in non-persistent buffer `cached_freqs`, and stores reusable XPos scale tables in
        `cached_scales` when `use_xpos = True`.

    See Also
    --------
    apply_rotary_emb : Apply rotary angles to a feature block.
    rotate_queries_or_keys : Rotate one query or key tensor.
    get_axial_freqs : Build broadcastable rotary angles for multiple axes.

    References
    ----------
    [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://doi.org/10.48550/arXiv.2104.09864
    [2] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
        Chaudhary, V., Song, X., & Wei, F. (2022).
        A Length-Extrapolatable Transformer.
        https://doi.org/10.48550/arXiv.2212.10554
    [3] Chen, S., Wong, S., Chen, L., & Tian, Y. (2023).
        Extending Context Window of Large Language Models via Position Interpolation.
        https://doi.org/10.48550/arXiv.2306.15595
    [4] bloc97. (2023). NTK-Aware Scaled RoPE allows LLaMA models to have
        extended (8k+) context size without any fine-tuning and minimal perplexity degradation.
        https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    """

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
        """Configure `RotaryEmbedding` for rotary dimension `dim` and optional XPos scaling.

        (AI generated docstring)

        You can use `__init__` to choose how `RotaryEmbedding` builds the base frequencies used by
        rotary position embeddings [1], the internal cache tables, and the optional XPos scale
        tables. `interpolate_factor` controls position interpolation [3], and `theta_rescale_factor`
        controls NTK-aware rescaling (a neural-tangent-kernel-inspired adjustment of `theta` for
        longer contexts) [4].

        Parameters
        ----------
        dim : int
            Feature count assigned to rotary rotation. `dim` is usually even because rotary rotation
            consumes adjacent coordinate pairs.
        custom_freqs : Tensor | None = None
            Optional custom base-frequency `Tensor` `custom_freqs`. If `custom_freqs` is present,
            `RotaryEmbedding` ignores `freqs_for`, `theta`, `max_freq`, and `num_freqs`.
        freqs_for : Literal['lang', 'pixel', 'constant'] = 'lang'
            Named frequency family used when `custom_freqs` is `None`. `'lang'` builds inverse powers
            of `theta`, `'pixel'` uses evenly spaced pixel frequencies, and `'constant'` uses ones.
        theta : int | float = 10000
            Base used for language-style inverse frequencies before optional rescaling.
        max_freq : int | float = 10
            Upper pixel frequency used when `freqs_for = 'pixel'`.
        num_freqs : int = 1
            Number of constant frequencies used when `freqs_for = 'constant'`.
        learned_freq : bool = False
            Whether parameter `freqs` should be optimized during training.
        use_xpos : bool = False
            Whether to enable XPos scaling for longer-sequence extrapolation [2].
        xpos_scale_base : int | float = 512
            Denominator used in the XPos exponent when `use_xpos = True`.
        interpolate_factor : float = 1.0
            Factor that compresses position numbers before angle generation. Values greater than
            `1.0` implement position interpolation [3].
        theta_rescale_factor : float = 1.0
            Factor used in the NTK-aware update θ ← θ · `theta_rescale_factor`^(dim / (dim − 2)) when
            `dim > 2` [4].
        seq_before_head_dim : bool = False
            Whether the default sequence axis is `-3` instead of `-2`.
        cache_if_possible : bool = True
            Whether angle and scale computations may be stored in internal caches.
        cache_max_seq_len : int = 8192
            Maximum cached sequence length for internal angle and scale buffers.

        Raises
        ------
        ValueError
            Raised when `interpolate_factor < 1.0`.

        Mathematics
        -----------
        frequency family : equation
        ```
            Let  d ≜ `dim`,  ω ≜ `freqs`

            `freqs_for` = 'lang'     ⟹  ωⱼ = θ^(−2j / d)
            `freqs_for` = 'pixel'    ⟹  ωⱼ = π · linspace(1, max_freq / 2, d / 2)ⱼ
            `freqs_for` = 'constant' ⟹  ωⱼ = 1
        ```

        position interpolation : equation
        ```
            Let  p ≜ positions later returned by `get_seq_pos`

            pₘ = (m + offset) / `interpolate_factor`
        ```

        xpos base : equation
        ```
            Let  b ≜ `scale`

            `use_xpos` = True  ⟹  bⱼ = (2j + 0.4 · d) / (1.4 · d)
        ```

        rescaled base : equation
        ```
            Let  θ₀ ≜ input `theta`,  θ ≜ stored base

            d > 2  ⟹  θ = θ₀ · `theta_rescale_factor`^(d / (d − 2))
        ```

        xpos scale : equation
        ```
            Let  L ≜ sequence length used by `get_scale`,  s ≜ XPos scale values

            sₘ,2ⱼ = sₘ,2ⱼ₊₁ = bⱼ^((m − ⌊L / 2⌋) / `xpos_scale_base`)
        ```

        PyTorch
        -------
        common state : rule
            `__init__` always creates parameter `freqs`, buffer `cached_freqs`, buffer `dummy`, and
            the cache-length attributes that track reusable angle tables.
        xpos state : rule
            If `use_xpos = True`, `__init__` also creates buffer `scale`, buffer `cached_scales`, and
            convenience attribute `apply_rotary_emb`.

        References
        ----------
        [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
            RoFormer: Enhanced Transformer with Rotary Position Embedding.
            https://doi.org/10.48550/arXiv.2104.09864
        [2] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
            Chaudhary, V., Song, X., & Wei, F. (2022).
            A Length-Extrapolatable Transformer.
            https://doi.org/10.48550/arXiv.2212.10554
        [3] Chen, S., Wong, S., Chen, L., & Tian, Y. (2023).
            Extending Context Window of Large Language Models via Position Interpolation.
            https://doi.org/10.48550/arXiv.2306.15595
        [4] bloc97. (2023). NTK-Aware Scaled RoPE allows LLaMA models to have
            extended (8k+) context size without any fine-tuning and minimal perplexity degradation.
            https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        """
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
        """Expose the device of registered buffer `dummy`.

        (AI generated docstring)

        You can use `device` when new `Tensor` values should follow the same device as
        `RotaryEmbedding`. `device` reads registered buffer `dummy`, so `device` changes
        automatically when the module moves between CPU and GPU.

        Returns
        -------
        moduleDevice : torch.device
            Device tracked by registered buffer `dummy`.

        PyTorch
        -------
        tracked buffer : Tensor
            `device` returns `cast(Tensor, self.dummy).device`, so `device` follows the placement of
            buffer `dummy`.
        """
        return cast(Tensor, self.dummy).device

    def get_seq_pos(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None, offset: int = 0) -> Tensor:
        """Generate `seq_len` position values with `device`, `dtype`, and `offset`.

        (AI generated docstring)

        You can use `get_seq_pos` to build position `Tensor` `seqPos` for query `Tensor` `q`, key
        `Tensor` `k`, or any other feature `Tensor` `t`. `get_seq_pos` produces the position tensor
        consumed by `forward` [1] and divides the position values by `interpolate_factor`, so
        longer-context fine-tuning can reuse a shorter training range [2].

        Parameters
        ----------
        seq_len : int
            Number of position values to generate.
        device : torch.device | None = None
            Device for returned `Tensor` `seqPos`. If `device` is `None`, `get_seq_pos` uses
            `RotaryEmbedding.device`.
        dtype : torch.dtype | None = None
            `dtype` for returned `Tensor` `seqPos`. If `dtype` is `None`, `get_seq_pos` uses the
            cached frequency `dtype`.
        offset : int = 0
            Additive offset applied before interpolation.

        Returns
        -------
        seqPos : Tensor
            One-dimensional position `Tensor` `seqPos` of length `seq_len`.

        See Also
        --------
        forward : Convert `seqPos` to rotary angles.

        Mathematics
        -----------
        position sequence : equation
        ```
            Let  p ≜ `seqPos`

            pₘ = (m + `offset`) / `interpolate_factor`   ∀ m ∈ {0, …, `seq_len` − 1}
        ```

        PyTorch
        -------
        default device and dtype : rule
            If `device` is `None`, `get_seq_pos` uses `RotaryEmbedding.device`. If `dtype` is `None`,
            `get_seq_pos` uses `cached_freqs.dtype`.

        References
        ----------
        [1] rotary_embedding_torch.RotaryEmbedding.forward

        [2] Chen, S., Wong, S., Chen, L., & Tian, Y. (2023).
            Extending Context Window of Large Language Models via Position Interpolation.
            https://doi.org/10.48550/arXiv.2306.15595
        """
        device = default(device, self.device)
        dtype = default(dtype, cast(Tensor, self.cached_freqs).dtype)

        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t: Tensor, seq_dim: int | None = None, offset: int = 0, scale: Tensor | float | None = None) -> Tensor:
        """Rotate `Tensor` `t` along `seq_dim` with `offset` and optional `scale`.

        (AI generated docstring)

        You can use `rotate_queries_or_keys` when query `Tensor` `q`, key `Tensor` `k`, or any other
        attention `Tensor` `t` needs rotary position embeddings [5] and no paired tensor is
        available. `rotate_queries_or_keys` builds position values with `get_seq_pos` [1], converts
        the position values to rotary angles with `forward` [2], and applies the angles with
        `apply_rotary_emb` [3]. When `use_xpos = True`, scale `Tensor` `scale` should usually come
        from `get_scale`, which supplies the length-extrapolation scaling used by XPos [4, 6].

        Parameters
        ----------
        t : Tensor
            Attention `Tensor` `t`. `Tensor` `t` can represent query `Tensor` `q`, key `Tensor` `k`,
            or another tensor that follows the same rotary convention.
        seq_dim : int | None = None
            Sequence axis of `Tensor` `t`. If `seq_dim` is `None`, `rotate_queries_or_keys` uses
            `default_seq_dim`.
        offset : int = 0
            Additive position offset applied before angle generation.
        scale : Tensor | float | None = None
            Optional broadcastable scale factor. If `use_xpos = True`, `scale` should usually be the
            result of `get_scale` [4].

        Returns
        -------
        rotated : Tensor
            Rotated `Tensor` with the same shape and `dtype` as `t`.

        Raises
        ------
        ValueError
            Raised when `use_xpos = True` and `scale` is `None`.

        See Also
        --------
        rotate_queries_and_keys : Rotate query tensor `q` and key tensor `k` together with XPos.
        rotate_queries_with_cached_keys : Rotate query tensor `q` against a longer cached key tensor `k`.

        Mathematics
        -----------
        phase application : equation
        ```
            Let  x ≜ `t`,  p ≜ `get_seq_pos(seq_len, offset = offset)`
                 Ω ≜ `forward(p)`,  y ≜ `rotated`,  J ≜ `rotate_half`

            `scale` = `None`  ⟹  s = 1
            `scale` ≠ `None`  ⟹  s = `scale`

            yₘ = xₘ ⊙ cos Ωₘ ⊙ sₘ + J(xₘ) ⊙ sin Ωₘ ⊙ sₘ
        ```

        head-axis broadcasting : transformation
        ```
            `seq_dim` = −3  ⟹  Ω ∈ ℝ^{n×d} ↦ Ω̃ ∈ ℝ^{n×1×d}
        ```

        PyTorch
        -------
        sequence axis : rule
            If `seq_dim == -3`, `rotate_queries_or_keys` inserts a singleton head axis into the
            generated angle tensor so that the angle tensor broadcasts over the head axis of `t`.

        References
        ----------
        [1] rotary_embedding_torch.RotaryEmbedding.get_seq_pos

        [2] rotary_embedding_torch.RotaryEmbedding.forward

        [3] rotary_embedding_torch.apply_rotary_emb

        [4] rotary_embedding_torch.RotaryEmbedding.get_scale

        [5] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
            RoFormer: Enhanced Transformer with Rotary Position Embedding.
            https://doi.org/10.48550/arXiv.2104.09864
        [6] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
            Chaudhary, V., Song, X., & Wei, F. (2022).
            A Length-Extrapolatable Transformer.
            https://doi.org/10.48550/arXiv.2212.10554
        """
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
        """Rotate query `Tensor` `q` against cached key `Tensor` `k` with `offset`.

        (AI generated docstring)

        You can use `rotate_queries_with_cached_keys` during autoregressive decoding, when query
        `Tensor` `q` contains only the newest positions but key `Tensor` `k` already contains earlier
        cached positions. `rotate_queries_with_cached_keys` shifts the rotary positions of `q` so
        that query `Tensor` `q` lines up with the tail of key `Tensor` `k` [1, 2]. If `use_xpos =
        True`, `rotate_queries_with_cached_keys` also builds the matching scale tensors with
        `get_scale` [3, 4].

        Parameters
        ----------
        q : Tensor
            Query `Tensor` `q` to rotate.
        k : Tensor
            Key `Tensor` `k` to rotate.
        seq_dim : int | None = None
            Sequence axis shared by `q` and `k`. If `seq_dim` is `None`,
            `rotate_queries_with_cached_keys` uses `default_seq_dim`.
        offset : int = 0
            Additional offset added to the derived query positions.

        Returns
        -------
        rotatedQueryAndKey : tuple[Tensor, Tensor]
            Tuple `(rotated_q, rotated_k)` with the same shapes and `dtype` values as `q` and `k`.

        Raises
        ------
        ValueError
            Raised when the query length exceeds the key length along `seq_dim`.

        See Also
        --------
        rotate_queries_or_keys : Rotate one attention tensor with an explicit position offset.
        rotate_queries_and_keys : Rotate query tensor `q` and key tensor `k` when both have the same
            sequence length.

        Mathematics
        -----------
        position alignment : equation
        ```
            Let  ℓ_q ≜ `q.shape[seq_dim]`,  ℓ_k ≜ `k.shape[seq_dim]`
                 Δ ≜ ℓ_k − ℓ_q + `offset`
                 q′ ≜ `rotated_q`,  k′ ≜ `rotated_k`

            p^(q) = {Δ, …, Δ + ℓ_q − 1}
            p^(k) = {0, …, ℓ_k − 1}
        ```

        xpos scaling : equation
        ```
            Let  s ≜ `get_scale(get_seq_pos(ℓ_k))`

            s^(q) = s_[|s| − ℓ_q, |s|)
            s^(k) = s^(−1)
        ```

        PyTorch
        -------
        cache alignment : rule
            `rotate_queries_with_cached_keys` is the convenience method for key-value cache decoding,
            because `rotate_queries_with_cached_keys` derives the query offset from the current
            lengths of `q` and `k`.

        References
        ----------
        [1] rotary_embedding_torch.RotaryEmbedding.rotate_queries_or_keys

        [2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
            RoFormer: Enhanced Transformer with Rotary Position Embedding.
            https://doi.org/10.48550/arXiv.2104.09864
        [3] rotary_embedding_torch.RotaryEmbedding.get_scale

        [4] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
            Chaudhary, V., Song, X., & Wei, F. (2022).
            A Length-Extrapolatable Transformer.
            https://doi.org/10.48550/arXiv.2212.10554
        """
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
        """Rotate query `Tensor` `q` and key `Tensor` `k` along `seq_dim`.

        (AI generated docstring)

        You can use `rotate_queries_and_keys` when query `Tensor` `q` and key `Tensor` `k` share the
        same sequence length and `RotaryEmbedding` should build the XPos scale tensors automatically
        [1]. `rotate_queries_and_keys` generates one shared angle tensor with `forward` [2] and
        applies inverse scaling on the key side with `apply_rotary_emb` [3].

        Parameters
        ----------
        q : Tensor
            Query `Tensor` `q` to rotate.
        k : Tensor
            Key `Tensor` `k` to rotate.
        seq_dim : int | None = None
            Sequence axis shared by `q` and `k`. If `seq_dim` is `None`, `rotate_queries_and_keys`
            uses `default_seq_dim`.

        Returns
        -------
        rotatedQueryAndKey : tuple[Tensor, Tensor]
            Tuple `(rotated_q, rotated_k)` with the same shapes and `dtype` values as `q` and `k`.

        Raises
        ------
        ValueError
            Raised when `use_xpos = False`.

        See Also
        --------
        get_scale : Compute the XPos scale tensor used by `rotate_queries_and_keys`.
        rotate_queries_or_keys : Rotate one attention tensor when scaling is supplied explicitly.

        Mathematics
        -----------
        shared angles : equation
        ```
            Let  p ≜ `get_seq_pos(seq_len)`,  Ω ≜ `forward(p)`
                 s ≜ `get_scale(p)`,  q′ ≜ `rotated_q`,  k′ ≜ `rotated_k`

            q′ₘ = R(Ωₘ) qₘ ⊙ sₘ
            k′ₘ = R(Ωₘ) kₘ ⊙ sₘ^(−1)
        ```

        head-axis broadcasting : transformation
        ```
            `seq_dim` = −3  ⟹  Ω ∈ ℝ^{n×d} ↦ Ω̃ ∈ ℝ^{n×1×d}
            `seq_dim` = −3  ⟹  s ∈ ℝ^{n×d} ↦ s̃ ∈ ℝ^{n×1×d}
        ```

        PyTorch
        -------
        sequence axis : rule
            If `seq_dim == -3`, `rotate_queries_and_keys` inserts a singleton head axis into both the
            angle tensor and the scale tensor so that both tensors broadcast over the head axis.

        References
        ----------
        [1] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
            Chaudhary, V., Song, X., & Wei, F. (2022).
            A Length-Extrapolatable Transformer.
            https://doi.org/10.48550/arXiv.2212.10554
        [2] rotary_embedding_torch.RotaryEmbedding.forward

        [3] rotary_embedding_torch.apply_rotary_emb
        """
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
        """Compute XPos scale values from position `Tensor` `t`.

        (AI generated docstring)

        You can use `get_scale` to build the XPos scale `Tensor` applied to query `Tensor` `q` and
        the inverse XPos scale `Tensor` applied to key `Tensor` `k` [1]. The formula used by
        `get_scale` follows the public `torchscale` XPos implementation [2]. `get_scale` also reuses
        cached values when `seq_len` and `offset` describe a previously stored window. `offset`
        affects cache lookup only. `offset` does not change the numeric values inside position
        `Tensor` `t`.

        Parameters
        ----------
        t : Tensor
            Position `Tensor` `t` whose values determine the XPos exponents.
        seq_len : int | None = None
            Known sequence length used for cache lookup and cache writes.
        offset : int = 0
            Cache offset for the requested scale window. `offset` does not modify `t`.

        Returns
        -------
        scale : Tensor
            XPos scale `Tensor` `scale` whose last axis matches the paired rotary layout.

        Raises
        ------
        ValueError
            Raised when `use_xpos = False`.

        See Also
        --------
        rotate_queries_and_keys : Apply `scale` to query tensor `q` and inverse `scale` to key tensor `k`.

        Mathematics
        -----------
        xpos scale : equation
        ```
            Let  B ≜ `self.scale`,  y ≜ `scale`

            powerₘ = (tₘ − ⌊|t| / 2⌋) / `scale_base`
            yₘ,2ⱼ = yₘ,2ⱼ₊₁ = Bⱼ^powerₘ
        ```

        PyTorch
        -------
        cache rule : bool
            Caching is enabled when `cache_if_possible` is `True`, `seq_len` is known, and
            `offset + seq_len ≤ cache_max_seq_len`.

        References
        ----------
        [1] Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,
            Chaudhary, V., Song, X., & Wei, F. (2022).
            A Length-Extrapolatable Transformer.
            https://doi.org/10.48550/arXiv.2212.10554
        [2] sunyt32/torchscale `torchscale/component/xpos_relative_position.py`.
            https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
        """
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
        """Generate axial rotary angles for axis lengths `dims` and optional `offsets`.

        (AI generated docstring)

        You can use `get_axial_freqs` when feature `Tensor` `t` has more than one position axis, such
        as frame, height, and width. `get_axial_freqs` builds one rotary-angle field per axis with
        `forward` [1] and concatenates the fields into one broadcastable result that
        `apply_rotary_emb` [2] can consume for axial rotary embeddings [3]. `offsets` lets each axis
        start from a different position.

        Parameters
        ----------
        *dims : int
            Axis lengths for which `get_axial_freqs` should generate rotary angles.
        offsets : tuple[int | float, ...] | Tensor | None = None
            Optional per-axis offsets. If `offsets` is present, `len(offsets)` must equal `len(dims)`.

        Returns
        -------
        axialFreqs : Tensor
            Broadcastable angle `Tensor` `axialFreqs` whose last axis concatenates the per-axis fields.

        Raises
        ------
        ValueError
            Raised when `offsets` is present and `len(offsets) != len(dims)`.

        See Also
        --------
        forward : Generate the one-axis rotary angles used inside `get_axial_freqs`.
        apply_rotary_emb : Apply `axialFreqs` to query tensor `q` or key tensor `k`.

        Mathematics
        -----------
        per-axis positions : equation
        ```
            Let  d⁽ⁱ⁾ ≜ `dims[i]`,  δ⁽ⁱ⁾ ≜ `offsets[i]`,  F⁽ⁱ⁾ ≜ axis-angle field i

            `freqs_for` = 'pixel'    ⟹  p⁽ⁱ⁾ = linspace(−1, 1, d⁽ⁱ⁾) + δ⁽ⁱ⁾
            `freqs_for` ≠ 'pixel'    ⟹  p⁽ⁱ⁾ = arange(d⁽ⁱ⁾) + δ⁽ⁱ⁾

            F⁽ⁱ⁾ = `forward(p⁽ⁱ⁾, seq_len = d⁽ⁱ⁾)`
        ```

        separable composition : equation
        ```
            Let  A ≜ `axialFreqs`

            F⁽ⁱ⁾ ∈ ℝ^{d⁽ⁱ⁾×r}  ↦  F̃⁽ⁱ⁾ ∈ ℝ^{1×…×d⁽ⁱ⁾×…×1×r}
            A = broadcast_cat(F̃⁽¹⁾, …, F̃⁽ⁿ⁾)
        ```

        PyTorch
        -------
        singleton axes : rule
            `get_axial_freqs` inserts singleton axes around each per-axis angle field so that the
            angle fields broadcast together before concatenation.

        References
        ----------
        [1] rotary_embedding_torch.RotaryEmbedding.forward

        [2] rotary_embedding_torch.apply_rotary_emb

        [3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
            RoFormer: Enhanced Transformer with Rotary Position Embedding.
            https://doi.org/10.48550/arXiv.2104.09864
        """
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
        """Convert position `Tensor` `t` to rotary angles with cache window `offset`.

        (AI generated docstring)

        You can use `forward` to convert position `Tensor` `t` into rotary-angle `Tensor` `freqs` in
        the repeated layout expected by `apply_rotary_emb` [1]. `forward` also serves cached values
        when `seq_len` and `offset` describe a window that `RotaryEmbedding` has already computed.
        `offset` affects cache lookup only. `offset` does not change the numeric values inside
        `Tensor` `t`.

        Parameters
        ----------
        t : Tensor
            Position `Tensor` `t` used to generate rotary angles.
        seq_len : int | None = None
            Known sequence length used for cache lookup and cache writes.
        offset : int = 0
            Cache offset for the requested angle window. `offset` does not modify `t`.

        Returns
        -------
        freqs : Tensor
            Rotary-angle `Tensor` `freqs` whose last axis is repeated across coordinate pairs.

        See Also
        --------
        apply_rotary_emb : Apply `freqs` to a feature tensor.
        get_seq_pos : Generate the position tensor commonly passed to `forward`.

        Mathematics
        -----------
        phase construction : equation
        ```
            Let  ω ≜ `self.freqs`,  Ω ≜ `freqs`

            Ωₘ,ⱼ = tₘ ωⱼ
            Ωₘ,2ⱼ = Ωₘ,2ⱼ₊₁ = tₘ ωⱼ
        ```

        cache slice : equation
        ```
            Let  Ω̂ ≜ `cached_freqs`,  L ≜ `seq_len`,  o ≜ `offset`

            cached  ⟹  Ω = Ω̂_[o, o + L)
        ```

        PyTorch
        -------
        autocast behavior : rule
            `forward` runs with CUDA automatic mixed precision disabled so that `forward` computes
            the outer product in `self.freqs.dtype` and returns cached values without autocast
            changes.
        cache rule : bool
            Caching is enabled only when `cache_if_possible` is `True`, `learned_freq = False`,
            `seq_len` is known, `freqs_for != 'pixel'`, and the requested window fits inside
            `cache_max_seq_len`.

        References
        ----------
        [1] rotary_embedding_torch.apply_rotary_emb

        [2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
            RoFormer: Enhanced Transformer with Rotary Position Embedding.
            https://doi.org/10.48550/arXiv.2104.09864
        """
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
