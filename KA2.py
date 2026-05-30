```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    KAN Linear layer with B-spline basis functions.

    This implementation follows the paper description:
    - base linear projection
    - spline basis evaluation
    - spline coefficient projection
    - gradient-free grid update using previous grid + adaptive grid momentum update
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 3,
        scale_noise: float = 0.01,
        scale_base: float = 0.3,
        scale_spline: float = 0.1,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_eps = grid_eps

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1)
            * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5) * self.scale_base,
        )

        with torch.no_grad():
            noise = (
                torch.rand(
                    self.grid_size + 1,
                    self.in_features,
                    self.out_features,
                )
                - 0.5
            ) * self.scale_noise / self.grid_size

            self.spline_weight.copy_(
                (
                    self.scale_spline
                    if not self.enable_standalone_scale_spline
                    else 1.0
                )
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline,
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline bases.

        Args:
            x: shape (B*, in_features)

        Returns:
            bases: shape (B*, in_features, grid_size + spline_order)
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (B, {self.in_features}), got {tuple(x.shape)}"
            )

        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]

            bases = (
                (x - grid[:, : -(k + 1)]) / left_den * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / right_den * bases[:, :, 1:]
            )

        expected_shape = (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if bases.size() != expected_shape:
            raise RuntimeError(
                f"Unexpected B-spline shape {tuple(bases.shape)}, expected {expected_shape}"
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit spline coefficients by least squares.

        Args:
            x: shape (num_points, in_features)
            y: shape (num_points, in_features, out_features)

        Returns:
            coeff: shape (out_features, in_features, grid_size + spline_order)
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (B, {self.in_features}), got {tuple(x.shape)}"
            )
        if y.size() != (x.size(0), self.in_features, self.out_features):
            raise ValueError(
                f"Expected y with shape {(x.size(0), self.in_features, self.out_features)}, "
                f"got {tuple(y.shape)}"
            )

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)

        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        expected_shape = (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if result.size() != expected_shape:
            raise RuntimeError(
                f"Unexpected coefficient shape {tuple(result.shape)}, expected {expected_shape}"
            )

        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., in_features)

        Returns:
            output: shape (..., out_features)
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dim={self.in_features}, got {x.size(-1)}"
            )

        original_shape = x.shape
        x = x.contiguous().view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        spline_bases = self.b_splines(x).view(x.size(0), -1)
        spline_weight = self.scaled_spline_weight.view(self.out_features, -1)
        spline_output = F.linear(spline_bases, spline_weight)

        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor):
        """
        Update grid according to the paper:

            G^(t) = (1 - eps) G^(t-1) + eps G_ada^(t)(X)

        Then refit spline coefficients to preserve the original function response.
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (B, {self.in_features}), got {tuple(x.shape)}"
            )

        batch = x.size(0)
        if batch < self.grid_size + 1:
            return

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)

        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)

        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]

        quantile_indices = torch.linspace(
            0,
            batch - 1,
            self.grid_size + 1,
            dtype=torch.long,
            device=x.device,
        )
        grid_adaptive = x_sorted[quantile_indices]

        previous_core_grid = self.grid[
            :, self.spline_order : -self.spline_order
        ].T

        grid_core = (
            (1.0 - self.grid_eps) * previous_core_grid
            + self.grid_eps * grid_adaptive
        )

        left_step = grid_core[1:2] - grid_core[0:1]
        right_step = grid_core[-1:] - grid_core[-2:-1]

        left_extension = grid_core[0:1] - left_step * torch.arange(
            self.spline_order,
            0,
            -1,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(1)

        right_extension = grid_core[-1:] + right_step * torch.arange(
            1,
            self.spline_order + 1,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(1)

        new_grid = torch.cat(
            [left_extension, grid_core, right_extension],
            dim=0,
        )

        self.grid.copy_(new_grid.T)
        self.spline_weight.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        l1_fake = self.spline_weight.abs().mean(-1)
        activation_loss = l1_fake.sum()

        p = l1_fake / (activation_loss + 1e-8)
        entropy_loss = -torch.sum(p * torch.log(p + 1e-8))

        return (
            regularize_activation * activation_loss
            + regularize_entropy * entropy_loss
        )


class KLAM(nn.Module):
    """
    Kolmogorov-Arnold Local Attention Module.

    This version is consistent with the manuscript:
    - split channels into groups
    - partition each group into fixed 2D windows
    - compute self-attention independently inside each window
    - use KAN-enhanced features for Q/K
    - use original window features as V
    - concatenate channel groups back to the original feature dimension
    """

    def __init__(
        self,
        dim: int = 768,
        window_size: int = 7,
        groups: int = 4,
        grid_size: int = 3,
        hidden_dim: int = 64,
        spline_order: int = 3,
        grid_eps: float = 0.02,
    ):
        super().__init__()

        if dim % groups != 0:
            raise ValueError(f"dim={dim} must be divisible by groups={groups}")

        self.dim = dim
        self.window_size = window_size
        self.groups = groups
        self.group_dim = dim // groups
        self.hidden_dim = hidden_dim

        self.kan_reduce = KANLinear(
            in_features=self.group_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_eps=grid_eps,
        )

    @staticmethod
    def _infer_hw_from_sequence_length(n: int) -> Tuple[int, int]:
        h = int(math.sqrt(n))
        if h * h != n:
            raise ValueError(
                f"Cannot infer square spatial size from sequence length N={n}. "
                f"Please provide hw=(H, W)."
            )
        return h, h

    @staticmethod
    def _pad_to_window_size(
        x: torch.Tensor,
        window_size: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: shape (B*, H, W, C)

        Returns:
            padded x, padded H, padded W
        """
        b, h, w, c = x.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size

        if pad_h == 0 and pad_w == 0:
            return x, h, w

        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1).contiguous()

        return x, h + pad_h, w + pad_w

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Args:
            x: shape (B*, H, W, C)

        Returns:
            windows: shape (B* num_windows, window_size*window_size, C)
        """
        b, h, w, c = x.shape
        ws = window_size

        x = x.view(b, h // ws, ws, w // ws, ws, c)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, ws * ws, c)
        )
        return windows

    @staticmethod
    def _window_reverse(
        windows: torch.Tensor,
        window_size: int,
        h: int,
        w: int,
        b: int,
    ) -> torch.Tensor:
        """
        Args:
            windows: shape (B* num_windows, window_size*window_size, C)

        Returns:
            x: shape (B, H, W, C)
        """
        ws = window_size
        c = windows.size(-1)

        x = windows.view(b, h // ws, w // ws, ws, ws, c)
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(b, h, w, c)
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        hw: Optional[Tuple[int, int]] = None,
        update_grid: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                - shape (B, N, C), with hw provided or inferred if N is square
                - shape (B, H, W, C), channels-last
            hw:
                spatial size for sequence input
            update_grid:
                whether to update the KAN grid using the current batch

        Returns:
            output sequence: shape (B, N, C)
        """
        if x.dim() == 4:
            b, h, w, c = x.shape
            x_seq = x.view(b, h * w, c)
        elif x.dim() == 3:
            b, n, c = x.shape
            if hw is None:
                h, w = self._infer_hw_from_sequence_length(n)
            else:
                h, w = hw
                if h * w != n:
                    raise ValueError(
                        f"hw={hw} is incompatible with sequence length N={n}"
                    )
            x_seq = x
        else:
            raise ValueError(
                f"KLAM expects 3D sequence or 4D channels-last tensor, got {x.dim()}D"
            )

        if c != self.dim:
            raise ValueError(f"Expected channel dim={self.dim}, got {c}")

        ws = self.window_size

        x_2d = x_seq.view(b, h, w, c)
        x_grouped = x_2d.view(
            b,
            h,
            w,
            self.groups,
            self.group_dim,
        )
        x_grouped = (
            x_grouped.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(b * self.groups, h, w, self.group_dim)
        )

        x_grouped_pad, hp, wp = self._pad_to_window_size(x_grouped, ws)

        x_windows = self._window_partition(x_grouped_pad, ws)

        if update_grid and self.training:
            self.kan_reduce.update_grid(
                x_windows.reshape(-1, self.group_dim)
            )

        u_windows = self.kan_reduce(x_windows)

        attn = torch.bmm(
            u_windows,
            u_windows.transpose(1, 2),
        ) / math.sqrt(self.hidden_dim)

        attn = torch.softmax(attn, dim=-1)

        y_windows = torch.bmm(attn, x_windows)

        y_grouped_pad = self._window_reverse(
            y_windows,
            ws,
            hp,
            wp,
            b * self.groups,
        )

        y_grouped = y_grouped_pad[:, :h, :w, :].contiguous()

        y_grouped = y_grouped.view(
            b,
            self.groups,
            h,
            w,
            self.group_dim,
        )

        y_2d = (
            y_grouped.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(b, h, w, c)
        )

        y_seq = y_2d.view(b, h * w, c)
        return y_seq


class KAM(nn.Module):
    """
    KAN Adaptive Mixer.

    This wrapper assumes the backbone returns either:
    - channels-last feature map: (B, H, W, C)
    - channels-first feature map: (B, C, H, W)
    - sequence feature: (B, N, C)
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 768,
        num_classes: int = 6,
        grid_size: int = 3,
        window_size: int = 7,
        groups: int = 4,
        hidden_dim: int = 64,
        spline_order: int = 3,
        grid_eps: float = 0.02,
    ):
        super().__init__()

        self.backbone = backbone
        self.dim = dim

        self.meta_kan = KANLinear(
            in_features=dim,
            out_features=2,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_eps=grid_eps,
        )

        self.kan_attention = KLAM(
            dim=dim,
            window_size=window_size,
            groups=groups,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            spline_order=spline_order,
            grid_eps=grid_eps,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    @staticmethod
    def _infer_hw_from_sequence_length(n: int) -> Tuple[int, int]:
        h = int(math.sqrt(n))
        if h * h != n:
            raise ValueError(
                f"Cannot infer square spatial size from sequence length N={n}. "
                f"Please pass a backbone that returns 4D features or modify forward()."
            )
        return h, h

    def _to_sequence(
        self,
        feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert backbone output to sequence format.

        Returns:
            feat_seq: shape (B, N, C)
            hw: (H, W)
        """
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]

        if feat.dim() == 4:
            if feat.size(-1) == self.dim:
                b, h, w, c = feat.shape
                feat_seq = feat.view(b, h * w, c)
                return feat_seq, (h, w)

            if feat.size(1) == self.dim:
                b, c, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 1).contiguous()
                feat_seq = feat.view(b, h * w, c)
                return feat_seq, (h, w)

            raise ValueError(
                f"Cannot identify channel dimension in 4D feature shape {tuple(feat.shape)} "
                f"with expected dim={self.dim}"
            )

        if feat.dim() == 3:
            b, n, c = feat.shape
            if c != self.dim:
                raise ValueError(
                    f"Expected feature dim={self.dim}, got {c}"
                )
            h, w = self._infer_hw_from_sequence_length(n)
            return feat, (h, w)

        raise ValueError(
            f"Backbone output must be 3D or 4D tensor, got {feat.dim()}D"
        )

    def forward(
        self,
        x: torch.Tensor,
        update_grid: bool = False,
    ) -> torch.Tensor:
        base_feat = self.backbone(x)
        base_feat_seq, hw = self._to_sequence(base_feat)

        x_g = base_feat_seq.mean(dim=1)

        if update_grid and self.training:
            self.meta_kan.update_grid(x_g)

        alpha = torch.softmax(self.meta_kan(x_g), dim=-1)

        attn_feat = self.kan_attention(
            base_feat_seq,
            hw=hw,
            update_grid=update_grid,
        )

        y_g = attn_feat.mean(dim=1)

        final_feat = (
            alpha[:, 0:1] * y_g
            + alpha[:, 1:2] * x_g
        )

        return self.classifier(final_feat)

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        loss = self.meta_kan.regularization_loss(
            regularize_activation,
            regularize_entropy,
        )
        loss = loss + self.kan_attention.kan_reduce.regularization_loss(
            regularize_activation,
            regularize_entropy,
        )
        return loss
```
