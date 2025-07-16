import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=0.3,
            scale_spline=0.1,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    @property
    def scaled_spline_weight(self):

        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):

        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.contiguous().view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )
        
class KAM(nn.Module):
    """
    KAN Adaptive Mixer.
    """
    def __init__(self, backbone: nn.Module, dim=768, num_classes=6, grid_size=3):
        super().__init__()
        self.backbone = backbone

        self.meta_kan = KANLinear(
            in_features=dim,
            out_features=2,
            grid_size=grid_size
        )
        self.kan_attention = KLAM(
            dim=dim,
            window_size=7,
            groups=4
        )
        self.tau = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        base_feat = self.backbone(x)  # (B, H, W, C)
        if base_feat.dim() == 4:
            B, H, W, C = base_feat.shape
            base_feat_seq = base_feat.view(B, H * W, C)  # (B, N, C)
        elif base_feat.dim() == 3:
            base_feat_seq = base_feat  # Already (B, N, C)
        else:
            raise ValueError("Backbone output must be 3D or 4D tensor")

        alpha = torch.softmax(self.meta_kan(base_feat_seq.mean(1)), dim=-1)  # (B, 2)
        attn_feat = self.kan_attention(base_feat_seq)  # (B, N, C)

        final_feat = alpha[:, 0:1] * attn_feat.mean(1) + alpha[:, 1:2] * base_feat_seq.mean(1)
        return self.classifier(final_feat)
class KLAM(nn.Module):
    """
    Kolmogorov-Arnold Local Attention Module.
    """
    def __init__(self, dim=768, window_size=7, groups=4, grid_size=3, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.groups = groups

        assert dim % groups == 0, "dim must be divisible by groups"

        self.kan_reduce = KANLinear(
            in_features=dim // groups,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=3
        )
        self.kan_attn = KANLinear(
            in_features=2 * hidden_dim,
            out_features=1,
            grid_size=grid_size,
            spline_order=3
        )

    def forward(self, x):
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
        B, N, D = x.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        x_grouped = x.view(B, N, self.groups, -1)  # (B, N, G, C//G)
        M = N // (self.window_size ** 2)
        if M == 0:
            raise ValueError("Window size too large for given input length")

        x_windows = x_grouped.view(
            B, M, self.window_size, self.window_size, self.groups, -1
        ).permute(0, 1, 4, 2, 3, 5).contiguous()

        attn_scores = []
        for g in range(self.groups):
            x_g = self.kan_reduce(x_windows[:, :, g])  # shape: (B, M, Wh, Ww, hidden_dim)
            q = x_g.view(B, -1, x_g.shape[-1])
            k = x_g.view(B, -1, x_g.shape[-1])
            scores = torch.einsum('bic,bjc->bij', q, k) / math.sqrt(x_g.shape[-1])
            attn_scores.append(scores.unsqueeze(1))

        attn_matrix = torch.softmax(torch.cat(attn_scores, dim=1), dim=-1)  # (B, G, N, N)
        output = torch.bmm(attn_matrix.sum(dim=1), x)  # (B, N, D)
        return output
