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
        self.kan_attention = KANAttentionLite(
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