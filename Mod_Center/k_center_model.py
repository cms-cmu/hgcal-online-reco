import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, pos_xyz):
        # pos_xyz: [B, T, 3]
        return self.mlp(pos_xyz)  # [B, T, dim]


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_output)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTConfig:
    def __init__(self, block_size, n_layer, n_head, n_embd, dropout):
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class SetTransformer(nn.Module):
    """Set Transformer for predicting k centers using set-to-set transformation."""

    def __init__(self, config, input_dim, k: int = 2):
        super().__init__()
        self.k = k

        self.feature_proj = nn.Linear(5, config.n_embd)
        self.pos_emb_mlp = PositionalEmbedding3D(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Set Transformer components
        self.encoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)

        # Learnable k query vectors (set transformer approach)
        self.k_queries = nn.Parameter(torch.randn(1, k, config.n_embd))

        # Cross-attention between k queries and encoded nodes
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output MLP for each center
        self.output_mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 3),  # 3 coordinates per center
        )

        # Covariance output MLP for each center (predict upper-triangular 6 values)
        self.cov_output_mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 6),  # 6 covariance parameters per center
        )

        # K-prediction head - predict number of clusters
        self.k_prediction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, k),  # Output logits for k=1 to max_k
        )

        # Node activity prediction head (per-node), used to gate inactive nodes
        self.node_activity_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1),
        )

    def forward(self, x):
        """Return predicted centres (B,k,3) using set transformer approach.

        Also returns per-node activity logits for optional BCE supervision.
        """
        # Use all 8 features
        features = x[:, :, 0:5]  # (B,T,8) - all features
        xyz = x[..., -3:]  # (B,T,3) - spatial coordinates for positional embedding

        # Process features through encoder
        h = self.feature_proj(features) + self.pos_emb_mlp(xyz)
        h = self.dropout(h)
        h = self.encoder_blocks(h)
        h = self.ln_f(h)  # (B,T,C)

        # Predict node activity and apply soft gating to suppress inactive nodes
        activity_logits = self.node_activity_head(h)            # (B,T,1)
        activity_prob = torch.sigmoid(activity_logits)          # (B,T,1)
        h = h * activity_prob                                   # soft gating

        # Set transformer: use k learnable queries to attend to encoded nodes
        batch_size = h.size(0)
        k_queries = self.k_queries.expand(batch_size, -1, -1)  # (B,k,C)

        # Cross-attention: k queries attend to all encoded nodes
        attended_centers, attention_weights = self.cross_attention(
            query=k_queries,    # (B,k,C) - k queries
            key=h,              # (B,T,C) - encoded nodes
            value=h,            # (B,T,C) - encoded nodes
        )  # (B,k,C)

        # Generate coordinates for each center
        centers = self.output_mlp(attended_centers)  # (B,k,3)
        covariances = self.cov_output_mlp(attended_centers)  # (B,k,6)

        # Predict k (number of clusters) - use global average pooling
        # Global average pooling over nodes to get event-level representation
        x_global = h.mean(dim=1)  # [B, n_embd] - average over all nodes
        k_logits = self.k_prediction_head(x_global)  # [B, max_k]

        return centers, covariances, k_logits, activity_logits


