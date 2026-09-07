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


class OffsetHead(nn.Module):
    """Predict per-node offset Δ so that centre = xyz + Δ.

    Uses two linear layers with a residual skip for stability.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_dim, 3)

    def forward(self, h):
        # h: (..., C)
        z = self.act(self.fc1(h))
        h = h + z                  # residual skip
        offset = self.fc2(h)       # (..., 3)
        return offset


class GPTEncoderModel(nn.Module):
    def __init__(self, config, input_dim, max_k=10):
        """input_dim is the number of *non-geometric* feature channels.

        The last three columns of the incoming tensor are xyz; model outputs
        centre = xyz + Δ where Δ is a learned offset.

        max_k: maximum number of clusters to predict (k will be in range [1, max_k])
        """
        super().__init__()
        self.config = config
        self.max_k = max_k
        # input_dim should be 5 for the non-geometric features (first 5 features)
        self.input_proj = nn.Linear(1, config.n_embd)  # Fixed: 5 features, not 8
        self.xyz_affine = nn.Linear(3, 3, bias=True)
        # initialise affine as identity
        with torch.no_grad():
            self.xyz_affine.weight.copy_(torch.eye(3))
            self.xyz_affine.bias.zero_()

        self.pos_emb_mlp = PositionalEmbedding3D(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)

        # Fusion layer to combine energy features and positional embeddings
        self.feature_fusion = nn.Linear(config.n_embd + config.n_embd, config.n_embd)

        # Center prediction head - predict absolute centers
        self.center_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd, 3),
        )

        # Energy attention head - learn which nodes are important
        self.energy_attention = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid(),
        )

        # NEW: Node indicator head - predict active/inactive nodes
        self.node_indicator_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid(),  # Output probability between 0 and 1
        )

        # K-prediction head - predict number of clusters
        self.k_prediction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, max_k),  # Output logits for k=1 to max_k
        )

        # NEW: Per-node covariance prediction head - predict inv_cov_upper (6 values) for each node
        self.covariance_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, 6),  # Output 6 values for inv_cov_upper per node
        )

    def forward(self, x):
        # Process all 8 features together: [node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z]
        # Extract xyz coordinates for positional embedding and residual connection
        pos_xyz = x[..., 5:8]  # xyz columns (indices 5,6,7)
        feats = x[..., :1]     # first 5 features (indices 0,1,2,3,4)

        # Project features to embedding dimension
        energy_emb = self.input_proj(feats)

        # Create positional embedding from xyz coordinates
        pos_emb = self.pos_emb_mlp(pos_xyz)

        # Concatenate feature embeddings and positional embeddings, then fuse
        combined = torch.cat([energy_emb, pos_emb], dim=-1)  # [B, T, 2*n_embd]
        x = self.feature_fusion(combined)  # [B, T, n_embd]

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        """
        # Energy attention
        energy_weights = self.energy_attention(x)  # [B, T]
        x = x * energy_weights  # [B, T, n_embd]
        """
        # Predict absolute center coordinates
        centres_pred = self.center_head(x)  # (B,T,3)
        centres_pred = centres_pred + pos_xyz  # Add residual jump with xyz

        # NEW: Predict node indicators (active/inactive)
        node_indicators = self.node_indicator_head(x)  # [B, T, 1]
        node_indicators = node_indicators.squeeze(-1)  # [B, T]

        # Predict k (number of clusters) - use global average pooling
        # Global average pooling over nodes to get event-level representation
        x_global = x.mean(dim=1)  # [B, n_embd] - average over all nodes
        k_logits = self.k_prediction_head(x_global)  # [B, max_k]

        # NEW: Predict per-node covariance (inv_cov_upper) - use node-level features
        inv_cov_upper_pred = self.covariance_head(x)  # [B, T, 6] - per-node predictions

        return centres_pred, k_logits, node_indicators, inv_cov_upper_pred  # Also return covariance prediction


