import torch
import torch.nn as nn
from torch.autograd import Function

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.silu1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.silu2 = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.silu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        # Residual addition: Preserves the fidelity of original features, greatly alleviating gradient fragmentation
        return self.silu2(out + residual)

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_MLP(nn.Module):
    def __init__(self, input_dim=4366, num_domains=2, num_genders=2, dropout_rate=0.3):
        super(DANN_MLP, self).__init__()
        # Encoder: Uses a feature mapping layer + two residual blocks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(256, dropout_rate),
            ResidualBlock(256, dropout_rate)
        )
        self.stage_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64), # Add LayerNorm in Head for enhanced stability
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, num_domains)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, num_genders)
        )
        self.age_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, alpha=1.0):
        features = self.encoder(x)
        stage_logits = self.stage_head(features)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_head(reversed_features)
        gender_logits = self.gender_head(reversed_features)
        age_pred = self.age_head(reversed_features)
        return stage_logits, domain_logits, gender_logits, age_pred
