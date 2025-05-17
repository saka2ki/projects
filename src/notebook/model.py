# %%
import torch
import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout),
                )
            }) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.LayerNorm) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.size()
        x = self.token_emb(x) + self.pos_emb[:,:T]
        x = self.dropout(x)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x_norm = layer['ln1'](x)
            attn_output, _ = layer['attn'](
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=False,
            )
            x = x + attn_output
            x = x + layer['mlp'](layer['ln2'](x))

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
