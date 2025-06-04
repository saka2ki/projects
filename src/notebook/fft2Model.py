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
        attn_mask = torch.triu(torch.ones(T-1, T-1, device=x.device), diagonal=1).bool()

        def forward(self, x):
    B, T, d_model = x.size()
    x = self.token_emb(x) + self.pos_emb[:, :T]
    x = self.dropout(x)

    attn_mask = torch.triu(torch.ones(T - 1, T - 1, device=x.device), diagonal=1).bool()

    for layer in self.layers:
        # 2D FFT
        x_fft = torch.fft.fft2(x, dim=(1, 2))

        # 低周波成分1/4抽出
        x_low = x_fft[:, :T//2, :d_model//2]

        # 実部・虚部に分離
        x_low_real = x_low.real
        x_low_imag = x_low.imag

        # そのままアテンションに通す
        x_low_real = layer['ln1'](x_low_real)
        attn_out_real, _ = layer['attn'](
            x_low_real, x_low_real, x_low_real,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x_low_real = x_low_real + attn_out_real
        x_low_real = x_low_real + layer['mlp'](layer['ln2'](x_low_real))

        x_low_imag = layer['ln1'](x_low_imag)
        attn_out_imag, _ = layer['attn'](
            x_low_imag, x_low_imag, x_low_imag,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x_low_imag = x_low_imag + attn_out_imag
        x_low_imag = x_low_imag + layer['mlp'](layer['ln2'](x_low_imag))

        # 複素数に戻す
        x_low_processed = torch.complex(x_low_real, x_low_imag)

        # ゼロパディングで元のサイズに戻す
        pad_T = T - T//2
        pad_d = d_model - d_model//2
        x_padded = torch.nn.functional.pad(x_low_processed, (0, pad_d, 0, pad_T))

        # 逆FFTで実空間に戻す
        x = torch.fft.ifft2(x_padded, dim=(1, 2)).real

    x = self.ln_f(x)
    logits = self.head(x)
    return logits


        x = self.ln_f(x)
        logits = self.head(x)
        return logits
