# This code is a slight modification of the original method, check https://github.com/zhouhaoyi/Informer2020 for the original implementation

import torch
import torch.nn as nn

from app.models.networks.types.informer.auxiliary import \
    ProbAttention, \
    FullAttention, \
    Encoder, \
    EncoderLayer, \
    AttentionLayer, \
    ConvLayer, \
    Decoder, \
    DecoderLayer, \
    DataEmbedding


# TODO add code citations


class Informer(nn.Module):
    def __init__(self, lag_size, prediction_size, number_features, device,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True):
        super(Informer, self).__init__()

        enc_in = number_features
        dec_in = number_features
        c_out = number_features  # M behaviour
        self.pred_len = 1

        self.output_modifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_out * self.pred_len, prediction_size)
        )

        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def real_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = self.output_modifier(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]

    def forward(self, x, y=None):
        seq_x, seq_x_mark, seq_y, seq_y_mark = x
        # seq_x: [Batch, Channel, Lag size]
        # seq_x_mark: [Batch, Lag size, Channel]
        # seq_y: [Batch, Channel, Label Len + Pred Len]
        # seq_y_mark: [Batch, Lag size, Channel]

        seq_x = torch.transpose(seq_x, 1, 2)
        seq_y = torch.transpose(seq_y, 1, 2)
        # seq_x: [Batch, Lag size, Channel]
        # seq_y: [Batch, Label Len + Pred Len, Channel]
        return self.real_forward(seq_x, seq_x_mark, seq_y, seq_y_mark)

    def predict(self, x):
        return self.forward(x)
