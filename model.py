import torch
from torch import nn
import torchvision.models as models
import math

TRANSFORMER_D_MODEL = 128
TRANSFORMER_N_HEADS = 4

pretrained_feature_extractor = models.vgg11(pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # Match the number of channels to 3 (RGB)
        self.up_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        # Use pretrained convlution network
        self.features = pretrained_feature_extractor.features
        # Average the results to match d_model features
        self.feed_forward = nn.Linear(1024, TRANSFORMER_D_MODEL)
        self.weights_init_()

    def weights_init_(self):
        for idx, m in enumerate(self.features):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                weight_clone = torch.clone(pretrained_feature_extractor.features[idx].weight.detach())
                self.features[idx].weight = nn.Parameter(weight_clone, requires_grad=True)
                bias_clone = torch.clone(pretrained_feature_extractor.features[idx].bias.detach())
                self.features[idx].bias = nn.Parameter(bias_clone, requires_grad=True)

    def forward(self, x):
        ''' Forward mode. x shape: (Batch, Frames, C, H, W)'''

        batch_size, num_of_frames = x.size(0), x.size(1) # (Batch, Frames, C, H, W)
        # Reshape x to perform convolution
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.up_conv(x)
        x = self.features(x)
        x = self.feed_forward(x.view(x.size(0), -1))
        x = x.view(batch_size, num_of_frames, -1) # Convert to (Batch, Frames, Features)
        return x


class Transformer(nn.Module):

    def __init__(self, target_size, d_model=TRANSFORMER_D_MODEL, num_heads=TRANSFORMER_N_HEADS):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(target_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True, nhead=num_heads)
        self.generator = nn.Linear(d_model, target_size)

    def forward(self, batch_inputs, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks):
        batch_inputs *= math.sqrt(self.d_model)
        batch_inputs = self.pos_encoder(batch_inputs)
        batch_targets = self.embedding(batch_targets)
        batch_targets = self.pos_encoder(batch_targets)
        batch_in_mask = torch.zeros(batch_inputs.size(1), batch_inputs.size(1))
        batch_in_mask = batch_in_mask.to(device)
        batch_tgt_mask = nn.Transformer.generate_square_subsequent_mask(batch_targets.size(1))
        batch_tgt_mask = batch_tgt_mask.to(device)
        outs = self.transformer(batch_inputs, batch_targets, src_mask=batch_in_mask, tgt_mask=batch_tgt_mask,
            src_key_padding_mask=batch_in_pad_masks, tgt_key_padding_mask=batch_tgt_pad_masks)
        outs = self.generator(outs)
        return outs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)