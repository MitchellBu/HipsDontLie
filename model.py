import torch
from torch import nn
import math
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LandmarksNN(nn.Module):
    def __init__(self, input_dim=cfg.INPUT_DIM, hidden_dim=cfg.HIDDEN_DIM,
     output_dim=cfg.TRANSFORMER_D_MODEL):
        super(LandmarksNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=cfg.LMARKS_DROPOUT)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        ''' 
            Input: batch of sequences of frame landmarks
            Output: batch of sequences of frame embeddings
            Args:
            x: Tensor, shape [batch_size, in_seq_len, tot_lmarks]
            Output:
            outs: Tensor, shape [batch_size, in_seq_len, embedding_dim]
        '''
        out = x
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Transformer(nn.Module):

    def __init__(self, target_size: int,
     d_model: int = cfg.TRANSFORMER_D_MODEL,
     num_heads: int = cfg.TRANSFORMER_N_HEADS,
     num_encoder_layers: int = cfg.TRANSFORMER_ENCODER_LAYERS,
     num_decoder_layers: int = cfg.TRANSFORMER_DECODER_LAYERS, 
     dim_feedforward: int = cfg.TRANSFORMER_DIM_FEEDFORWARD,
     dropout: int = cfg.TRANSFORMER_DROPOUT):
        super(Transformer, self).__init__()
        # Positional encoding and Embedding layers
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(target_size, d_model)

        # Transformer architecture and feedforward layer
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True, nhead=num_heads,
         num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
         dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(d_model, target_size)

        # Generate input and target masks
        self.batch_in_mask = torch.zeros(cfg.SEQUENCE_IN_MAX_LEN, cfg.SEQUENCE_IN_MAX_LEN)
        self.batch_in_mask = self.batch_in_mask.to(device)
        self.batch_tgt_mask = nn.Transformer.generate_square_subsequent_mask(cfg.SEQUENCE_OUT_MAX_LEN)
        self.batch_tgt_mask = self.batch_tgt_mask.to(device)

    def forward(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor, batch_in_pad_masks: torch.Tensor, batch_tgt_pad_masks: torch.Tensor, inference_mode: bool =False) -> torch.Tensor:
        ''' 
            Input: batch of sequences of frame embeddings
            Output: token probabilities for each sequence in the batch
            Args:
            batch_inputs: Tensor, shape [batch_size, in_seq_len, embedding_dim]
            batch_targets:  Tensor, shape [batch_size, out_seq_len, embedding_dim]
            batch_in_pad_masks: Tensor, shape [batch_size, in_seq_len]
            batch_tgt_pad_masks: Tensor, shape [batch_size, out_seq_len] 
            Output:
            outs: Tensor, shape [batch_size, out_seq_len, vocab_size] 
        '''

        # Add a positional encoding to the input and target tokens
        batch_inputs = self.pos_encoder(batch_inputs)
        batch_targets = self.embedding(batch_targets)
        batch_targets = self.pos_encoder(batch_targets)

        # Perform a forward pass in the transformer architecture
        outs = self.transformer(batch_inputs, batch_targets, src_mask=self.batch_in_mask, tgt_mask=self.batch_tgt_mask,
            src_key_padding_mask=batch_in_pad_masks, tgt_key_padding_mask=batch_tgt_pad_masks)
        outs = self.generator(outs)
        return outs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x[:] += self.pe[:x.size(1)]
        return self.dropout(x)