import torch
from torch import nn
import torchvision.models as models
from dataloader import DataLoader, Tokenizer
import datetime

pretrained_feature_extractor = models.vgg11(pretrained=True)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.up_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.features = pretrained_feature_extractor.features
        self.avg_pool = nn.AvgPool2d(kernel_size=(9,11), stride=1)
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
        res = []
        for y in x:
            y = self.up_conv(y)
            y = self.features(y)
            y = self.avg_pool(y)
            res.append(y.view(y.size(0), -1))
        return res


VOCABULARY_FILE = 'vocab.txt'
VIDEOS_DIR = 'videos'
ANNOTATIONS_DIR = 'alignment'
vocab_file = open(VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
word2idx = {word:idx for idx, word in enumerate(idx2word)}

encoder = Encoder()

loader = DataLoader(videos_path=VIDEOS_DIR, annotations_path=ANNOTATIONS_DIR, batch_size=1, shuffle=False)

net_in, labels = next(loader)

t1 = datetime.datetime.now()

print(f'Net input shape: {net_in[0].shape}')
net_out = encoder(net_in)
print(f'Net output shape: {net_out[0].shape}')

tokenizer = Tokenizer(word2idx, net_out, labels)
batch_inputs, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks = tokenizer.tokenize()
print(f'Transformer input shape:  {batch_inputs.shape}')
print(f'Transformer target shape: {batch_targets.shape}')
print(f'Transformer input padding mask shape: {batch_in_pad_masks.shape}')
print(f'Transformer target padding mask shape: {batch_tgt_pad_masks.shape}')

embedding = nn.Embedding(len(idx2word), 512)
batch_targets = embedding(batch_targets)
transformer = nn.Transformer(batch_first=True)

batch_in_mask = torch.zeros(batch_inputs.size(1), batch_inputs.size(1))
batch_tgt_mask = nn.Transformer.generate_square_subsequent_mask(batch_targets.size(1))
out = transformer.forward(batch_inputs, batch_targets, src_mask=batch_in_mask, tgt_mask=batch_tgt_mask,
 src_key_padding_mask=batch_in_pad_masks, tgt_key_padding_mask=batch_tgt_pad_masks)

t2 = datetime.datetime.now()
print(out.shape)
print(f'Forward pass + gradient computation took {t2-t1}')