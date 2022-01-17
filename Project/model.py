import torch
from torch import nn
import torchvision.models as models

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


class Transformer(nn.Module):

    def __init__(self, target_size, d_model=512):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(target_size, d_model)
        self.transformer = nn.Transformer(batch_first=True)
        self.generator = nn.Linear(d_model, target_size)

    def forward(self, batch_inputs, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks):
        batch_targets = self.embedding(batch_targets)
        batch_in_mask = torch.zeros(batch_inputs.size(1), batch_inputs.size(1))
        batch_tgt_mask = nn.Transformer.generate_square_subsequent_mask(batch_targets.size(1))
        outs = self.transformer(batch_inputs, batch_targets, src_mask=batch_in_mask, tgt_mask=batch_tgt_mask,
            src_key_padding_mask=batch_in_pad_masks, tgt_key_padding_mask=batch_tgt_pad_masks)
        outs = self.generator(outs)
        return outs