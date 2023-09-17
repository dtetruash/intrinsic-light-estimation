import torch
import torch.nn.functional as F
import torchvision


class LightMLP(torchvision.ops.MLP):
    def __init__(self, num_feats, hidden_channels, norm_layer=None, activation_layer=torch.nn.modules.activation.ReLU, inplace=None, bias=True, dropout=0.0):
        super().__init__(num_feats * 3, hidden_channels + [3], norm_layer, activation_layer, inplace, bias, dropout)

    def forward(self, x):
        x = super().forward(x)
        x = F.normalize(x)
        assert x.shape[-1] == 3, f"Model inputs' last dimention was not 3, was {x.shape[-1]}."
        return x

# TODO: Impelemenet model which predicts SH coeficients
class LightSH(torchvision.ops.MLP):
    pass
