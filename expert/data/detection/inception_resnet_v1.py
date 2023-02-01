from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import gdown
import os

from expert.core.utils import get_torch_home


class BasicConv2d(nn.Module):
    
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int = 0
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class Block35(nn.Module):
    
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        
        self.scale = scale
        self.branch0 = BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        
        return out


class Block17(nn.Module):
    
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        
        self.scale = scale
        self.branch0 = BasicConv2d(in_planes=896, out_planes=128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=128, kernel_size=1, stride=1),
            BasicConv2d(in_planes=128, out_planes=128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=128, out_planes=128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv2d = nn.Conv2d(in_channels=256, out_channels=896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        
        return out


class Block8(nn.Module):
    
    def __init__(self, scale: float = 1.0, noReLU: bool = False) -> None:
        super().__init__()
        
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv2d = nn.Conv2d(in_channels=384, out_channels=1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        
        return out


class Mixed_6a(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.branch0 = BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=192, kernel_size=1, stride=1),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=192, out_planes=256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        
        return out


class Mixed_7a(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1),
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights
    
    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.
    
    Example:
        >>> import torch
        >>> device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        >>> resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    """
    
    def __init__(
        self,
        pretrained: str | None = None,
        classify: bool = False,
        num_classes: int | None = None,
        dropout_prob: float | None = 0.6,
        device: torch.device | None = None
    ) -> None:
        """
        Args:
            pretrained (string, optional): Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            classify (bool, optional): Whether the model should output classification probabilities or feature embeddings.
            num_classes (int | None, optional): Number of output classes. If 'pretrained' is set and num_classes not
                equal to that used for the pretrained model, the final linear layer will be randomly
                initialized.
            dropout_prob (float | None, optional): Dropout probability.
            device (torch.device | None, optional): Object representing device type.
        """
        
        super().__init__()
        
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes
        
        if pretrained == "vggface2":
            tmp_classes = 8631
            path = "https://drive.google.com/uc?export=view&id=1P4OqfwcUXXuycmow_Fb8EXqQk5E7-H5E"
            model_name = "20180402-114759-vggface2.pt"
        elif pretrained == "casia-webface":
            tmp_classes = 10575
            path = "https://drive.google.com/uc?export=view&id=1rgLytxUaOUrtjpxCl-mQFGYdUfSWgQCo"
            model_name = "20180408-102900-casia-webface.pt"
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified.')
        
        self.conv2d_1a = BasicConv2d(in_planes=3, out_planes=32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_3b = BasicConv2d(in_planes=64, out_planes=80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(in_planes=80, out_planes=192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(in_planes=192, out_planes=256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.last_linear = nn.Linear(in_features=1792, out_features=512, bias=False)
        self.last_bn = nn.BatchNorm1d(num_features=512, eps=0.001, momentum=0.1, affine=True)
        self._device = torch.device("cpu")
        
        if pretrained is not None:
            self.logits = nn.Linear(in_features=512, out_features=tmp_classes)
            
            model_dir = os.path.join(get_torch_home(), "checkpoints")
            os.makedirs(model_dir, exist_ok=True)
            cached_file = os.path.join(model_dir, os.path.basename(model_name))
            
            if not os.path.exists(cached_file):
                gdown.download(path, cached_file, quiet=False)
            
            state_dict = torch.load(cached_file, map_location=self._device)
            self.load_state_dict(state_dict, strict=True)
        
        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(in_features=512, out_features=self.num_classes)
        
        if device is not None:
            self._device = device
            self.to(self._device)
    
    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device
    
    def forward(self, x: Tensor) -> Tensor:
        """Calculate embeddings or logits given a batch of input image tensors
        
        Args:
            x (Tensor): Batch of image tensors representing faces.
        
        Returns:
            Tensor: Batch of embedding vectors or multinomial logits.
        """
        
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        
        return x