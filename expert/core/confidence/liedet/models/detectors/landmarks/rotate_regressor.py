import torch.nn as nn


class Regressor(nn.Sequential):
    def __init__(self) -> None:
        layers_dims = [478 * 3, 1024, 512, 256, 128, 64]
        drop_probs = [0.2, 0.2, 0.2, 0.1, 0.1]
        layers: list[nn.Module] = []
        for in_features, out_features, drop_prob in zip(
            layers_dims[:-1], layers_dims[1:], drop_probs
        ):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=in_features, out_features=out_features
                    ),
                    nn.Dropout(p=drop_prob, inplace=True),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(in_features=64, out_features=3))

        super().__init__(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)
