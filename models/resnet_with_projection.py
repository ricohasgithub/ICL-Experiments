import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from models.resnet import CustomResNet

class ProjectionResNet(CustomResNet):
    def __init__(self, blocks_per_group, channels_per_group, num_classes, flatten_superpixels=False):
        super(ProjectionResNet, self).__init__(
            blocks_per_group, channels_per_group, flatten_superpixels
        )
        self.num_classes = num_classes
        self.projection_layer = nn.Linear(channels_per_group[3], self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        # self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def sequence_pass(self, x):
        x = super(ProjectionResNet, self).forward(x)
        x = self.projection_layer(x)
        x = self.softmax(x)
        return x
    
    def forward(self, x):
        # Apply the function to each item in the batch independently
        batch_size = x.size(0)
        return torch.stack([self.sequence_pass(x[i]) for i in range(batch_size)])

if __name__ == "__main__":
    blocks_per_group = [2, 2, 2, 2]
    channels_per_group = [9, 32, 32, 27]
    num_classes = 1623
    custom_resnet = ProjectionResNet(blocks_per_group, channels_per_group, num_classes=num_classes)

    input_tensor = torch.randn(100, 9, 1, 224, 224)

    output = custom_resnet(input_tensor)
    print("Output shape:", output.shape)