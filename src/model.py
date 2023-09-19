import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models.resnet import ResNet, BasicBlock

def load_model(model_name, pretrained, label_len, task_loss='cross_entropy'):

    if task_loss == 'mse':
        label_len = 1

    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, label_len)
    elif model_name == 'resnet18_activations':
        model = resnet18_activations(pretrained, label_len)

    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, label_len)
        )
    elif model_name == 'vgg16_activations':
        model = VGG16_activations(pretrained, label_len)
    elif model_name == 'vgg16_4_activations':
        model = VGG16_4_activations(pretrained, label_len)

    elif model_name == 'Net_one':
        model = Net_one(label_len)

    else:
        print("Invalid model name, exiting...")
        exit(1)
    return model

# use <model>_activations to output activations as part of the forward pass
# https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7
class resnet18_activations(ResNet):
    def __init__(self, pretrained, label_len):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        self.load_state_dict(models.resnet18(weights=weights).state_dict())
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, label_len)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        act1 = x
        x = self.layer2(x)
        act2 = x
        x = self.layer3(x)
        act3 = x
        x = self.layer4(x)
        act4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, (act1, act2, act3, act4)

class Densenet121_activations(nn.Module):
    def __init__(self, pretrained, label_len):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        self.model = models.densenet121(weights=weights)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, label_len)

    def forward(self, input):
        x = self.model.features[:4](input)

        x = self.model.features.denseblock1(x)
        x = self.model.features.transition1(x)
        act1 = x
        
        x = self.model.features.denseblock2(x)
        x = self.model.features.transition2(x)
        act2 = x

        x = self.model.features.denseblock3(x)
        x = self.model.features.transition3(x)
        act3 = x

        x = self.model.features.denseblock4(x)
        x = self.model.features.norm5(x)
        act4 = x
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x, (act1, act2, act3, act4)

class VGG16_activations(nn.Module):
    def __init__(self, pretrained, label_len):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        self.model = models.vgg16(weights=weights)
        num_ftrs = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, label_len)
        )

    def forward(self, input):
        x = self.model.features[:5](input)
        act1 = x
        x = self.model.features[5:10](x)
        act2 = x
        x = self.model.features[10:17](x)
        act3 = x
        x = self.model.features[17:24](x)
        act4 = x
        x = self.model.features[24:](x)
        act5 = x
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x, (act1, act2, act3, act4, act5)

class VGG16_4_activations(nn.Module):
    def __init__(self, pretrained, label_len):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        if pretrained:
            print("loading imagenet weights")
        self.model = models.vgg16(weights=weights)
        num_ftrs = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, label_len)
        )

    def forward(self, input):
        x = self.model.features[:5](input)
        x = self.model.features[5:10](x)
        act1 = x
        x = self.model.features[10:17](x)
        act2 = x
        x = self.model.features[17:24](x)
        act3 = x
        x = self.model.features[24:](x)
        act4 = x
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x, (act1, act2, act3, act4)

class Net_one(nn.Module):
    
    def __init__(self, label_len):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, stride=1, padding=1) # num_params = out_c * (in_c * kernel * kernel + 1 bias)
        self.conv2 = nn.Conv2d(100, 100, 3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(6*6*100, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        act1 = x
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        act2 = x
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x, (act1, act2)