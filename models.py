# models/resnet32.py
import torch.nn as nn
import torch.nn.functional as F
from avalanche.models import MultiHeadClassifier, MultiTaskModule, BaseModel

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.layer1 = self._make_layer(16, 5)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ────────────────────── 공통 유틸 ──────────────────────
def conv3x3(in_planes, out_planes, stride=1):
    """3×3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1×1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


# ───────────────────── Bottleneck 블록 ─────────────────────
class Bottleneck(nn.Module):
    """ResNet Bottleneck (출력 채널 = planes × expansion)"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1×1 ↓차원축소
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        # 3×3
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2   = nn.BatchNorm2d(planes)
        # 1×1 ↑차원확장
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        # shortcut (identity or 1×1 conv matching shape)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


# ───────────────────── ResNet-50 (Tiny용) ─────────────────────
class ResNet50Tiny(nn.Module):
    """
    •  입력: 3×64×64  
    •  conv1: 3×3 stride 1 (TinyImageNet에 맞춰 downsampling 최소화)  
    •  layer 구성: [3, 4, 6, 3] Bottleneck  
    •  최종 FC: num_classes (=200)
    """
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.in_planes = 64

        # conv1 (64 → 64, 그대로)  ─────────────
        # TinyImageNet은 작기 때문에 7×7 stride 2 대신 3×3 stride 1
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        # ↓ 원본 ResNet 구조를 쓰려면 아래 두 줄을 활성화
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()   # max-pool 생략

        # 4개의 스테이지 ─────────────
        self.layer1 = self._make_layer(planes=64,  blocks=3, stride=1)  # 64×64
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)  # 32×32
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)  # 16×16
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)  # 8×8

        self.bn_final = nn.BatchNorm2d(512 * Bottleneck.expansion)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(512 * Bottleneck.expansion, num_classes)

    # ───────────────── make_layer helper ─────────────────
    def _make_layer(self, planes, blocks, stride):
        """planes: 내부 채널(축소 전), blocks: Bottleneck 개수"""
        strides = [stride] + [1] * (blocks - 1)
        layers  = []
        for s in strides:
            layers.append(Bottleneck(self.in_planes, planes, s))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    # ───────────────── forward ─────────────────
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
    
    

class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 drop_rate=0, relu_act=True):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        """
        :param initial_out_features: if >0 override output size and build an
            IncrementalClassifier with `initial_out_features` units as first.
        """
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)

        if initial_out_features > 0:
            self.classifier = avalanche.models.IncrementalClassifier(in_features=hidden_size,
                                                                     initial_out_features=initial_out_features)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        return self.features(x)


class SI_CNN(MultiTaskModule):
    def __init__(self, hidden_size=512):
        super().__init__()
        layers = nn.Sequential(*(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(2304, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5)
                                 ))
        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size, initial_out_features=10)

    def forward(self, x, task_labels):
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class FlattenP(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class MLP_gss(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP_gss, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(FlattenP(), *layers)

    def forward(self, x):
        return self.net(x)
class FeatureExtractorMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=512, hidden_layers=3, output_size=10, drop_rate=0.2, relu_act=True):
        """
        SingleHeadReducedResNet18과 동일한 기능을 수행하는 FeatureExtractorMLP 모델
        """
        super().__init__()
        self._input_size = input_size  # 입력 크기 (MNIST: 28x28 → 784)
        self.feature_size = hidden_size  # 특징 벡터 크기 (512)

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))  # 첫 번째 FC 레이어
        layers.append(nn.ReLU(inplace=True) if relu_act else nn.Tanh())
        layers.append(nn.Dropout(p=drop_rate))

        # Hidden layers 추가
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True) if relu_act else nn.Tanh())
            layers.append(nn.Dropout(p=drop_rate))

        self.feature_extractor = nn.Sequential(*layers)  # Feature Extractor 부분

        # 🔹 Single-head Classifier 추가 (ResNet과 동일한 구조)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (이미지 → 1D 벡터 변환)
        features = self.feature_extractor(x)  # Feature Extractor 통과
        logits = self.classifier(features)  # Classification
        return logits

    def get_features(self, x):
        """ Feature Extractor 역할 수행 (512-D 특징 벡터 반환) """
        x = x.view(x.size(0), -1)  # Flatten
        return self.feature_extractor(x)



__all__ = ['MultiHeadMLP', 'MLP', 'SI_CNN', 'MLP_gss', 'FeatureExtractorMLP']