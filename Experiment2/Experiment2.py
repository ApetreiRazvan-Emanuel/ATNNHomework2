import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
import pandas as pd
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm


# Epoch 200: Train Acc: 98.56%, Val Acc: 71.45%, Best: 71.60%: 100%|██████████| 200/200 [2:28:38<00:00, 44.59s/it]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(enabled=enable_half)


mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]


train_transforms = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])


train_set = CIFAR100('./', download=False, train=True, transform=train_transforms)
test_set = CIFAR100('./', download=False, train=False, transform=test_transforms)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


model = VGG16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.85, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10,
                                                       threshold=0.001, threshold_mode='rel')


def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


best = 0.0
epochs = list(range(200))
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()
        scheduler.step(val_acc / 100)
        if val_acc > best:
            best = val_acc
        tbar.set_description(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best: {best:.2f}%")



@torch.inference_mode()
def inference():
    model.eval()
    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels

data = {"ID": [], "target": []}
for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("submission2.csv", index=False)
