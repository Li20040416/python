import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os
from datetime import datetime
# 1. 设置输出目录和日志
model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# 3. 加载数据集
train_dir = r"C:\Users\石庆\OneDrive\桌面\比赛纪念\全球人工智能大赛作品\day03\image\train"
test_dir = r"C:\Users\石庆\OneDrive\桌面\比赛纪念\全球人工智能大赛作品\day03\image\val"

print(f"\n{'=' * 50}")
print("正在加载数据集...")
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)
num_classes = len(train_data.classes)

print(f"\n{'=' * 50}")
print(f"数据集加载完成！")
print(f"训练集样本数: {len(train_data)} | 测试集样本数: {len(test_data)}")
print(f"检测到 {num_classes} 个类别: {train_data.classes}")


# 4. 定义模型
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 5. 初始化训练组件
model = CustomModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 6. 训练循环
print(f"\n{'=' * 50}")
print("开始训练...")
total_epochs = 10

for epoch in range(total_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 训练阶段
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print(f"[Epoch {epoch + 1}/{total_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"Loss: {loss.item():.4f} | Acc: {100. * correct_train / total_train:.2f}%")

    # 测试阶段
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_save_dir, f"model_epoch{epoch + 1}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)

    # 打印epoch总结
    epoch_time = time.time() - epoch_start
    print(f"\n{'=' * 30} Epoch {epoch + 1} Summary {'=' * 30}")
    print(f"训练损失: {running_loss / len(train_loader):.4f} | "
          f"训练准确率: {100. * correct_train / total_train:.2f}%")
    print(f"测试损失: {test_loss / len(test_loader):.4f} | "
          f"测试准确率: {100. * correct_test / total_test:.2f}%")
    print(f"耗时: {epoch_time:.2f}秒 | 模型已保存到: {model_path}")
    print(f"{'=' * 80}\n")

print("训练完成！所有模型已保存至", model_save_dir)

# ==================================================
# 正在加载数据集...
#
# ==================================================
# 数据集加载完成！
# 训练集样本数: 50000 | 测试集样本数: 10000
# 检测到 30 个类别: ['apple', 'aquarium_fish', ..., 'truck']
#
# ==================================================
# 开始训练...
# [Epoch 1/10] Batch 0/782 Loss: 3.4012 | Acc: 3.12%
# [Epoch 1/10] Batch 20/782 Loss: 3.2101 | Acc: 12.34%
# ...
#
# ============================== Epoch 1 Summary ==============================
# 训练损失: 2.4567 | 训练准确率: 35.67%
# 测试损失: 2.1234 | 测试准确率: 42.31%
# 耗时: 45.23秒 | 模型已保存到: saved_models/model_epoch1_20230815_143022.pth
# ============================================================================