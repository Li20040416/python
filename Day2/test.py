import torch
import torchvision
from PIL import Image
from model import *
from torchvision import transforms

image_path = "Image/real.png"
image = Image.open(image_path)
print(image)
# png格式是四个通道，除了RGB三个通道外，还有一个透明通道,调用image = image.convert('RGB')
image = image.convert('RGB')

# 改变成tensor格式
trans_reszie = transforms.Resize((32, 32))
trans_totensor = transforms.ToTensor()
transform = transforms.Compose([trans_reszie, trans_totensor])
image = transform(image)
print(image.shape)

# 加载训练模型
model = torch.load("model_save\\chen_3.pth", map_location="cpu")

# print(model)

image = torch.reshape(image, (1, 3, 32, 32))
# image = image.cuda()

# 将模型转换为测试模型
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))

#测试图片fruit.png
#torch.Size([3, 32, 32])
#tensor([7])