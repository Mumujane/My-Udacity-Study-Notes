#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports here
import json
import time
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb

import torch
from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets, transforms, models


# # 开发 AI 应用
# 
# 未来，AI 算法在日常生活中的应用将越来越广泛。例如，你可能想要在智能手机应用中包含图像分类器。为此，在整个应用架构中，你将使用一个用成百上千个图像训练过的深度学习模型。未来的软件开发很大一部分将是使用这些模型作为应用的常用部分。
# 
# 在此项目中，你将训练一个图像分类器来识别不同的花卉品种。可以想象有这么一款手机应用，当你对着花卉拍摄时，它能够告诉你这朵花的名称。在实际操作中，你会训练此分类器，然后导出它以用在你的应用中。我们将使用[此数据集](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)，其中包含 102 个花卉类别。你可以在下面查看几个示例。 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# 该项目分为多个步骤：
# 
# * 加载和预处理图像数据集
# * 用数据集训练图像分类器
# * 使用训练的分类器预测图像内容
# 
# 我们将指导你完成每一步，你将用 Python 实现这些步骤。
# 
# 完成此项目后，你将拥有一个可以用任何带标签图像的数据集进行训练的应用。你的网络将学习花卉，并成为一个命令行应用。但是，你对新技能的应用取决于你的想象力和构建数据集的精力。例如，想象有一款应用能够拍摄汽车，告诉你汽车的制造商和型号，然后查询关于该汽车的信息。构建你自己的数据集并开发一款新型应用吧。
# 
# 首先，导入你所需的软件包。建议在代码开头导入所有软件包。当你创建此 notebook 时，如果发现你需要导入某个软件包，确保在开头导入该软件包。

# ## 加载数据
# 
# 在此项目中，你将使用 `torchvision` 加载数据（[文档](http://pytorch.org/docs/master/torchvision/transforms.html#)）。数据应该和此 notebook 一起包含在内，否则你可以[在此处下载数据](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)。数据集分成了三部分：训练集、验证集和测试集。对于训练集，你需要变换数据，例如随机缩放、剪裁和翻转。这样有助于网络泛化，并带来更好的效果。你还需要确保将输入数据的大小调整为 224x224 像素，因为预训练的网络需要这么做。
# 
# 验证集和测试集用于衡量模型对尚未见过的数据的预测效果。对此步骤，你不需要进行任何缩放或旋转变换，但是需要将图像剪裁到合适的大小。
# 
# 对于所有三个数据集，你都需要将均值和标准差标准化到网络期望的结果。均值为 `[0.485, 0.456, 0.406]`，标准差为 `[0.229, 0.224, 0.225]`。这样使得每个颜色通道的值位于 -1 到 1 之间，而不是 0 到 1 之间。

# In[3]:


data_dir = 'flowers'
# 训练集，，需要进行变换数据，比如剪裁图片、随机缩放和对图片进行翻转。
train_dir = data_dir + '/train'
# 验证集， 用来衡量模型对未知数据的预测效果。
valid_dir = data_dir + '/valid'
# 测试集， 只需对图像大小进行核算的裁剪，不需要对其中的图像作任何缩放或选中变换。
test_dir = data_dir + '/test'


# In[4]:


# 此处用来定义训练集、验证集和测试集的转换方式
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(0.25),
                                     transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize( (0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])
valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(0.25),
                                     transforms.ToTensor(),
                                     transforms.Normalize( (0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])
test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(0.25),
                                     transforms.ToTensor(),
                                     transforms.Normalize( (0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)


# ### 标签映射
# 
# 你还需要加载从类别标签到类别名称的映射。你可以在文件 `cat_to_name.json` 中找到此映射。它是一个 JSON 对象，可以使用 [`json` 模块](https://docs.python.org/2/library/json.html)读取它。这样可以获得一个从整数编码的类别到实际花卉名称的映射字典。

# In[5]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# 
# # 构建和训练分类器
# 
# 数据准备好后，就开始构建和训练分类器了。和往常一样，你应该使用 `torchvision.models` 中的某个预训练模型获取图像特征。使用这些特征构建和训练新的前馈分类器。
# 
# 这部分将由你来完成。如果你想与他人讨论这部分，欢迎与你的同学讨论！你还可以在论坛上提问或在工作时间内咨询我们的课程经理和助教导师。
# 
# 请参阅[审阅标准](https://review.udacity.com/#!/rubrics/1663/view)，了解如何成功地完成此部分。你需要执行以下操作：
# 
# * 加载[预训练的网络](http://pytorch.org/docs/master/torchvision/models.html)（如果你需要一个起点，推荐使用 VGG 网络，它简单易用）
# * 使用 ReLU 激活函数和丢弃定义新的未训练前馈网络作为分类器
# * 使用反向传播训练分类器层，并使用预训练的网络获取特征
# * 跟踪验证集的损失和准确率，以确定最佳超参数
# 
# 我们在下面为你留了一个空的单元格，但是你可以使用多个单元格。建议将问题拆分为更小的部分，并单独运行。检查确保每部分都达到预期效果，然后再完成下个部分。你可能会发现，当你实现每部分时，可能需要回去修改之前的代码，这很正常！
# 
# 训练时，确保仅更新前馈网络的权重。如果一切构建正确的话，验证准确率应该能够超过 70%。确保尝试不同的超参数（学习速率、分类器中的单元、周期等），寻找最佳模型。保存这些超参数并用作项目下个部分的默认值。

# In[6]:


# TODO: Build and train your network
# 设置GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1: 构建网络
model = models.densenet121(pretrained=True)


# In[7]:


# 设置参数
for param in model.parameters():
    param.requires_grad= False

# 分类器超参数
input_size = 1024
hidden_layers = [512]
output_size = 102
drop_p = 0.5
epochs = 30

# 添加自己的分类器
class classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        if len(hidden_layers) != 0:
            layers = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])
        
        self.dropout = nn.Dropout(p=drop_p)
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

model.classifier = classifier(input_size, output_size, hidden_layers, drop_p)
model.to(device)


# In[8]:


# 验证试验
def validation(model, criterion, validloader):
    loss = 0
    accuracy = 0
    for images, targets in iter(validloader):
        images, targets = images.to(device), targets.to(device)

        output = model.forward(images)
        loss += criterion(output, targets).item()
        
        ps = torch.exp(output)
        equality = (targets.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return loss, accuracy


# In[9]:


# 2: 训练网络
# criterion & optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# In[10]:


# 定义训练模型
def train(model, trainloader, validloader, criterion, optimizer, print_step=32, epochs=30):
    
    # training
    steps = 0
    running_loss = 0
    print_step = 32
    
    # validation result variables
    vloss = 0
    vaccuracy = 0
    
    start = time.time()
    
    for e in range(epochs):
        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps%print_step == 0:

                model.eval()

                changed = False

                if steps%(10*print_step) == 0:
                    with torch.no_grad():
                        vloss, vaccuracy = validation(model, criterion, validloader)
                        changed = True

                end=time.time()

                change = "(c)" if changed else ""

                print('Epoch: {}/{}\t'.format(e+1, epochs),
                     'Train Loss: {:.3f}\t'.format(running_loss/print_step),
                     'Valid Loss: {:.3f}\t'.format(vloss/len(validloader)),
                     'Valid Accuracy: {:.3f}\t'.format(vaccuracy/len(validloader)*100),
                     'Step Time: {:.2f}  {}'.format(end-start, change))
                start = end
                running_loss = 0

                model.train()


# In[11]:


# 进行训练
train(model.to(device), trainloader, testloader, criterion, optimizer, print_step=32, epochs=30)


# ## 测试网络
# 
# 建议使用网络在训练或验证过程中从未见过的测试数据测试训练的网络。这样，可以很好地判断模型预测全新图像的效果。用网络预测测试图像，并测量准确率，就像验证过程一样。如果模型训练良好的话，你应该能够达到大约 70% 的准确率。

# In[12]:


# TODO: Do validation on the test set
# 定义测试网络
def test(model, criterion, testloader):
    model.eval()

    with torch.no_grad():
        tloss, taccuracy = validation(model, criterion, testloader)

    print('Test Accuracy: {:.3f}'.format(taccuracy/len(testloader)*100))

    model.train()


# In[13]:


# 进行测试
test(model.to(device), criterion, testloader)


# ## 保存检查点
# 
# 训练好网络后，保存模型，以便稍后加载它并进行预测。你可能还需要保存其他内容，例如从类别到索引的映射，索引是从某个图像数据集中获取的：`image_datasets['train'].class_to_idx`。你可以将其作为属性附加到模型上，这样稍后推理会更轻松。

# In[14]:



# 注意，稍后你需要完全重新构建模型，以便用模型进行推理。确保在检查点中包含你所需的任何信息。如果你想加载模型并继续训练，则需要保存周期数量和优化器状态 `optimizer.state_dict`。你可能需要在下面的下个部分使用训练的模型，因此建议立即保存它。


# ```python
# TODO: Save the checkpoint  
# 保存检查点数据
checkpoint = {
    'input_size': input_size,
    'output_size': output_size,
    'hidden_layer_size': [each.out_features for each in model.classifier.hidden_layers],
    'drop_p': drop_p,
    'epochs': epochs,
    'optimizer_state': optimizer.state_dict(),
    'model_state_dict': model.state_dict(),
    'model': 'densenet121',
    'class_to_idx': train_datasets.class_to_idx
}

torch.save(checkpoint, 
           'checkpoint_{}_bs=64_2.pth'.format(
               "_".join([str(each.out_features) for each in  model.classifier.hidden_layers])))


# ## 加载检查点
# 
# 此刻，建议写一个可以加载检查点并重新构建模型的函数。这样的话，你可以回到此项目并继续完善它，而不用重新训练网络。

# In[15]:




# TODO: Write a function that loads a checkpoint and rebuilds the model、
# 加载检查点数据
def load_checkpoint(filepath, drop_p=None):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier(checkpoint['input_size'],
                                 checkpoint['output_size'],
                                 checkpoint['hidden_layer_size'],
                                 drop_p if drop_p else checkpoint['drop_p'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    return model, optimizer


# 
# # 类别推理
# 
# 现在，你需要写一个使用训练的网络进行推理的函数。即你将向网络中传入一个图像，并预测图像中的花卉类别。写一个叫做 `predict` 的函数，该函数会接受图像和模型，然后返回概率在前 $K$ 的类别及其概率。应该如下所示：

# In[19]:


probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']


# 首先，你需要处理输入图像，使其可以用于你的网络。
# 
# ## 图像处理
# 
# 你需要使用 `PIL` 加载图像（[文档](https://pillow.readthedocs.io/en/latest/reference/Image.html)）。建议写一个函数来处理图像，使图像可以作为模型的输入。该函数应该按照训练的相同方式处理图像。
# 
# 首先，调整图像大小，使最小的边为 256 像素，并保持宽高比。为此，可以使用 [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 或 [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 方法。然后，你需要从图像的中心裁剪出 224x224 的部分。
# 
# 图像的颜色通道通常编码为整数 0-255，但是该模型要求值为浮点数 0-1。你需要变换值。使用 Numpy 数组最简单，你可以从 PIL 图像中获取，例如 `np_image = np.array(pil_image)`。
# 
# 和之前一样，网络要求图像按照特定的方式标准化。均值应标准化为 `[0.485, 0.456, 0.406]`，标准差应标准化为 `[0.229, 0.224, 0.225]`。你需要用每个颜色通道减去均值，然后除以标准差。
# 
# 最后，PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度。你可以使用 [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html)对维度重新排序。颜色通道必须是第一个维度，并保持另外两个维度的顺序。

# In[22]:


# 进行图像处理
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    tsize = (256, 256)
    img.thumbnail(tsize)

    lwsize = (img.size[0] - 224)/2
    thsize = (img.size[1] - 224)/2
    rwsize = (img.size[0] + 224)/2
    bhsize = (img.size[1] + 224)/2
    
    img = img.crop((lwsize, thsize, rwsize, bhsize))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    npimg = np.array(img)
    npimg = npimg/255
    
    npimg = (npimg - mean)/std
    
    npimg = npimg.transpose((2,0,1))

    return torch.from_numpy(npimg)


# 要检查你的项目，可以使用以下函数来转换 PyTorch 张量并将其显示在  notebook 中。如果 `process_image` 函数可行，用该函数运行输出应该会返回原始图像（但是剪裁掉的部分除外）。

# In[23]:


# 图像大小
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # pyTorch tensors中颜色通道为一维 matplotlib中为三维
    image = image.numpy().transpose((1, 2, 0))
    
    # 撤消预处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # 图像剪裁范围为区间（0，1），或者当其显示时看起来像噪声时也需要剪裁
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## 类别预测
# 
# 可以获得格式正确的图像后 
# 
# 要获得前 $K$ 个值，在张量中使用 [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk)。该函数会返回前 `k` 个概率和对应的类别索引。你需要使用  `class_to_idx`（希望你将其添加到了模型中）将这些索引转换为实际类别标签，或者从用来加载数据的[ `ImageFolder`](https://pytorch.org/docs/master/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)进行转换。确保颠倒字典
# 
# 同样，此方法应该接受图像路径和模型检查点，并返回概率和类别。

# In[ ]:


probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']


# In[24]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    
    imaget = process_image(image_path)
    imaget.to(device)
    
    imaget = imaget.unsqueeze(0)
    image = imaget.type(torch.cuda.FloatTensor)
    output = model.forward(image)
    ps = torch.exp(output)
    model.train()
    return ps.topk(topk)


# In[25]:


model, optimizer = load_checkpoint('checkpoint_512_bs=64_2.pth')


# ## 检查运行状况
# 
# 你已经可以使用训练的模型做出预测，现在检查模型的性能如何。即使测试准确率很高，始终有必要检查是否存在明显的错误。使用 `matplotlib` 将前 5 个类别的概率以及输入图像绘制为条形图，应该如下所示：
# 
# <img src='assets/inference_example.png' width=300px>
# 
# 你可以使用 `cat_to_name.json` 文件（应该之前已经在 notebook 中加载该文件）将类别整数编码转换为实际花卉名称。要将 PyTorch 张量显示为图像，请使用定义如下的 `imshow` 函数。

# In[26]:


# TODO: Display an image along with the top 5 classes
image_path = 'flowers/test/21/image_06807.jpg'

probs, labels = predict(image_path, model)

ps = [x for x in probs.cpu().detach().numpy()[0]]
npar = [x for x in labels.cpu().numpy()[0]]
names = list()

inv_mapping = {v: k for k, v in model.class_to_idx.items()}

for i in npar:
    names.append(cat_to_name[str(inv_mapping[i])])


imshow(process_image(image_path), ax=plt.subplot(2,1,1));
plt.title(cat_to_name['21'])

plt.subplot(2,1,2)
sb.barplot(y=names, x=ps, color=sb.color_palette()[0]);


# In[ ]:




