import json
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
from PIL import Image



def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(0.25),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])

    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize( (0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])


    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_transforms)

    class_to_idx = train_dataset.class_to_idx

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    return trainloader, validloader, testloader, class_to_idx


# label mapping between classes and names
def label_mapping(filepath):
    with open(filepath) as f:
        return json.load(f)


# model selection
def model_selection(arch):

    # if arch.lower() == "vgg16":
    #     model = models.vgg16(pretrained=True)
    # elif arch.lower() == "vgg19":
    #     model = models.vgg19(pretrained=True)
    # elif arch.lower() =="alexnet":
    #     model = models.alexnet(pretrained=True)
    # else:
    #     model = models.densenet121(pretrained=True)

    model = getattr(models, arch.lower())(pretrained=True)

    # freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

# custom classifier for pretrained model
class classifier(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer, drop_p):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layer[0])])

        if len(hidden_layer) > 1:
            layers = zip(hidden_layer[:-1], hidden_layer[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])

        self.dropout = nn.Dropout(p=drop_p)

        self.output = nn.Linear(hidden_layer[-1], output_size)

    def forward(self, x):

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


# saving trained model as checkpoint
def save_model(arch, model, optimizer, input_size, output_size, epochs, drop_p, save_dir, learning_rate):

    checkpoint = {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_layer_size': [each.out_features for each in model.classifier.hidden_layers],
        'model_state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'drop_p': drop_p,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'model': arch
    }

    if save_dir:
        filepath = save_dir + '/' + 'checkpoint_{}_{}.pth'.format("_".join(checkpoint['hidden_layers']), checkpoint['model'])
    else:
        filepath = 'checkpoint_{}.pth'.format("_".join([str(each.out_features) for each in model.classifier.hidden_layers]))

    torch.save(checkpoint, filepath)


# loading model from checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = model_selection(checkpoint['model'])

    model.classifier = classifier(checkpoint['input_size'],
                                 checkpoint['output_size'],
                                 checkpoint['hidden_layer_size'],
                                 checkpoint['drop_p'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return model, optimizer


# image preprocessing
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


# show tensor as image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
