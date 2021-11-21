
# coding: utf-8

# # Convolutional Neural Networks (2)

# ## Dataset

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import time


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # datasets and pretrained neural nets
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.48216, 0.44653),
                                                     (0.24703, 0.24349, 0.26159))])

trainset = torchvision.datasets.CIFAR10(root='/Users/Lxy/Desktop/PyTorch/data', train=True, 
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/Users/Lxy/Desktop/PyTorch/data', train=False, 
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, 
                                         shuffle=False, num_workers=4)


# In[3]:


trainset.data.shape


# ## Building convolutional network

# In[4]:


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # 32-3+2+1=32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 16 x 16
        x = self.pool(F.relu(self.conv2(x))) # 8 x 8
        x = self.pool(F.relu(self.conv3(x)))  #  4 x 4    feature map C x W x H --> linear.   
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)  # (batch size, num_classes)


# In[5]:


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)


# In[6]:


net


# ## Training

# In[10]:


num_epochs = 10

since = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels) 
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / trainloader.dataset.data.shape[0]
    epoch_acc = running_corrects.double() / trainloader.dataset.data.shape[0]
    
    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    print('-' * 10)
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


# ## Evaluation

# fc acc: 53%

# In[11]:


correct, total = 0, 0
predictions = []
net.eval()  # mode:1.train(), 2.eval()
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1) 
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % ( 100 * correct / total))


# ## Add Validation

# In[13]:


indices = np.arange(50000) 
indices


# In[26]:


np.random.shuffle(indices)
indices


# In[12]:


import numpy as np

indices = np.arange(50000) 
np.random.shuffle(indices)
indices


# In[14]:


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        root='/Users/Lxy/Desktop/PyTorch/data', 
        train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), 
                                                           (0.229, 0.224, 0.225))])), 
    batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:45000]))

val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='/Users/Lxy/Desktop/PyTorch/data', 
                     train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406), 
                                                                        (0.229, 0.224, 0.225))])), 
    batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[45000:50000]))


# ## Save and Load models

# In[22]:


net


# `state_dict()`

# In[8]:


print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())


# In[9]:


optimizer.state_dict()['param_groups']


# ### Saving & Loading Model for Inference
# **Save**

# In[15]:


PATH = './models/model_torch.pth'
torch.save(net.state_dict(), PATH)


# **Load**

# In[16]:


model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()


# The 1.6 release of PyTorch switched torch.save to use a new zipfile-based file format. torch.load still retains the ability to load files in the old format. If for any reason you want torch.save to use the old format, pass the kwarg `_use_new_zipfile_serialization=False`.
# 
# ### Save/Load Entire Model
# **Save**

# In[21]:


torch.save(model, PATH)


# **Load**

# In[24]:


abd = torch.load(PATH)
abd.eval()


# ### Saving & Loading a General Checkpoint for Inference and/or Resuming Training
# **Save**

# In[31]:


PATH_CHECKPOINT = './models/checkpoint.tar'

torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH_CHECKPOINT)


# **Load**

# In[32]:


model = Net()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

checkpoint = torch.load(PATH_CHECKPOINT)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()


# ### Saving Multiple Models in One File
# **Save**

# In[ ]:


torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)


# **Load**

# In[ ]:


modelA = ModelA(*args, **kwargs)
modelB = ModelB(*args, **kwargs)
optimizerA = OptimizerA(*args, **kwargs)
optimizerB = OptimizerB(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()


# ## Arch
# AlexNet

# In[56]:


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000): 
        super(AlexNet, self).__init__() 
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(96, 256, kernel_size=5, padding=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # 最终输的size
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(256 * 6 * 6, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(), 
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace=True), 
            nn.Linear(4096, num_classes))
            
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6) 
        x = self.classifier(x)
        return x


# In[74]:


alex = AlexNet()
alex


# VGG

# In[58]:


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())


# In[59]:


test()


# In[61]:


vgg_16 = VGG('VGG16')
vgg_16


# ## Transfer learning
# 

# ### Freeze

# In[63]:


alex


# In[64]:


alex.classifier


# In[75]:


# Freeze all the layers bar the final one
for param, name in zip(alex.parameters(), alex.state_dict().keys()): 
    s = param.requires_grad
    param.requires_grad = False
    print(name,":", s, param.requires_grad)


# In[78]:


alex.classifier[6] = nn.Linear(4096, 10)
for param, name in zip(alex.parameters(), alex.state_dict().keys()): 
    print(name,":", param.requires_grad)


# In[79]:


alex


# ### Framework

# In[81]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# **Load Data**

# In[ ]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/Users/Lxy/Desktop/PyTorch/data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[84]:


image_datasets


# In[85]:


dataset_sizes
class_names


# **Visualize**

# In[86]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# **Training function**
# * Scheduling the learning rate
# * Saving the best model

# In[87]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# **Finetuning the convnet**
# 
# Load a pretrained model and reset final fully connected layer.

# In[108]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[109]:


model_ft


# **Train and evaluate**

# In[110]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)


# **ConvNet as fixed feature extractor**
# 
# Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().

# In[ ]:


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
    
# Parameters of newly sconstructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_con = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_con, step_size=7, gamma=0.1)


# In[ ]:


model_conv = train_model(model_conv, criterion, optimizer_con,
                                      exp_lr_scheduler, num_epochs=25)

