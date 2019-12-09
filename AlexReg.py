#!/usr/bin/env python
# coding: utf-8

# In[3]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


data_dir = '/content/gdrive/My Drive/Colab Notebooks/input'
train_dir = data_dir +  '/train_images/'
test_dir = data_dir + '/test_images/'


# In[ ]:


import pandas as pd
labels = pd.read_csv(data_dir + '/train_labels.csv')


# In[5]:


labels.head()


# In[ ]:


print(len(labels))


# In[ ]:


import cv2
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

plt.figure(figsize=[15,15])
i = 1
for img_name in labels['name'][:10]:
    img = cv2.imread(train_dir + img_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(6,5,i)
    plt.imshow(img)
    i += 1
plt.show()


# In[ ]:


labels2 = pd.read_csv(data_dir + '/train_labels.csv')


# In[ ]:


#image.shape torch.Size([3, 256, 256])


# In[ ]:


#labelを0~39にする
labels['label'] -= 1979


# 

# In[ ]:


from sklearn.model_selection import train_test_split
df_train, df_dev = train_test_split(labels, test_size=0.01, random_state=41)


# In[ ]:


df_train.reset_index(inplace=True)
df_dev.reset_index(inplace=True)


# In[ ]:


df_dev.info()


# In[ ]:


labels2.info()


# In[ ]:


from torchvision import transforms
data_transform = {
    'train': transforms.Compose([
                                 transforms.RandomResizedCrop(256),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])                       
    ]),
}


# In[ ]:


import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir =  root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):     
        img_name = self.df.name[index]
        label = self.df.label[index]
        img_path = os.path.join(self.root_dir, img_name)    
        with open(img_path, 'rb') as f:
          image = Image.open(f)
          image = image.convert('RGB')
        if self.transform is not None:
          image = self.transform(image)
        return image, label


# In[ ]:


import torch
import torch.nn as nn
size_check = torch.FloatTensor(10,3,256,256)
features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )


# In[ ]:


fc_size = features(size_check).view(size_check.size(0),-1).size()[1]


# In[ ]:


train_data = CustomDataset(df_train, train_dir, transform = data_transform["train"])
dev_data = CustomDataset(df_dev, train_dir, transform = data_transform["val"])

train_loader = DataLoader(dataset = train_data, batch_size=4, shuffle=True)
dev_loader = DataLoader(dataset = dev_data, batch_size=4, shuffle=False)


# In[ ]:




class AlexNet(nn.Module):
    def __init__(self, num_classes,fc_size):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_size,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1)
            #nn.Linear(4096,num_classes)
        )



    def forward(self,x):
      x = self.features(x)
      x = x.view(x.size(0),-1)
      x = self.classifier(x)
      return x


# In[ ]:


import torch.optim as optim

device = 'cuda'  if torch.cuda.is_available() else 'cpu'
num_classes = 40
net = AlexNet(num_classes,fc_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)
#(0.01,0.1) good
#(0.001,0.9)


# In[15]:


print(device)


# In[ ]:


from tqdm import tqdm
from torch.autograd._functions import Resize

num_epochs = 50

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
  train_loss = 0
  train_acc = 0
  val_loss = 0
  val_acc = 0

  net.train()

  for i, (images, labels) in enumerate(tqdm(train_loader)):
    images,labels = images.to(device),labels.to(device)

    optimizer.zero_grad()
    outputs = net(images)
    outputs = outputs.view(1, -1)
    print(outputs)
    labels = labels.float() /10
    labels = labels.view(1, -1)
    loss = criterion(outputs,labels)
    train_loss += loss.item()
    train_acc += ((outputs - labels).abs() <= 0.1).sum().item()   #0.1
    loss.backward()
    optimizer.step()

  avg_train_loss = train_loss / len(train_loader.dataset)
  avg_train_acc = train_acc / len(train_loader.dataset)

  net.eval()

  with torch.no_grad():
    for images,labels in dev_loader:
      images,labels = images.to(device),labels.to(device)
      outputs = net(images)
      outputs = outputs.view(1, -1)
      print(outputs)
      labels = labels.float() /10
      labels = labels.view(1, -1)
      loss = criterion(outputs,labels)
      val_loss += loss.item()
      val_acc += ((outputs - labels).abs() <= 0.1).sum().item()
 
  avg_val_loss = val_loss / len(dev_loader.dataset)
  avg_val_acc = val_acc / len(dev_loader.dataset)

  print('Epoch [{}/{}], Loss: {loss:.4f},  val_loss: {val_loss:.4f},  val_acc: {val_acc:.4f}'.format(
     epoch+1,num_epochs,i+1,loss=avg_train_loss,val_loss=avg_val_loss,val_acc=avg_val_acc))
  train_loss_list.append(avg_train_loss)
  train_acc_list.append(avg_train_acc)
  val_loss_list.append(avg_val_loss)
  val_acc_list.append(avg_val_acc)
                                                                                                    


# In[ ]:


#state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
torch.save(net.state_dict(),'net.ckpt')


# In[23]:


net = AlexNet(num_classes,fc_size).to(device)
net.load_state_dict(torch.load('net.ckpt'))


# In[ ]:


avg_train_loss


# In[ ]:


predictions = []
test = pd.read_csv(data_dir + '/sample_submission.csv')

test_data = CustomDataset(test, test_dir, transform = data_transform["train"])
test_loader = DataLoader(dataset = test_data, batch_size=1, shuffle=False)

net.eval()

for images,labels in tqdm(test_loader):
  with torch.no_grad():
    images,labels = images.to(device),labels.to(device)
    outputs = (net2(images)*10).round()
  outputs = outputs.data.cpu().numpy()
  s = int(outputs.item())
  print(s)
  predictions.append(s+1979)
test['label'] = predictions
test.to_csv('predictions.csv', index=False)
test.head()


# In[ ]:


predictions.shape


# In[19]:


train_acc_list


# In[20]:


train_loss_list


# In[21]:


val_acc_list


# In[22]:


val_loss_list


# In[ ]:




