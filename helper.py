import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image as im
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch

import copy


# SET SEED
seedNum = 3
torch.manual_seed(seedNum)
np.random.seed(seedNum)

# DATA PREPARATION
def get_mnist_data():
    data_train = torchvision.datasets.MNIST(root='./data/fashion_mnist', train=True, download=True)
    data_test = torchvision.datasets.MNIST(root='./data/fashion_mnist', train=False, download=True)
    
    x_train = data_train.train_data.reshape(-1, 1, 28, 28).float() / 255.0
    y_train = data_train.train_labels
    x_test = data_test.test_data.reshape(-1, 1, 28, 28).float() / 255.0
    y_test = data_test.test_labels
    
    classes = {i:c for i,c in enumerate(data_train.classes)}
    return (classes, x_train, y_train, x_test, y_test)

def shuffle_image_array(image,label):
    shuffle_index = np.linspace(0,len(image)-1,len(image)).astype(int)
    np.random.shuffle(shuffle_index)
    
    return image[shuffle_index], label[shuffle_index]

def split_data(images,labels,num_clients,shuffle,data_size=600):
    extra = False
    
    if num_clients>len(images):
        print("Impossible Split!!")
        exit()
        
    if shuffle:
        images, labels = shuffle_image_array(images, labels)
    
    client_list = []
    for i in range(num_clients):
        client_list.append("client_"+str(i))
    
    
    # Nonedefined Datasize
    if data_size==None:
        if(len(images)%num_clients != 0):
            extra_images = len(images)%num_clients
            extra = True
        len_data_per_clients = len(images)//num_clients
    
    # Predefined Datasize
    else:
        len_data_per_clients = data_size
        
    Data_Split_Dict = {} # Client_name: (image,label)
    for index,name in enumerate(client_list): 
        array_split = images[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
        label_split = labels[index*len_data_per_clients:(index*len_data_per_clients)+len_data_per_clients]
        Data_Split_Dict[name] = [array_split,label_split]

    
    client_names = [k for k,v in Data_Split_Dict.items()]
    if extra:
        for i, (image,label) in enumerate(zip(images[-1*extra_images:],labels[-1*extra_images:])):   
            new_data = torch.reshape(image,(-1,image.size()[0],image.size()[1],image.size()[2]))
            Data_Split_Dict[client_names[i%num_clients]][0] = torch.cat((Data_Split_Dict[client_names[i%num_clients]][0],new_data),dim=0)

            label_list = torch.reshape(Data_Split_Dict[client_names[i%num_clients]][1],(-1,1))
            new_label = torch.reshape(label,(-1,1))
            Data_Split_Dict[client_names[i%num_clients]][1] = torch.flatten(torch.cat((label_list,new_label),dim=0))
    
    return client_names,Data_Split_Dict     

def collect_batch(data,label,batch_num,batch_size):
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    
    if batch_num == batch_count and extra != 0:
        batch = (data[batch_num*batch_size:],label[batch_num*batch_size:])
    else:
        batch = (data[batch_num*batch_size:batch_num*batch_size+batch_size],label[batch_num*batch_size:batch_num*batch_size+batch_size])
    if batch_num >= batch_count:
        batch = (-1,-1)
        
    return batch


# MODEL CREATION
class Net(nn.Module):
    def __init__(self,num_class,dim_img):
        super(Net,self).__init__()
        self.flatten_size = dim_img[0]*dim_img[1]*dim_img[2]

        self.fc1 = nn.Linear(self.flatten_size,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,num_class)
        
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)
  
    def forward(self,x):
        # flatten: (batch_size,1,24,24) => (batch_size,576)
        x = x.view(-1, self.flatten_size)

        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = F.relu(self.batchnorm3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return x


# TRAINING
def train_local(model,data,label,client_name,epoch,batch_size):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        print("Cuda Activated")
    
    extra = len(data)-(len(data)//batch_size)*batch_size
    batch_count = len(data)//batch_size if extra == 0 else len(data)//batch_size+1
    print("{} {} training starts!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    #accuracy = 0
    for e in range(epoch):
        print("*",end=" ")
        training_losses = []
        for b in range(batch_count):
            batch_data,batch_label = collect_batch(data,label,b,batch_size)
            
            batch_label = batch_label.type(torch.LongTensor)
            if torch.cuda.is_available():
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs,batch_label)
            
            training_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            #accuracy = torch.sum(torch.max(outputs,dim=1)[1]==batch_label).item() / len(batch_label)

    training_loss = np.mean(training_losses)    
    print("\nLast Epoch!!! \t training loss: {} ".format(training_loss))
    print("{} {} training ends!!".format(client_name.split("_")[0],client_name.split("_")[1]))
    
    return model, training_loss

def validation(model,test_data,test_label):
    val_x,val_y = copy.deepcopy(test_data), copy.deepcopy(test_label)
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        val_x = val_x.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        outputs = model(val_x)

    overall_accuracy = torch.sum(torch.max(outputs.cpu(),dim=1)[1]==val_y).item() / len(val_y)
    preds = torch.max(outputs.cpu(),dim=1)

    outputs,val_y = outputs.cpu(),val_y.cpu()
    loss = criterion(outputs,val_y)
    
    return overall_accuracy,loss,preds,val_y


# SYNCRONIZATION & AGGREGATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def syncronize_with_server_voter(server, client):
    target = {name:value.to(device) for name,value in client.named_parameters()}
    source = {name:value.to(device) for name,value in server.named_parameters()}
    
    for name in target:
        target[name].data = source[name].data.clone()

def federated_averaging(server, clients):
    device = next(server.parameters()).device

    # Cache server parameters ONCE on correct device
    server_params = {
        name: param.data.clone().to(device)
        for name, param in server.named_parameters()
    }

    with torch.no_grad():
        # Initialize update accumulator
        delta = {
            name: torch.zeros_like(param, device=device)
            for name, param in server.named_parameters()
        }

        # Accumulate client updates
        for client_model in clients.values():
            for name, param in client_model.named_parameters():
                delta[name] += (param.data.to(device) - server_params[name])

        # Average updates
        num_clients = len(clients)
        for name in delta:
            delta[name] /= num_clients

        # Apply updates to server
        for name, param in server.named_parameters():
            param.data += delta[name]
