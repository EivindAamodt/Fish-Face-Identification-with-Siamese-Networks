
#
# inspired by
# https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
# https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
# https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
#


import matplotlib
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

tic = time.time()

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


class Config():
    # human face test
    #training_dir = "./data/faces/training/"
    #testing_dir = "./data/faces/testing/"

    # face

    #training_dir = "data/wrasse/training_head_sameway/"
    #testing_dir = "data/wrasse/testing_head_sameway/"

    #training_dir = "data/wrasse/training_head_bothways/"
    #testing_dir = "data/wrasse/testing_head_bothways/"

    # body

    training_dir = "data/wrasse/training_body_sameway/"
    testing_dir = "data/wrasse/testing_body_sameway/"

    #training_dir = "data/wrasse/training_body_bothways/"
    #testing_dir = "data/wrasse/testing_body_bothways/"


    train_batch_size = 64
    train_number_epochs = 100


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L") # L - greyscale, RGB - true color, CMYK - pre-press images
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)



folder_dataset = dset.ImageFolder(root=Config.training_dir)


siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)


vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# train

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = []
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter,loss_history)


#test

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)


# test1

#test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
#dataiter = iter(test_dataloader)
#x0, _, _ = next(dataiter)
#
#for i in range(10):
#    _, x1, label2 = next(dataiter)
#   concatenated = torch.cat((x0, x1), 0)
#
#    output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
#    euclidean_distance = F.pairwise_distance(output1, output2)
#    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))






# test2

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)

#for graph at end
# accuracy, 1 = total, 2 = same fish, 3 = different fish
xx1 = []
yy1 = []

xx2 = []
yy2 = []

xx3 = []
yy3 = []

tickpos = []


for j in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9]:
    dataiter = iter(test_dataloader)
    correct = 0
    total = 0
    predict = 0.0

    totalsame = 0
    totaldifferent = 0
    correctsame = 0
    correctdifferent = 0

    showpictures = 0

    for i in range(len(dataiter)):
        x0, x1, label1 = next(dataiter)

        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        if euclidean_distance.item() < j:
            predict = 0.0
        else:
            predict = 1.0
        if showpictures < 2:
            imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}, predicted: {:.1f}, truth: {:.1f}'.format(euclidean_distance.item(), predict, label1.item()))
            showpictures += 1

        #print("predicted :", predict, "truth:", label1.item())
        #print(label1.item())

        if label1.item() == 0.0:
            totalsame += 1
        else:
            totaldifferent += 1

        if predict == label1.item():
            correct += 1
            if label1.item() == 0.0:
                correctsame += 1
            else:
                correctdifferent += 1
        total += 1

    print("distance threshold:", j)
    print("total accuracy: ", correct/total)
    print("same fish accuracy:", correctsame/totalsame)
    print("different fish accuracy:", correctdifferent/totaldifferent)
    print()

    # for plotting

    tickpos.append(j)

    xx1.append(j)
    yy1.append(correct/total*100)

    xx2.append(j)
    yy2.append(correctsame/totalsame*100)

    xx3.append(j)
    yy3.append(correctdifferent/totaldifferent*100)


toc = time.time()

print('time taken: {:.2f} seconds'.format(toc-tic))

import matplotlib.pyplot as plt

# line 1 points

plt.plot(xx1, yy1, label="Total accuracy")

plt.plot(xx2, yy2, label="Same fish accuracy")

plt.plot(xx3, yy3, label="Different fish accuracy")

plt.xticks(tickpos,tickpos)

plt.grid(axis='y')

plt.ylim(-1, 101)

# Set the x axis label of the current axis.
plt.xlabel('Distance thresholds')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy in percent')
# Set a title of the current axes.
plt.title('Accuracies with different distance thresholds - body SW')



# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
