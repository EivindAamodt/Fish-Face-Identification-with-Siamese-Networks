import torch
import torchvision
import torchvision.transforms as transforms
import time

from customDataset import FoodDataset

tic = time.perf_counter()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = FoodDataset(csv_file = 'training.csv', root_dir = 'training', transform=transform)
testset = FoodDataset(csv_file = 'validation.csv', root_dir = 'validation', transform=transform)


#trainset, testset = torch.utils.data.random_split(dataset, [len(dataset)-2000, 2000])
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=16, shuffle=True)


classes = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
    'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit')


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img,s=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if s and not '\n' in s:
        s = ' '.join(s.split())
        p = s.find(' ',int(len(s)/2))
        s = s[:p]+"\n"+s[p+1:]
    plt.text(0,-20,s)
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
#import pdb;pdb.set_trace()
# show images
s = ' '.join('%5s' % classes[labels[j]] for j in range(16))
print(s)
imshow(torchvision.utils.make_grid(images),s)
#print(' '.join('%5s' % classes[labels[j]] for j in range(16)))
#imshow(torchvision.utils.make_grid(images),labels)
# print labels


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 69 * 69, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 69 * 69)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

accuracyepoch = []

epochs = 25

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()



        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            tak = time.perf_counter()

            #print(f"Finished training and validation in {tak - tic:0.4f} seconds")

            print('[%d, %5d] loss: %.3f, total time: %.3f seconds' %
                  (epoch + 1, i + 1, running_loss / 100, tak-tic))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(torch.float32)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the 3430 validation images in epoch %d: %d %%' % (
            epoch+1, 100 * correct / total))
    accuracyepoch.append(100 * correct / total)

print('Finished Training')


dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.to(torch.float32)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))


outputs = net(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(16)))


print('Accuracy of the network on the 3430 validation images:', accuracyepoch[-1])

# graph total accuracy epochs

tickpos = []

for i in range(epochs):
    tickpos.append(i+1)

plt.plot(accuracyepoch, label = "accuracy in epoch x")


plt.xticks(tickpos,tickpos)

#plt.grid(axis='y')

plt.ylim((0,100))

# Set the x axis label of the current axis.
plt.xlabel('epoch')
# Set the y axis label of the current axis.
plt.ylabel('accuracy in %')
# Set a title of the current axes.
plt.title('accuracy on validation set (3484 images)')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()





class_correct = list(0. for i in range(11))
class_total = list(0. for i in range(11))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(torch.float32)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(11):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

toc = time.perf_counter()

print(f"Finished training and validation in {toc - tic:0.4f} seconds")
