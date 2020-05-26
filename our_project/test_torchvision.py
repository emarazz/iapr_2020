import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from torch import nn
from torch.nn import functional as F
from torch import optim


class NClassifierNet5(nn.Module):
    def __init__(self):

        super(NClassifierNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3) # 28 -> 26 (13) / 14 -> 12
        self.bn1 = nn.BatchNorm2d(num_features = 32) 
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3) # 26 -> 24 (13 -> 11 (5))/ 12 -> 10
        self.bn2 = nn.BatchNorm2d(num_features = 64)
        self.conv3 = nn.Conv2d(64,128, kernel_size = 3) # 24 -> 22 / 10 -> 8
        self.bn3 = nn.BatchNorm2d(num_features = 128)
        self.conv4 = nn.Conv2d(128,256, kernel_size = 3)
        self.bn4 = nn.BatchNorm1d(num_features = 256*2*2)
        self.fc2 = nn.Linear(256*2*2,9)

    def forward(self, xA):

        A = F.relu(F.max_pool2d(self.conv1(xA),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv2(self.bn1(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv3(self.bn2(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv4(self.bn3(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.leaky_relu(self.fc2(self.bn4(A.view(-1, 256*2*2))))
        #print(A.shape)

        return A

## Calls

class IntToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample)

class RandomTranslation(object):
    def __init__(self, new_size):       
        self.new_size = new_size
    def __call__(self, x):
        rand_pads = torch.randint(0,self.new_size[0]-x.size[0]+1,(2,)) 
        paddings = (rand_pads[0], rand_pads[1],
                    self.new_size[0] - rand_pads[0] - x.size[0],
                    self.new_size[1] - rand_pads[1] - x.size[1])
        return TF.pad(x, paddings)

class RandomScale(object):
    def __init__(self, min_max):
        self.min = min_max[0]
        self.max = min_max[1]
    def __call__(self, x):
        rand_scale = (self.max - self.min)*torch.rand(1,) + self.min
        new_height = int((x.size[0]*rand_scale))
        new_width =  new_height
        size = (new_height, new_width)
        return TF.resize(x, size)

class RandomRotation(object):
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        rand_angle = torch.randint(self.angles[0],self.angles[1]+1,(1,))
        return TF.rotate(x, rand_angle)

if __name__ == '__main__':
    rotation = True
    resize = True
    translation = True

    ran_trans = []
    if rotation: ran_trans.append(RandomRotation((0,360)))
    if resize: ran_trans.append(RandomScale((1,2)))
    if translation: ran_trans.append(RandomTranslation((64,64)))
    Random_transformations = transforms.Compose(ran_trans)



    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU available')
    else:
        device = 'cpu'
        print('No GPU available')


    ### testing generator

    test_set = torchvision.datasets.MNIST('./pytorch_mnist',
                                            train = False, # Test set
                                            transform = transforms.Compose([Random_transformations,transforms.ToTensor()]),
                                            target_transform = IntToTensor(),
                                            download=True)
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 6}

    test_noNines = []
    for i, mnist_data in enumerate(test_set):
        if(mnist_data[1]!=9): test_noNines.append(i)

    test_set_noNines = torch.utils.data.Subset(test_set, test_noNines)
    testing_generator = torch.utils.data.DataLoader(test_set_noNines, **params)

    ## Loading weights

    PATH = './collab_noNines_cnn5_TTT_bs1000_ne50.pth'

    eta = 1e-3
    model, criterion = NClassifierNet5(), nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=eta, betas=[0.99,0.999])

    model.to(device)
    criterion.to(device)

    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.to(device)

    # model.train() # In this state for the batch_norm

    correct = 0
    total = 0

    with torch.no_grad():
        model.eval() ## Take care with this!
        for local_batch, local_targets in testing_generator:
            local_batch, local_targets = local_batch.to(device), local_targets.to(device)
            output = model(local_batch)
            _, predicted = torch.max(output, 1)
            total += len(local_targets)
            correct += (predicted == local_targets).sum().item()
            
    print('Accuracy of the network on test images: %0.2f %%' % (
        100 * (correct) / total))
