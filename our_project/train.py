import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--TRAIN', default = True, help = 'If False, it only prints the PATH')
parser.add_argument('--n_epochs', default = 120, help = 'Number of epochs for the training')
parser.add_argument('--batch_size', default = 1000, help = 'Batch size')
parser.add_argument('--eta', default = 1e-3, help = 'Learning rate')


parser.add_argument('--rotation', default = True, help = 'Random rotation in the dataloader')
parser.add_argument('--resize', default = True, help = 'Random resize in the dataloader')
parser.add_argument('--translation', default = True, help = 'Random translation in the dataloader')


args = parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU available')
    else:
        device = 'cpu'
        print('No GPU available')
    return device

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
        A = F.relu(F.max_pool2d(self.conv2(self.bn1(A)),kernel_size = 2, stride = 2))
        A = F.relu(F.max_pool2d(self.conv3(self.bn2(A)),kernel_size = 2, stride = 2))
        A = F.relu(F.max_pool2d(self.conv4(self.bn3(A)),kernel_size = 2, stride = 2))
        A = self.fc2(self.bn4(A.view(-1, 256*2*2)))
        
        return A


###  Custom dataset - our augmented dataset

class MyMNISTDataSet(Dataset):
    """Augmented MNIST dataset - List of tuples - """

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with the dataset - List of tuples (PIL.Image, int).
            transform (callable): Transform to be applied on the PIL.Image.
            target_transform (callable): Transform to be applied on the target or label.
        """
        self.data = torch.load(root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        image = self.data[idx][0]
        label = self.data[idx][1] 

        # sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        #return sample
        return (image, label)

###  Random transformations - Dataloader

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

class IntToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype= torch.long)


if __name__ == '__main__':

    max_epochs = args.n_epochs
    batch_size = args.batch_size
    eta = args.eta

    PATH = './noNines_cnn5_TTT'
    PATH += '_Aug' +'_bs' + str(batch_size) + '_ne' + str(max_epochs) + '.pth'

    rotation = args.rotation
    resize = args.resize
    translation = args.translation

    TRAIN = args.TRAIN

    # Data augmentation

    ran_trans = []

    if rotation: ran_trans.append(RandomRotation((0,360)))
    if resize: ran_trans.append(RandomScale((1.,2.)))
    if translation: ran_trans.append(RandomTranslation((64,64)))

    Random_transformations = transforms.Compose(ran_trans)

    # Check device 
    device = get_device()

    # Model and criterion declaration
    
    model, criterion = NClassifierNet5(), nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=eta, betas=[0.99,0.999])

    model.to(device)
    criterion.to(device)

    
    # Dataloaders

    train_set = MyMNISTDataSet('./my_nmist_list_tuples.pt',
                                transform = transforms.Compose([Random_transformations,transforms.ToTensor()]),
                                target_transform = IntToTensor())

    train_noNines = []
    for i, mnist_data in enumerate(train_set):
        if(mnist_data[1]!=9):
            train_noNines.append(i)

    params = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 6}

    train_set_noNines = torch.utils.data.Subset(train_set, train_noNines)
    train_set_noNines, val_set_noNines = torch.utils.data.random_split(train_set_noNines,
                                                                        [len(train_set_noNines) - len(train_set_noNines)//100, len(train_set_noNines)//100])

    training_generator = torch.utils.data.DataLoader(train_set_noNines,
                                                    **params)
    params2 = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 6}
    validation_generator = torch.utils.data.DataLoader(val_set_noNines,
                                                 **params2)

    ## Train

    if TRAIN:
        for epoch in range(max_epochs):
            sum_loss, total, correct = 0,0,0
            model.train()
            for local_batch, local_targets in training_generator:
                local_batch, local_targets = local_batch.to(device), local_targets.to(device)
                output = model(local_batch)
                loss = criterion(output, local_targets)
                model.zero_grad()
                loss.backward()
                _, predicted = torch.max(output, 1)
                correct += (predicted == local_targets).sum().item()
                total += len(local_targets)
                sum_loss += loss.item()
                optimizer.step()
            print('Epoch: %i - train_acc = %0.4f %%' % (epoch, 100 * (correct) / total), end = '')
            
            sum_loss, total, correct = 0,0,0
            with torch.set_grad_enabled(False):
                model.eval()
                for local_batch, local_targets in validation_generator:
                    local_batch, local_targets = local_batch.to(device), local_targets.to(device)
                    output = model(local_batch)
                    loss = criterion(output, local_targets)
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == local_targets).sum().item()
                    total += len(local_targets)
                    sum_loss += loss.item()
                print(' - validation_acc =%0.4f %%' % (100 * (correct) / total))
            
            if (int((epoch+1)%10) == 0 ):
                print('saving epoch: ', epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)

    print('Done!')
    print('PATH = ',PATH)
    # test_set = torchvision.datasets.MNIST('./pytorch_mnist',
    #                                         train = False, # Test set
    #                                         transform = transforms.Compose([Random_transformations,transforms.ToTensor()]),
    #                                         target_transform = IntToTensor(),
    #                                         download=True)

    # params3 = {'batch_size': 1,
    #         'shuffle': False,
    #         'num_workers': 6}

    # test_noNines = []
    # for i, mnist_data in enumerate(test_set):
    #     if(mnist_data[1]!=9):
    #         test_noNines.append(i)

    # test_set_noNines = torch.utils.data.Subset(test_set, test_noNines)
    # testing_generator = torch.utils.data.DataLoader(test_set_noNines,
    #                                                 **params3)