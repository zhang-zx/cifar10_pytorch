import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from models import *
from util import progress_bar

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.model_name = config.net
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.hist = dict()
        self.min_loss = 1e8 + 1

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.model_dic = {
            'AlexNet': AlexNet().to(self.device),
        }
        self.model = self.model_dic[self.model_name]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1,
                                                              patience=5,
                                                              verbose=False, threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=1e-5, eps=1e-08)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))
        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        if test_loss < self.min_loss * 0.95:
            torch.save(self.model, "./%s.pth" % (self.model_name))
            self.min_loss = test_loss
            print("Best model saved for {}".format(self.model_name))
        return test_loss, test_correct / total

    def save(self):
        model_out_path = "./model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def process(self, path='./Train.png'):
        x = range(self.epochs)
        y1 = self.hist['Train Accu']
        y2 = self.hist['Test Accu']
        plt.plot(x, y1, label='Train Accu')
        plt.plot(x, y2, label='Test Accu')
        plt.xlabel('Epoch')
        plt.ylabel('Accu')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def draw_learning_rate(self):
        x = range(self.epochs)
        y3 = self.hist['lr']
        plt.plot(x, y3, label='learning rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./%s Learning Rate.png' % (self.model_name))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        hist = dict()
        hist['Train Accu'] = list()
        hist['Test Accu'] = list()
        hist['lr'] = list()
        for epoch in range(1, self.epochs + 1):
            hist['lr'].append(self.optimizer.param_groups[0]['lr'])
            print("\n===> epoch: %d / %d" % (epoch, self.epochs))
            train_result = self.train()
            hist['Train Accu'].append(train_result[1])
            test_result = self.test()
            hist['Test Accu'].append(test_result[1])
            accuracy = max(accuracy, test_result[1])
            self.scheduler.step(test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
        self.hist = hist
        self.process()
        self.draw_learning_rate()


