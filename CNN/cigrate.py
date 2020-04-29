import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import sys
import os

import torch.nn.functional as F
from tensorboardX import SummaryWriter


torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.insert(0, './src/')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.cuda.current_device()
torch.cuda._initialized = True


writer = SummaryWriter(log_dir='/home/baiyang/liyazhao/image/')


EPOCH = 1500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_transform = transforms.Compose([transforms.Resize((64,64)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])


train_dataset = datasets.ImageFolder(root='/home/baiyang/liyazhao/train/', transform=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.ImageFolder(root='/home/baiyang/liyazhao/test/', transform=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64,shuffle=True)


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.line_layer_conv1 = nn.Conv2d(3, 32, 3)
        self.line_layer_conv2 = nn.Conv2d(32, 64, 3)
        self.line_layer_conv3 = nn.Conv2d(64, 128, 3)
        self.line_layer_conv4 = nn.Conv2d(128, 256, 3)

        self.upper_layer_conv1 = nn.Conv2d(32, 128, 7)
        self.upper_layer_conv2 = nn.Conv2d(128, 256, 5)

        self.lower_layer_conv1 = nn.Conv2d(3, 32, 4)
        self.lower_layer_conv2 = nn.Conv2d(32, 64, 2)

        self.fc1 = nn.Linear(2*2*256, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)
        # 设置前置网络及 可学习参数
        #self.cnn = cnn_output4()
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # 初始化
        self.fuse_weight_1.data.fill_(0.25)
        self.fuse_weight_2.data.fill_(0.25)
        self.fuse_weight_3.data.fill_(0.25)
        self.fuse_weight_4.data.fill_(0.25)

        def forward(x):
            x1, x2, x3, x4 = self.cnn(x)
            return fuse_weight_1 * x1 + fuse_weight_2 * x2 + fuse_weight_3 * x3 + fuse_weight_4 * x4

    def forward(self, x):
        in_size = x.size(0)
        lower_x = x

        lower_x = self.lower_layer_conv1(lower_x)
        lower_x = F.relu(lower_x)
        lower_x = F.max_pool2d(lower_x, 4, padding=0)
        lower_x = self.lower_layer_conv2(lower_x)
        lower_x = F.relu(lower_x)

        out = self.line_layer_conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        upper_layer = out
        upper_layer = self.upper_layer_conv1(upper_layer)
        #upper_layer = F.dropout(upper_layer, 0.5, training=self.training)
        upper_layer = F.relu(upper_layer)
        upper_layer = F.max_pool2d(upper_layer, 4, padding=0)
        upper_layer = self.upper_layer_conv2(upper_layer)

        out = self.line_layer_conv2(out)
        out= F.relu(out)
        out = F.max_pool2d(out, 2, padding=0)

        out = out + lower_x

        # bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False).to(DEVICE)
        # out = bn(out)

        out = self.line_layer_conv3(out)

        out = F.relu(out)

        out = F.max_pool2d(out, 2, padding=0)

        out = self.line_layer_conv4(out)

        out = F.dropout(out, 0.6, training=self.training)

        out = F.relu(out)

        out = F.max_pool2d(out, 2, padding=0)

        out = out + upper_layer

        out = out.view(in_size, -1)

        out = self.fc1(out)

        out = F.relu(out)

        out = F.dropout(out, 0.6, training=self.training)

        out = self.fc2(out)

        out = F.relu(out)

        out = self.fc3(out)

        out= F.relu(out)

        out = torch.log_softmax(out, dim=1)

        return out


#optimizer = optim.Adam(Net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
optimizer = optim.SGD(Net.parameters(), lr=0.006)
# print(1e-4)
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        output = torch.max(output.data, 1)[1].float().requires_grad_(True)

        train_pred = output.view(-1)

        train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_avg_loss = train_loss / len(train_dataloader)
    train_avg_pre = 100. * train_correct / len(train_dataset)
    print('\nTrain set: Average loss: {:.4f}, '
          'Training Average Precision: {}/{} ({:.2f}%)'.format(train_avg_loss, train_correct,
                                                                   len(train_dataset), train_avg_pre))

    writer.add_scalar('train_loss', train_avg_loss, epoch)
    writer.add_scalar('train_precision', train_avg_pre, epoch)
    writer.add_graph(model, (data,))
    writer.flush()

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target)

            output = torch.max(output.data, 1)[1].float()

            test_pred = output.view(-1)

            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))


    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('precision', 100. * test_correct / len(test_loader.dataset), epoch)
    writer.flush()



for epoch in range(1, EPOCH+1):
    start_time = time.time()
    train(Net, DEVICE, train_dataloader, optimizer, epoch)
    test(Net, DEVICE, test_dataloader)
    end_time = time.time()
    print('这一轮训练时间为：{:.2f}\n'.format(end_time - start_time))

