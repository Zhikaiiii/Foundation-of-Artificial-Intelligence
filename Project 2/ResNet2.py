import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import torch.utils.data as Data
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

#数据处理 读取数据
test_npy = np.load('test.npy')
train_npy = np.load('train.npy')
num_train = train_npy.shape[0]
num_test = test_npy.shape[0]
#test_pic = np.empty([num_test,28,28], dtype= float)
test_pic = np.reshape(test_npy,(num_test,28,28))
print(len(train_npy))
train_num = len(train_npy)
test_num = len(test_npy)
train_npy = np.reshape(train_npy, (num_train,28, 28))
train_all = []
test_img = []
for i in range(train_num):
    train_all.append(Image.fromarray(train_npy[i,...]))
    train_all[i] = train_all[i].resize((112,112))
for j in range(test_num):
    test_img.append(Image.fromarray(test_pic[j,...]))
    test_img[j] = test_img[j].resize((112,112))
# val_pic = train_npy[0:600,...]
# train_pic = train_npy[600:3000,...]
val_label = np.zeros(600,dtype=int)
train_label = np.zeros(2400,dtype=int)
val_img = []
train_img = []
#划分训练集和验证集
for i in range(10):
    # val_pic = np.append(val_pic,train_npy[3000*i:3000*i+600,...], axis=0)
    # train_pic = np.append(train_pic,train_npy[3000*i+600:3000*(i+1),...], axis=0)
    val_img.extend(train_all[3000*i:3000*i+600])
    train_img.extend(train_all[3000*i+600:3000*(i+1)])
    if i>0:
        val_label = np.append(val_label,i*np.ones(600,dtype=np.int64))
        train_label = np.append(train_label,i*np.ones(2400,dtype=np.int64))

train_transform = transforms.Compose([transforms.RandomCrop(112, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])
temp = train_img[0]
#转换为张量
transform = transforms.ToTensor()
BATCH_SIZE = 64
class MyDataSet(Data.Dataset):
    def __init__(self, pic_data, label_data=None, transform=None, type = 'train'):  # 第一步初始化各个变量
        self.pic_data = pic_data #导入所有数据
        self.label_data = label_data
        self.type = type
        self.transform = transform
    def __getitem__(self, idx):  # 获取数据
        img = self.pic_data[idx]
        data = self.transform(img)
        if self.type == 'test':
            return data
        else:
            label = self.label_data[idx]
            return data,label
    def __len__(self):
        return len(self.pic_data)  # 返回数据集长度
#导入数据集
train_dataset = MyDataSet(train_img,train_label,train_transform, type = 'train')
val_dataset = MyDataSet(val_img,val_label,test_transform, type = 'train')
test_dataset = MyDataSet(test_img,transform = test_transform, type = 'test')
#加载小批次数据，即将数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)
print("train_loader:", len(train_loader))
print("val_loader:", len(val_loader))
print("test_loader:", len(test_loader))

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        #输入输出维度一致时直接相加
        self.downsample = nn.Sequential()
        #维度不一致时
        if stride != 1 or inchannel != outchannel:
            #1X1卷积核改变维度
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        residual = self.downsample(x)
        out = out + residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock):
        #预先处理
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(1,128,kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, bias=False)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #输入维度
        self.in_channel = 64
        #残差模块
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.drop1 = nn.Dropout2d()
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, 10)

    #残差层，每层有两个残差模块
    def make_layer(self, block, channels, num_blocks, stride):
        layers = []
        #第一个模块
        layers.append(block(self.in_channel, channels, stride))
        self.in_channel = channels
        #第二个模块，stride=1，不需要downsample
        layers.append(block(self.in_channel, channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)


model = ResNet18().cuda() #实例化网络
#model = ResNet18() #实例化网络
loss = nn.CrossEntropyLoss() #损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(),lr = 0.1,momentum=0.9, weight_decay=1e-4)
num_epochs = 15

#训练和验证集的准确率和损失值
train_losses = []
train_acces = []
val_losses = []
val_acces = []

for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    model.train()  # 将网络转化为训练模式
    train_loader = tqdm(train_loader)
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
        X = Variable(X).cuda()  #转化数据类型
        #X = Variable(X)
        X = X.float()
        label = Variable(label).cuda()
        #label = Variable(label)
        out = model(X)  # 正向传播
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数
        # 计算损失
        train_loss += float(lossvalue)
        # 计算精确度
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    train_losses.append(train_loss / len(train_loader))
    train_acces.append(train_acc / len(train_loader))
    print("train epoch:" + ' ' + str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' ' + str(train_acc / len(train_loader)))

    val_pred = None #预测分类
    val_label_all = None #实际分类
    val_pred_pro_all = None #对应概率
    val_loss = 0  # 定义验证损失
    val_acc = 0
    model.eval() #模型转化为评估模式
    for i,(X, label) in enumerate(val_loader):
        X = Variable(X).cuda()
        X = X.float()
        label = Variable(label).cuda()
        with torch.no_grad():
            testout = model(X)
        testloss = loss(testout,label)
        val_loss += float(testloss)
        _, pred = testout.max(1)
        if val_pred is None:
            val_pred = torch.cat([pred])
        else:
            val_pred = torch.cat([val_pred, pred])
        if val_label_all is None:
            val_label_all = label
        else:
            val_label_all = torch.cat([val_label_all, label])
        if val_pred_pro_all is None:
            val_pred_pro_all = torch.cat([F.sigmoid(testout)])
        else:
            val_pred_pro_all = torch.cat([val_pred_pro_all, F.sigmoid(testout)])
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        val_acc += acc
    val_losses.append(val_loss / len(val_loader))
    val_acces.append(val_acc / len(val_loader))
    print("echo:" + ' ' + str(echo))
    print("lose:" + ' ' + str(val_loss / len(val_loader)))
    print("accuracy:" + ' ' + str(val_acc / len(val_loader)))

y_val = val_label_all.cpu().detach().numpy()
# print(y_test)
y_val_pred = val_pred.cpu().detach().numpy()
y_pred_pro = val_pred_pro_all.cpu().detach().numpy()
print('ACC:%.7f' % accuracy_score(y_true=y_val, y_pred=y_val_pred))
print('Precision-macro:%.7f' % precision_score(y_true=y_val, y_pred=y_val_pred, average='macro'))
print('Recall-macro:%.7f' % recall_score(y_true=y_val, y_pred=y_val_pred, average='macro'))
print('F1-macro:%.7f' % f1_score(y_true=y_val, y_pred=y_val_pred, average='macro'))

fpr = dict()
tpr = dict()
roc_auc = dict()
average_precision = dict()
recall = dict()
precision = dict()
for i in range(10):
    y_test2 = copy.deepcopy(y_val)
    y_test2[y_test2!=i] = 10
    y_test2[y_test2==i] = 1
    y_test2[y_test2==10] = 0
    y_pred_pro2 = y_pred_pro[:,i]
    #print(y_pred_pro2)
    #print(y_test2)
    fpr[i], tpr[i], _ = roc_curve(y_test2, y_pred_pro2)
    #面积
    roc_auc[i] = roc_auc_score(y_test2,y_pred_pro2)
    average_precision[i] = average_precision_score(y_test2, y_pred_pro2)
    #print('Average precision-recall score: %.7f' % average_precision)
    precision[i], recall[i], _ = precision_recall_curve(y_test2,y_pred_pro2)

# Plot of a ROC curve for a specific class
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - MNIST')
for i in range(10):
    plt.plot(fpr[i], tpr[i], label="class" + str(i) + ':ROC curve (area = %0.3f)' % roc_auc[i],color=colors[i])
plt.legend(loc="lower right")
#plt.savefig("roc.png")


# Plot of a ROC curve for a specific class
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('precision recall curve - MNIST')
for i in range(10):
    plt.plot(recall[i], precision[i], label="class" + str(i) + ':AP (score = %0.3f)' % average_precision[i],color=colors[i])
plt.legend(loc="lower right")

#保存网络
torch.save(model, 'model.pkl')
label_all = None
pred_all = None
pred_pro_all = None

model2 = torch.load('model.pkl')
model2.eval() #模型转化为评估模式
for X in test_loader:
    X = Variable(X).cuda()
    X = X.float()
    testout = model2(X)
    _, pred = testout.max(1)
    if pred_all is None:
        pred_all = torch.cat([pred])
    else:
        pred_all = torch.cat([pred_all,pred])

y_pred = pred_all.cpu().detach().numpy()
y_imagid = np.arange(5000)
y_result = np.vstack((y_imagid,y_pred))
y_result = y_result.transpose()
y_output = pd.DataFrame(y_result)
print(y_output)
y_output.columns = ['image_id', 'label']
y_output.to_csv('output2.csv',index=False)