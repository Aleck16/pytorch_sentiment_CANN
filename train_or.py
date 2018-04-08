# _*_ coding: UTF8 _*_
"""
code reference: https://github.com/junwang4/CNN-sentence-classification-pytorch-2017/blob/master/cnn_pytorch.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.autograd as autograd
import data_helpers
from sklearn.model_selection import KFold
import time
import random

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

'''
CNN中LSTM和卷积滤波器的隐含单元数为100，
卷积滤波器长度为3，丢包率为0.5，最小批量为16个。
这些超参数通过SST-2上的网格搜索来选择开发集。
'''

use_cuda = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
EPOCH = 10
BATCH_SIZE = 16
LR = 1e-3

# 得到词索引
X, Y, word_to_ix, ix_to_word = data_helpers.load_data()

vocab_size = len(word_to_ix)
max_sent_len = X.shape[1]
num_classes = Y.shape[1]

print('vocab size       = {}'.format(vocab_size))
print('max sentence len = {}'.format(max_sent_len))
print('num of classes   = {}'.format(num_classes))


class MyModel(nn.Module):
    """
        1,将句子矩阵，先做卷积，通过avarage pool生成一个特征列向量,长度为句子长度
        2,将句子矩阵做LSTM，得到每一个时刻输出的矩阵
        3,将1的结果的每个元素点乘到2结果的向量
        4,将最终的矩阵求和输出
        5,全连接层
        """

    def __init__(self, max_sent_len, hidden_dim, embedding_dim, filter_size,
                 num_filters, vocab_size, tagset_size, batch_size):
        super(MyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # [16,300,61]--->[16,100,61]
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                              kernel_size=filter_size, padding=int(filter_size / 2))
        # 上一步的结果需要做维度转换：[16,61,100]--->[16,61,1]
        self.pool = nn.AvgPool1d(num_filters)

        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, dropout=0.5)
        # 下一步是点乘处理

        # 求和操作

        self.hidden2lab = nn.Linear(max_sent_len, tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, x):
        # x: (batch, sentence_len)
        x = self.word_embeddings(x)  # [16,61,300]
        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        cnn_in = x.transpose(1, 2)  # switch 2nd and 3rd axis; [16,300,61]
        lstm_in = x.transpose(0, 1)
        # 1,卷积+池化
        cnn_out = F.relu(self.conv(cnn_in))  # [16,100,61]
        cnn_out = self.pool(cnn_out.transpose(1, 2))  # [16,61,1]

        # 2,LSTM1
        # Expected hidden[0] size (1, 16, 100), got (1, 1, 100)

        lstm1_out, lstm1_hidden = self.lstm1(lstm_in, self.hidden)  # lstm_out:[61,16,100]
        # 乘法操作[61,16,100]*[16,61,1]
        input = torch.mul(lstm1_out.transpose(0,1), cnn_out)  # [16,61,100]

        # 将各个时间点的向量相加,[16,61,100]
        in_sum = torch.sum(input, 2)  # x:[16,61,1]


        y = self.hidden2lab(in_sum)  # [16,2]
        # log_probs = F.log_softmax(y)
        log_probs = F.softmax(y, dim=1)
        return log_probs  # [16,2]


def evaluate(model, x_test, y_test):
    inputs = autograd.Variable(x_test)
    model.batch_size = len(inputs.data)
    model.hidden = model.init_hidden()
    preds = model(inputs)
    preds = torch.max(preds, 1)[1]
    y_test = torch.max(y_test, 1)[1]
    if use_cuda:
        preds = preds.cuda()
    eval_acc = sum(preds.data == y_test) * 1.0 / len(y_test)
    return eval_acc


def train_test_one_split(train_index, test_index):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    # x_train=x_train[:20]
    # y_train=y_train[:20]
    # x_test=x_test[:20]
    # y_test=y_test[:20]

    # numpy array to torch tensor
    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).float()
    # 包装数据和目标张量的数据集。一个封装类，两个属性：data_tensor[8529,56];target_tensor[8529,2];
    # 通过沿着第一个维度索引两个张量来恢复每个样本。
    dataset_train = data_utils.TensorDataset(x_train, y_train)
    # 数据加载器,将dataset_train数据再一次封装，加入batch_size等属性
    train_loader = data_utils.DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4,
                                         pin_memory=False)

    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).float()
    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    # max_sent_len, hidden_dim, embedding_dim, filter_size,
    # num_filters, vocab_size, tagset_size
    model = MyModel(max_sent_len=max_sent_len,
                    hidden_dim=HIDDEN_DIM,
                    embedding_dim=EMBEDDING_DIM,
                    filter_size=3,
                    num_filters=100,
                    vocab_size=vocab_size,
                    tagset_size=2,
                    batch_size=BATCH_SIZE)

    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    loss_fn = nn.BCELoss()

    for epoch in range(EPOCH):

        model.train()  # set the model to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            model.batch_size = len(labels.data)
            model.hidden = model.init_hidden()
            preds = model(inputs)
            if use_cuda:
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # set the model to evaluation mode
        eval_acc = evaluate(model, x_test, y_test)
        print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}'.format(epoch, loss.data[0], eval_acc))

    model.eval()  # set the model to evaluation mode
    eval_acc = evaluate(model, x_test, y_test)
    return eval_acc


cv_folds = 5  # 5-fold cross validation
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
acc_list = []
tic = time.time()
for cv, (train_index, test_index) in enumerate(kf.split(X)):
    acc = train_test_one_split(train_index, test_index)
    print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
    acc_list.append(acc)
print('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format(sum(acc_list) / len(acc_list), time.time() - tic))
