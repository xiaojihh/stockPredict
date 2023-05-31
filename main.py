import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model import LSTM
from datasets import TrainSet


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='深证指数')
parser.add_argument('--file_name', type=str, default='./data/十年399001.SZ.csv')
parser.add_argument('--device', type=str, default='1')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--before_days', type=int, default=180)
parser.add_argument('--pred_days', type=int, default=30)
parser.add_argument('--split', type=float, default=0.15, help='')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def getData(df, column, return_all=True, generate_index=False):
    series = df[column].copy()
    train_end = -int(len(series) * args.split)
    # 创建训练集
    data = pd.DataFrame()
    
    # 准备天数
    for i in range(args.before_days):
        # 最后的 -days_before - days_pred 天只是用于预测值，预留出来
        data['b%d' % i] = series.tolist()[i: -args.before_days - args.pred_days + i]
    
    # 预测天数
    for i in range(args.pred_days):
        data['y%d' % i] = series.tolist()[args.before_days + i: - args.pred_days + i]
    
    # 是否生成 index
    if generate_index:
        data.index = series.index[args.before_days:]
        
    train_data, val_data, test_data = data[:train_end+train_end], data[train_end+train_end:train_end], data[train_end:]
                
    if return_all:
        return train_data, val_data, test_data, series, df.index.tolist()
    
    return train_data, val_data, test_data

def train(rnn, train_loader, loss_func, optimizer, val_loader):
    best_loss = 1000
    for step in range(args.epochs):
        for tx, ty in train_loader:
            if torch.cuda.is_available():
                tx = tx.cuda()
                ty = ty.cuda() 
            
            output = rnn(torch.unsqueeze(tx, dim=2))             
            loss = loss_func(torch.squeeze(output), ty)        
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
            
            print('epoch : %d  ' % step, 'train_loss : %.4f' % loss.cpu().item())
            
        with torch.no_grad():
            for tx, ty in val_loader:
                if torch.cuda.is_available():
                    tx = tx.cuda()
                    ty = ty.cuda() 
            
                output = rnn(torch.unsqueeze(tx, dim=2))             
                loss = loss_func(torch.squeeze(output), ty)
            
                print('epoch : %d  ' % step, 'val_loss : %.4f' % loss.cpu().item())
            
            if loss.cpu().item() < best_loss:
                best_loss = loss.cpu().item()
                torch.save(rnn, os.path.join('weights', args.exp_name, 'rnn.pkl'))
                print('new model saved at epoch {} with val_loss {}'.format(step, best_loss))

def predict(rnn, train_mean, train_std, df_index, all_series_test1, train_end):
    generate_data_train = []
    generate_data_test = []

    # 测试数据开始的索引
    test_start = len(all_series_test1) + train_end

    # 对所有的数据进行相同的归一化
    all_series_test1 = (all_series_test1 - train_mean) / train_std
    all_series_test1 = torch.Tensor(all_series_test1)


    for i in range(args.before_days, len(all_series_test1) - args.pred_days, args.pred_days):
        x = all_series_test1[i - args.before_days:i]
        # 将 x 填充到 (bs, ts, is) 中的 timesteps
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)
        
        if torch.cuda.is_available():
            x = x.cuda()

        y = torch.squeeze(rnn(x))
        
        if i < test_start:
            generate_data_train.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
        else:
            generate_data_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
            
    generate_data_train = np.concatenate(generate_data_train, axis=0)
    generate_data_test  = np.concatenate(generate_data_test, axis=0)


    plt.figure(figsize=(12,8))
    plt.plot(df_index[args.before_days: len(generate_data_train) + args.before_days], generate_data_train, 'b', label='generate_train')
    plt.plot(df_index[train_end:len(generate_data_test) + train_end], generate_data_test, 'k', label='generate_test')
    plt.plot(df_index, all_series_test1.clone().numpy() * train_std + train_mean, 'r', label='real_data')
    plt.legend()
    plt.savefig(os.path.join('results', args.exp_name+'.jpg'))
    plt.show()

def normalize_data(train_data, val_data, test_data):
    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)

    val_data_numpy = np.array(val_data)
    val_data_numpy = (val_data_numpy - train_mean) / train_std
    val_data_tensor = torch.Tensor(val_data_numpy)

    test_data_numpy = np.array(test_data)
    test_data_numpy = (test_data_numpy - train_mean) / train_std
    test_data_tensor = torch.Tensor(test_data_numpy)
    return train_data_tensor, val_data_tensor, test_data_tensor, train_mean, train_std

def main():
    # 读取数据
    df = pd.read_csv(args.file_name, index_col=0)
    df.index = list(map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'), df.index))
    # df = df[::-1]
    # 生成数据集
    train_data, val_data, test_data, all_series, df_index = getData(df, 'High')
    train_end = -len(test_data)

    # 获取所有原始数据
    all_series_test1 = np.array(all_series.copy().tolist())


    # 归一化，便与训练
    train_data_tensor, val_data_tensor, test_data_tensor, train_mean, train_std = normalize_data(train_data, val_data, test_data)

    # 创建 dataloader
    train_set = TrainSet(train_data_tensor, pred_days=args.pred_days)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    val_set = TrainSet(val_data_tensor, pred_days=args.pred_days)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=True)


    rnn = LSTM(pred_days=args.pred_days)
    if torch.cuda.is_available():
        rnn = rnn.cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    if not os.path.exists(os.path.join('weights', args.exp_name)):
        os.mkdir(os.path.join('weights', args.exp_name))    

    train(rnn, train_loader, loss_func, optimizer, val_loader)

    predict(rnn, train_mean, train_std, df_index, all_series_test1, train_end=train_end)

if __name__=='__main__':
    main()