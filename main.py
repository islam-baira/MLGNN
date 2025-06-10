import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import argparse
from network import Net
from fnc import FND 
import os.path as osp
from pathlib import Path
import json
from datetime import datetime
from torch_geometric.datasets import UPFD
from PHEME_Dataset import PHEME_Dataset
from WEIBO_Dataset import WEIBO_Dataset

from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
from pytorch_metric_learning import losses
import statistics


# Hyperparameters & Setup
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.1, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='gossipcop')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=64, help='maximum number of epochs')
parser.add_argument('--num_features', type=int, default=768, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=256, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=0.05, help='Margin in the triplet loss')

args = parser.parse_args()


device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
if args.dataset == 'pheme' or args.dataset == 'PHEME':
    dataset = PHEME_Dataset()
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test])
    args.batch_size = 64
    args.epochs = 512
if args.dataset == 'weibo' or args.dataset == 'WEIBO':
    dataset = WEIBO_Dataset()
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test])
    args.batch_size = 16
    args.epochs = 64
if args.dataset == "politifact" or args.dataset == "gossipcop":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'UPFD')
    train_dataset = UPFD(path, args.dataset, 'bert', 'test', ToUndirected())
    val_dataset = UPFD(path, args.dataset, 'bert', 'val', ToUndirected())
    test_dataset = UPFD(path, args.dataset, 'bert', 'train', ToUndirected())
    if args.dataset == "politifact":
        args.batch_size = 16
        args.epochs = 512
        args.alpha = 1.0
    if args.dataset == "gossipcop":
        args.batch_size = 32
        args.epochs = 256

# for fake/real classes.
num_classes = 2
# since we used BERT.
num_features = 768 #dataset.num_features

# fnd_model learning rate:
fnd_model_learning_rate = 0.0001#0.0001


# to compute std dev. of AUC-ROC 
__aus = []
# to compute std dev. of Absolute Error.
__maes = []
# to compute std dev. of accs
__accs = []
# to compite std dev. of pres.
__precs = []
# to compute std dev. of recalls 
__recalls = []
# to compute std dev. of F1
__f1s = [] 
for iter in range(args.iterations):    
    training_set = train_dataset
    validation_set = val_dataset
    test_set = test_dataset
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, 1, shuffle=False)
    test_loader = DataLoader(test_set, 1, shuffle=True)
    
    gnn_model = Net(num_features, args.nhid, args.final_dim, args.pooling_ratio, args.dropout_ratio).to(device)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   
    # Triplet loss function
    loss_func = losses.TripletMarginLoss(margin=args.alpha)

    gnn_model.train()

    fnd_model = FND(args.final_dim).to(device)
    
    fnd_optimizer = torch.optim.Adam(fnd_model.parameters(), lr=fnd_model_learning_rate)
   
    #Loss Computation.
    fnd_loss_func = nn.BCELoss()
    # training mode.
    fnd_model.train()

    for epoch in tqdm(range(args.epochs) , colour="red", desc="Epoch    :"):
         # .......
        gnn_loss = []
        fnd_loss = []
        y_true = []
        y_preds = []
        for batch in tqdm(train_loader, desc="Training :"):
            batch = batch.to(device)
            embeddings = []
            for i in range(batch.num_graphs):
                data = batch[i].to(device)
                embedding , _y = gnn_model(data)
                embeddings.append(embedding)
            gnn_optimizer.zero_grad()

            out = torch.squeeze(torch.stack(embeddings),1)
            loss = loss_func(out, batch.y.to(device))
            loss.backward(retain_graph=True)
            gnn_optimizer.step()
            gnn_loss.append(loss.item())

            with torch.autograd.set_detect_anomaly(True):
                    for index , (y, em) in enumerate(zip(batch.y, embeddings)):
                        input_tensor = em.clone().detach()
                        y_pred = fnd_model(input_tensor)        # Forward Propagation
                        y_preds.append(torch.round(y_pred.clone().cpu().detach()).numpy().flatten().astype(int))
                        y_true.append(y.cpu().numpy())
                        #--------------
                        y = y.to(torch.float)
                        loss2 = fnd_loss_func(y_pred.squeeze(1) , y.unsqueeze(0))  # Loss Computation #.squeeze(-1)
                        fnd_optimizer.zero_grad()          # Clearing all previous gradients, setting to zero 
                        loss2.backward()                    # Back Propagation
                        fnd_optimizer.step()               # Updating the parameters 
                        fnd_loss.append(loss2.item())
                        

        
    gnn_model.eval()
    fnd_model.eval()

    # real ys
    test_y = []
    # to store predicted
    y_pred_list = []
    with torch.no_grad():
        # Evaluate the results
        for data in tqdm(test_dataset, desc="Test     :", colour="red"):
            data = data.to(device)
            test_y.append(data.y.item())
            embeding , _ = gnn_model(data)
            y_test_pred = fnd_model(embeding)
    
            y_pred_tag = torch.round(y_test_pred)
                
            y_pred_list.append(y_pred_tag.detach().cpu().numpy())

    #Takes arrays and makes them list of list for each batch        
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


    y_test_predected = y_pred_list
    y_true_test = test_y
    auc_score = metrics.roc_auc_score(y_true_test, y_test_predected)

    mae_score = metrics.mean_absolute_error(y_true_test, y_test_predected)

    conf_matrix = metrics.confusion_matrix(y_true_test ,y_test_predected)
    __accs.append(metrics.accuracy_score(y_true_test,  y_test_predected))
    __precs.append(metrics.precision_score(y_true_test, y_test_predected))
    __recalls.append(metrics.recall_score(y_true_test, y_test_predected))
    __f1s.append(metrics.f1_score(y_true_test, y_test_predected))
    __aus.append(auc_score)
    __maes.append(mae_score)

separator = '-' * 79

print(separator)
print("Final results: ")
print("Acc                :\t " + str(statistics.mean(__accs))[0:7], "with std. dev. of " +str(statistics.stdev(__accs))[0:7])
print("Precision          :\t " + str(statistics.mean(__precs))[0:7], "with std. dev. of " +str(statistics.stdev(__precs))[0:7])
print("Recall             :\t " + str(statistics.mean(__recalls))[0:7], "with std. dev. of " +str(statistics.stdev(__recalls))[0:7])
print("F1                 :\t " + str(statistics.mean(__f1s))[0:7], "with std. dev. of " +str(statistics.stdev(__f1s))[0:7])
print("Mean AUC-ROC score :\t " + str(statistics.mean(__aus))[0:7], "with std. dev. of " +str(statistics.stdev(__aus))[0:7])
print("MAE mean           :\t " + str(statistics.mean(__maes))[0:7], "with std. dev. of " +str(statistics.stdev(__maes))[0:7])
print(separator)
