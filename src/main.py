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
from FND import FND #FND
import os.path as osp
from pathlib import Path
import json
from datetime import datetime
from torch_geometric.datasets import UPFD
from PHEME_Dataset import PHEME_Dataset
from WEIBO_Dataset import WEIBO_Dataset

from torch_geometric.transforms import ToUndirected
import pickle
from tqdm import tqdm
from pytorch_metric_learning import losses
# Tensorboard
from torch.utils.tensorboard import SummaryWriter

import time # to mesure the execution time.
import statistics

import os
import pandas as pd
from auc_roc_draw import draw_roc_curve

print("---------------start--------------------------")

# funtction to save the model for latter use.
def save_model(model, path=f'model.latest.pkl'):
    path_obj = Path(path)
    if not path_obj.exists():
        # open a file, where you ant to store the data
        file = open(path, 'wb')
        # dump information to that file
        pickle.dump(model, file)
        # close the file
        file.close()
    else: 
        path_obj.unlink()
        # open a file, where you ant to store the data
        file = open(path, 'wb')
        # dump information to that file
        pickle.dump(model, file)
        # close the file
        file.close()

# tensorboad writer.
writer = SummaryWriter()

def train_metrics(y_true, y_preds):
    acc = metrics.accuracy_score(y_true,y_preds)
    prec = metrics.precision_score(y_true,y_preds)
    recall = metrics.recall_score(y_true,y_preds)
    f1 = metrics.f1_score(y_true,y_preds)
    return acc, prec, recall, f1
# Hyperparameters & Setup

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_fnc', type=float, default=0.0001, help='learning rate for FNC')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.1, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='gossipcop')
parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=64, help='maximum number of epochs')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
parser.add_argument('--num_features', type=int, default=768, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=256, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=0.05, help='Margin in the triplet loss')


args = parser.parse_args()

experience_name = f"DATASET_{args.dataset}_BATCH_SIZE_{args.batch_size}_EPOCHS_{args.epochs}_LR_GNN_{args.lr}_LR_FNC_{args.lr_fnc}_ALPHA_{args.alpha}"

device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
print(args.dataset)
if args.dataset == 'pheme' or args.dataset == 'PHEME':
    dataset = PHEME_Dataset()
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test])
if args.dataset == 'weibo' or args.dataset == 'WEIBO':
    dataset = WEIBO_Dataset()
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test])
if args.dataset == "politifact" or args.dataset == "gossipcop":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'UPFD')
    train_dataset = UPFD(path, args.dataset, 'bert', 'test', ToUndirected())
    val_dataset = UPFD(path, args.dataset, 'bert', 'val', ToUndirected())
    test_dataset = UPFD(path, args.dataset, 'bert', 'train', ToUndirected())

num_classes = 2 #dataset.num_classes
num_features = 768 #dataset.num_features

# fnd_model learning rate:
fnd_model_learning_rate = args.lr_fnc#0.0001#0.0001


# Train
min_loss = 1e10

# to store metrics and calculate mean + std
val_accs = []
test_accs = []
val_precs = []
test_precs = []
val_recalls = []
test_recalls = []
val_f1s = []
test_f1s = []
# new addition.
test_conf_matrices = []
test_auc_roc = []
test_classification_reports = []
print(args)

# Record the start time
times = []

for iter in range(args.iterations):    
    # record the start time here.
    start_time = time.time()
    
    # start training.
    training_set = train_dataset
    validation_set = val_dataset
    test_set = test_dataset
    print('----------------------')
    print(len(training_set))
    print(len(test_set))
    print('----------------------')
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, 1, shuffle=False)
    test_loader = DataLoader(test_set, 1, shuffle=False)
    
    gnn_model = Net(num_features, args.nhid, args.final_dim, args.pooling_ratio, args.dropout_ratio).to(device)
    # gnn model optimizer
    #Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   
   
    # Triplet loss function
    #criterion = torch.nn.MarginRankingLoss(margin=args.alpha)
    loss_func = losses.TripletMarginLoss(margin=args.alpha)


    gnn_model.train()



    fnd_model = FND(args.final_dim).to(device)
    #Adam(fnd_model.parameters(), lr=fnd_model_learning_rate)
    fnd_optimizer = torch.optim.Adam(fnd_model.parameters(), lr=fnd_model_learning_rate)
    #Loss Computation.
    fnd_loss_func = nn.BCELoss() #.BCEWithLogitsLoss() #BCELoss()
    # training mode.
    fnd_model.train()
    # FIRST STAGE: TRAINING WITH TRIPLET LOSS
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
                #print(embedding, _y)
                embeddings.append(embedding)
            gnn_optimizer.zero_grad()

            out = torch.squeeze(torch.stack(embeddings),1)
            loss = loss_func(out, batch.y.to(device))
            loss.backward(retain_graph=True)
            gnn_optimizer.step()
            #print('GNN loss value: '+str(loss.item()))
            gnn_loss.append(loss.item())

            # tensorboard

            with torch.autograd.set_detect_anomaly(True):
                    for index , (y, em) in enumerate(zip(batch.y, embeddings)):
                        input_tensor = em.clone().detach()
                        y_pred = fnd_model(input_tensor)        # Forward Propagation
                        # to be used in train_metrics
                        y_preds.append(torch.round(y_pred.clone().cpu().detach()).numpy().flatten().astype(int))
                        y_true.append(y.cpu().numpy())
                        #--------------
                        y = y.to(torch.float)
                        loss2 = fnd_loss_func(y_pred.squeeze(1) , y.unsqueeze(0))  # Loss Computation #.squeeze(-1)
                        fnd_optimizer.zero_grad()          # Clearing all previous gradients, setting to zero 
                        loss2.backward()                    # Back Propagation
                        fnd_optimizer.step()               # Updating the parameters 
                        #print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
                        fnd_loss.append(loss2.item())
                        

        
        #writer.add_scalar(f"GNN_Train_Loss_{args.dataset}_EPOCHS:{args.epochs}_BATCH_SIZE:{args.batch_size}_LR:{args.lr}", loss.item(), epoch)
        writer.add_scalar(f"Train_Loss_{experience_name}", loss2.item(), epoch)
        #writer.add_scalar(f"Model_Loss_{args.dataset}_EPOCHS:{args.epochs}_BATCH_SIZE:{args.batch_size}_LR:{args.lr}", loss.item() + loss2.item(), epoch)
        # metrics for training + plot.
        train_acc, train_prec, train_recall, train_f1 = train_metrics(y_true, y_preds)
        writer.add_scalar(f"Train_ACC_{experience_name}", train_acc, epoch)

        
        # validation goes here.
        val_predections = []
        val_real_ys = []
        val_loss = 0.0        
        with torch.no_grad():
            for data in tqdm(validation_set , colour="green", desc="Validation"):
                data = data.to(device)
                val_real_ys.append(data.y)
                gnn_model.eval()
                fnd_model.eval()
                val_out , _ = gnn_model(data)
                pridected = fnd_model(val_out)

                out = pridected.clone().detach()
                val_loss += fnd_loss_func(out, data.y.clone().unsqueeze(0).to(torch.float)).item()

                val_predections.append(pridected)
            
            val_predections_ = []
            for v in val_predections:
                val_predections_.append(torch.squeeze(v,1))
    
            y_hat = []
            yyy = []
            for v in val_predections_:
                y_hat.append(v.item())
            for v in val_real_ys:
                yyy.append(v.item())
            new_list = [round(item) for item in y_hat]

            validation_acc = metrics.accuracy_score(yyy,new_list)
            
            # save the accuracy validation and polot it.
            writer.add_scalar(f"Val_Acc/{experience_name}", validation_acc , epoch)
            writer.add_scalar(f"Val_Loss/{experience_name}", val_loss/len(validation_set) , epoch)


                

        # todo save the model here.
        # todo: check for the best model before saving
        save_model(gnn_model, f"saved_models/gnn_model_{args.dataset}_batch_{args.batch_size}_epochs_{args.epochs}.pkl")
        save_model(fnd_model, f"saved_models/fnd_model_{args.dataset}_batch_{args.batch_size}_epochs_{args.epochs}.pkl")

    

    #for epoch in tqdm(1 , colour="red", desc="Test     :"):
    gnn_model.eval()
    fnd_model.eval()

    # real ys
    test_y = []
    # to store predicted
    y_pred_list = []
    # to store probs to calculate AUC-ROC.
    probs = []
    with torch.no_grad():
        # Evaluate the results
        for data in tqdm(test_dataset, desc="Test     :", colour="red"):
            data = data.to(device)
            test_y.append(data.y.item())
            embeding , _ = gnn_model(data)
            y_test_pred = fnd_model(embeding)
    
            # for AUC-ROC  
            y_probabilities = y_test_pred.detach().clone()

            y_pred_tag = torch.round(y_test_pred)
                
            y_pred_list.append(y_pred_tag.detach().cpu().numpy())
            # new.
            probs.append(y_probabilities.detach().cpu().numpy())

    #Takes arrays and makes them list of list for each batch        
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # new
    probs = [a.squeeze().tolist() for a in probs]
    #flattens the lists in sequence

    y_test_predected = y_pred_list
    #y_true_test = np.array(test_y).ravel()
    y_true_test = test_y
    auc_score = metrics.roc_auc_score(y_true_test, probs)
    test_auc_roc.append(auc_score)
    

    ###### here plot the auc-roc curve.
    # Calculate ROC curve
    # fpr, tpr, thresholds = metrics.roc_curve(y_true_test, probs)
    draw_roc_curve(y_true_test, probs, f"{experience_name}.svg")
    # Calculate AUC
    # roc_auc = metrics.auc(fpr, tpr)
    ######
    
    # Calculate the mean absolute error for this fold
    #mae_score = metrics.mean_absolute_error(y_true_test, y_test_predected)
    #mae_scores.append(mae_score)

    conf_matrix = metrics.confusion_matrix(y_true_test ,y_test_predected)
    #print("Confusion Matrix of the Test Set ")
    #print("-----------")
    #print(conf_matrix)
    #print("-----------")
    #print("Acc :\t " +str(metrics.accuracy_score(y_true_test,y_test_predected)))
    #print("Precision :\t " +str(metrics.precision_score(y_true_test,y_test_predected)))
    #print("Recall :\t " +str(metrics.recall_score(y_true_test,y_test_predected)))
    #print("F1 :\t "+str(metrics.f1_score(y_true_test,y_test_predected)))
    #print(metrics.classification_report(y_true_test ,y_test_predected))
    #print("-----------")
    #print(f'Mean AUC-ROC score: {auc_score:.5f}')
    ##print(f'Std dev of AUC-ROC scores: {std_auc_score:.5f}')
    #print("\n")
    #print(f'MAE mean : {mae_score:.5f}')
    ##print(f'MAE std dev : {mae_std:.5f}')
    #print("\n")
    #print("-------------")
    
    # new changes goes here.
    test_conf_matrices.append(conf_matrix)
    test_accs.append(metrics.accuracy_score(y_true_test,y_test_predected))
    test_precs.append(metrics.precision_score(y_true_test,y_test_predected))
    test_recalls.append(metrics.recall_score(y_true_test,y_test_predected))
    test_f1s.append(metrics.f1_score(y_true_test,y_test_predected))
    test_classification_reports.append(metrics.classification_report(y_true_test ,y_test_predected, output_dict=True))
    #writer.add_scalar(f"Test_Acc/{args.dataset}_EPOCHS:{args.epochs}_BATCH_SIZE:{args.batch_size}_LR:{args.lr}", metrics.accuracy_score(y_true_test,y_test_predected) , epoch)
    #writer.add_scalar(f"Test_Loss/{args.dataset}_EPOCHS:{args.epochs}_BATCH_SIZE:{args.batch_size}_LR:{args.lr}", test_loss , epoch)

    
    # ensure to flush data and close the writer.
    writer.flush()
    writer.close()
    
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    times.append(elapsed_time)
    

final_test_acc = round(np.mean(test_accs), 5)
std_test_acc = round(np.std(test_accs), 5)

final_test_prec = round(np.mean(test_precs), 5)
std_test_prec = round(np.std(test_precs), 5)

final_test_recall = round(np.mean(test_recalls), 5)
std_test_recall = round(np.std(test_recalls), 5)

final_test_f1 = round(np.mean(test_f1s), 5)
std_test_f1 = round(np.std(test_f1s), 5)

final_test_auc_roc = round(np.mean(test_auc_roc), 5)
std_test_auc_roc = round(np.std(test_auc_roc), 5)

final_exec_time = round(np.mean(times), 5)
#print(f"Execution time: {statistics.mean(times)} seconds")
# Calculate mean classification report
final_classification_report = {
    'precision_0': round(np.mean([report['0']['precision'] for report in test_classification_reports]), 5),
    'recall_0': round(np.mean([report['0']['recall'] for report in test_classification_reports]), 5),
    'f1-score_0': round(np.mean([report['0']['f1-score'] for report in test_classification_reports]), 5),
    'precision_1': round(np.mean([report['1']['precision'] for report in test_classification_reports]), 5),
    'recall_1': round(np.mean([report['1']['recall'] for report in test_classification_reports]), 5),
    'f1-score_1': round(np.mean([report['1']['f1-score'] for report in test_classification_reports]), 5)
}

# Add mean confusion matrix to DataFrame
final_confusion_matrix = np.mean(test_conf_matrices, axis=0)
cm = final_confusion_matrix

report = final_classification_report
#confusion_matrix_columns = [f'CM_{i}' for i in range(final_confusion_matrix.shape[0])]

# File path for the CSV file
csv_file_path = 'MLGNN.csv'

# Create a DataFrame with the mean metrics
data = {

    'dataset': [args.dataset],
    'batch_size': [args.batch_size],
    'epochs': [args.epochs],
    'lr_gnn': [args.lr],
    'lr_fnc': [args.lr_fnc],
    'alpha': [args.alpha],
    
    'accuracy': [final_test_acc],
    'std_acc': [std_test_acc],

    'precesion': [final_test_prec],
    'std_prec': [std_test_prec],
    
    'recall': [final_test_recall],
    'std_recall': [std_test_recall],
    
    'f1_score': [final_test_f1],
    'std_f1': [std_test_f1],
    
    'auc_roc': [final_test_auc_roc],
    'std_auc_roc': [std_test_auc_roc],

    'exec_time': [final_exec_time],
    
    'tp': [cm[0, 0]],
    'tn': [cm[1, 1]],
    'fn': [cm[1, 0]],
    'fp': [cm[0, 1]],
    
    'precision_0': [report['precision_0']],
    'recall_0': [report['recall_0']],
    'f1_score_0': [report['f1-score_0']],
    'precision_1': [report['precision_1']],
    'recall_1': [report['recall_1']],
    'f1_score_1': [report['f1-score_1']],
}

df = pd.DataFrame(data)


#df_confusion_matrix = pd.DataFrame(final_confusion_matrix, columns=confusion_matrix_columns)
#df = pd.concat([df, df_confusion_matrix], axis=1)
#print(confusion_matrix_columns)

#df_final_classification_report = pd.DataFrame(final_classification_report, columns=['precision','recall','f1-score'])



# Convert the dictionary to a DataFrame
#df_final_classification_report = pd.DataFrame(list(final_classification_report.items()), columns=['Metric', 'Value'])

# Extract information from the 'Metric' column to create new columns
#df_final_classification_report['Metric_Type'] = df_final_classification_report['Metric'].str.split('_').str[-1]
#df_final_classification_report['Metric_Name'] = df_final_classification_report['Metric'].str.split('_').str[0]

# Pivot the DataFrame to reshape it
#df_final_classification_report_pivoted = df_final_classification_report.pivot(index='Metric_Type', columns='Metric_Name', values='Value').reset_index()

#print(df_final_classification_report)
#df = pd.concat([df, df_final_classification_report_pivoted], axis=1)
#print(df)
#print("--------")

# If the CSV file already exists, append the results; otherwise, create a new file
if os.path.exists(csv_file_path):
    # Load existing data
    existing_df = pd.read_csv(csv_file_path)

    # Append the new results to the existing data
    updated_df = pd.concat([existing_df, df], ignore_index=True)

    # Write the updated DataFrame to the CSV file
    updated_df.to_csv(csv_file_path, index=False)

else:
    # Write the DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False)


print("---------------end--------------------------")
