import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g. 0 or 1)"
    )
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import os
import torch
import mlflow
import random
import numpy as np
from monai.utils import set_determinism
from tqdm import tqdm 
from glob import glob
from pathlib import Path

from data.Dataset import ISPY2
from torch.utils.data import DataLoader
from models.CNN_LSTM import CNNLSTM
from models.CNN_distLSTM import CNNdistLSTM
from models.CNN import CNN
from torch.amp import autocast, GradScaler
from scipy.special import softmax

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from utils.utils import set_deterministic, log_all_python_files

BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 100

def main(method, timepoints, fold):   

    # set_deterministic()  
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)   
       
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('accumulation_steps', ACCUMULATION_STEPS)
    mlflow.log_param('epochs', EPOCHS) 
    mlflow.log_param('device', device)   

    if method == "CNN_distLSTM":
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_time_dists=True)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_time_dists=True)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_time_dists=True)
    
    else:
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints)

    # Count samples per class
    patient_ids = torch.load(f"./data/breast_cancer/data_splits_{timepoints}_timepoints.pt")[fold]["train"]
    
    labels = []
    for patient_id in patient_ids:
        files = sorted(glob(f"./data/breast_cancer/data_processed/{patient_id}/*.pt"))
        label = files[0].split('/')[-1].split('_')[-1].replace('.pt', '')
        labels.append(int(label)) 
    labels = np.array(labels)
    
    class_counts = np.bincount(labels)

    # Inverse frequency as weights
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[labels]

    sample_weights = torch.DoubleTensor(sample_weights)

    # Sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=sampler)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if method == "CNN_LSTM":
        # model = CNNLSTM().cuda()
        model = CNNdistLSTM().cuda()
    elif method == "CNN_distLSTM":
        model = CNNdistLSTM().cuda()
    elif method == "CNN":
        model = CNN(num_timepoints=timepoints).cuda()    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, nesterov=True, momentum=0.99, weight_decay=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = np.inf
    best_val_metric = -np.inf

    for epoch in tqdm(range(1, EPOCHS + 1)):

        train_loss_list = []
        train_true_list = []
        train_pred_list = []
        train_score_list = []

        model.train()
        optimizer.zero_grad()

        for step, batch_data in tqdm(enumerate(train_dl), total=len(train_dl)):
            images = batch_data[0][:, timepoints:, :, :, :, :].float().to(device) # (B, T, C, D, H, W)
            labels = batch_data[1].long().to(device)  # (B,)           

            with torch.amp.autocast("cuda"):
                if method == "CNN_distLSTM":
                    time_dists = batch_data[2].float().to(device)  # (B, T)
                    logits = model(images, time_dists)
                else:
                    logits = model(images)
                
                loss = loss_fn(logits, labels)
                logits = logits.detach()
                preds = torch.argmax(logits, dim=1)
                scores = torch.softmax(logits, dim=1)[:, 1]

            scaler.scale(loss).backward()            
            train_loss_list.append(loss.item())
            train_true_list.extend(labels.cpu().numpy().tolist())
            train_pred_list.extend(preds.cpu().numpy().tolist())
            train_score_list.extend(scores.cpu().numpy().tolist())            

            # perform optimizer step every accum_steps
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        if (step + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        val_loss_list = []
        val_true_list = []
        val_pred_list = []
        val_score_list = []

        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(val_dl):
                images = batch_data[0][:, timepoints:, :, :, :, :].float().to(device) # (B, T, C, D, H, W)
                labels = batch_data[1].long().to(device)  # (B,)           

                with torch.amp.autocast("cuda"):
                    if method == "CNN_distLSTM":
                        time_dists = batch_data[2].float().to(device)  # (B, T)
                        logits = model(images, time_dists)
                    else:
                        logits = model(images)
                    
                    loss = loss_fn(logits, labels)
                    logits = logits.detach()
                    preds = torch.argmax(logits, dim=1)
                    scores = torch.softmax(logits, dim=1)[:, 1]
            
                val_loss_list.append(loss.item())
                val_true_list.extend(labels.cpu().numpy().tolist())
                val_pred_list.extend(preds.cpu().numpy().tolist())
                val_score_list.extend(scores.cpu().numpy().tolist())
        
        train_loss = np.mean(train_loss_list)
        train_bal_acc = balanced_accuracy_score(train_true_list, train_pred_list)
        train_mcc = matthews_corrcoef(train_true_list, train_pred_list)
        train_roc_auc = roc_auc_score(train_true_list, train_score_list)
        train_cm = confusion_matrix(train_true_list, train_pred_list)
        train_sensitivity = train_cm[1,1] / (train_cm[1,0] + train_cm[1,1])
        train_specificity = train_cm[0,0] / (train_cm[0,0] + train_cm[0,1])

        val_loss = np.mean(val_loss_list)
        val_bal_acc = balanced_accuracy_score(val_true_list, val_pred_list)
        val_mcc = matthews_corrcoef(val_true_list, val_pred_list)
        val_roc_auc = roc_auc_score(val_true_list, val_score_list)
        val_cm = confusion_matrix(val_true_list, val_pred_list)
        val_sensitivity = val_cm[1,1] / (val_cm[1,0] + val_cm[1,1])
        val_specificity = val_cm[0,0] / (val_cm[0,0] + val_cm[0,1])

        mlflow.log_metric("train_bal_acc", train_bal_acc, step=epoch)
        mlflow.log_metric("train_roc_auc", train_roc_auc, step=epoch)
        mlflow.log_metric("train_mcc", train_mcc, step=epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_sensitivity", train_sensitivity, step=epoch)
        mlflow.log_metric("train_specificity", train_specificity, step=epoch)

        mlflow.log_metric("val_bal_acc", val_bal_acc, step=epoch)
        mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)
        mlflow.log_metric("val_mcc", val_mcc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_sensitivity", val_sensitivity, step=epoch)
        mlflow.log_metric("val_specificity", val_specificity, step=epoch)

        if lr_scheduler is not None:
            mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=epoch)        
            lr_scheduler.step()

        if epoch == 1:            
            best_val_loss = val_loss
            best_val_metric = val_bal_acc
            torch.save(model.state_dict(), f"{method}_best_loss.pt")
            torch.save(model.state_dict(), f"{method}_best_metric.pt")
            mlflow.log_artifact(f"{method}_best_loss.pt")
            mlflow.log_artifact(f"{method}_best_metric.pt")
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{method}_best_loss.pt")
            mlflow.log_artifact(f"{method}_best_loss.pt")
        
        if val_bal_acc >= best_val_metric:
            best_val_metric = val_bal_acc
            torch.save(model.state_dict(), f"{method}_best_metric.pt")
            mlflow.log_artifact(f"{method}_best_metric.pt")
        
        torch.save(model.state_dict(), f"{method}_latest_epoch.pt")
        mlflow.log_artifact(f"{method}_latest_epoch.pt")
    
    # Test evaluation
    test_loss_list = []
    test_true_list = []
    test_pred_list = []
    test_score_list = []

    model.load_state_dict(torch.load(f"{method}_best_metric.pt"))

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_dl):
            images = batch_data[0][:, timepoints:, :, :, :, :].float().to(device) # (B, T, C, D, H, W)
            labels = batch_data[1].long().to(device)  # (B,)           

            with torch.amp.autocast("cuda"):
                if method == "CNN_distLSTM":
                    time_dists = batch_data[2].float().to(device)  # (B, T)
                    logits = model(images, time_dists)
                else:
                    logits = model(images)
                
                loss = loss_fn(logits, labels)
                logits = logits.detach()
                preds = torch.argmax(logits, dim=1)
                scores = torch.softmax(logits, dim=1)[:, 1]
        
            test_loss_list.append(loss.item())
            test_true_list.extend(labels.cpu().numpy().tolist())
            test_pred_list.extend(preds.cpu().numpy().tolist())
            test_score_list.extend(scores.cpu().numpy().tolist())
    
    test_bal_acc = balanced_accuracy_score(test_true_list, test_pred_list)
    test_roc_auc = roc_auc_score(test_true_list, test_score_list)
    test_mcc = matthews_corrcoef(test_true_list, test_pred_list)
    test_cm = confusion_matrix(test_true_list, test_pred_list)
    test_sensitivity = test_cm[1,1] / (test_cm[1,0] + test_cm[1,1])
    test_specificity = test_cm[0,0] / (test_cm[0,0] + test_cm[0,1])
    test_loss = np.mean(test_loss_list)

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_mcc", test_mcc)
    mlflow.log_metric("test_sensitivity", test_sensitivity)
    mlflow.log_metric("test_specificity", test_specificity)
    mlflow.log_metric("test_bal_acc", test_bal_acc)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # clean up
    os.remove(f"{method}_best_loss.pt")
    os.remove(f"{method}_best_metric.pt")
    os.remove(f"{method}_latest_epoch.pt")
    

if __name__ == "__main__":    

    set_deterministic()  

    for method in ["CNN", "CNN_LSTM", "CNN_distLSTM"]:
        for timepoints in [4]:
            for fold in range(5):

                mlflow.set_experiment("end-to-end")

                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():
                    main(method, timepoints, fold)
