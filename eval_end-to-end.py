import argparse
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g. 0 or 1)",
    )
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import mlflow
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import GradScaler

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from data.Dataset import ISPY2
from models.CNN import CNN
from models.CNN_distLSTM import CNNdistLSTM
from utils.utils import log_all_python_files, seed_worker, set_deterministic


BATCH_SIZE = 16

def main(method, timepoints, fold, checkpoint_path):   

    set_deterministic()  
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)   
       
    mlflow.log_param('batch_size', BATCH_SIZE) 
    mlflow.log_param('device', device)   

    if method == "CNN_distLSTM":
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_time_dists=True)
    
    else:
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints)

    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    if method == "CNN_LSTM":
        # model = CNNLSTM().cuda()
        model = CNNdistLSTM().cuda()
    elif method == "CNN_distLSTM":
        model = CNNdistLSTM().cuda()
    elif method == "CNN":
        model = CNN(num_timepoints=timepoints).cuda()    
    else:
        raise ValueError(f"Unknown method: {method}")    
    
    # Test evaluation
    test_true_list = []
    test_pred_list = []
    test_score_list = []

    model.load_state_dict(torch.load(checkpoint_path))

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
                
                logits = logits.detach()
                preds = torch.argmax(logits, dim=1)
                scores = torch.softmax(logits, dim=1)[:, 1]        
           
            test_true_list.extend(labels.cpu().numpy().tolist())
            test_pred_list.extend(preds.cpu().numpy().tolist())
            test_score_list.extend(scores.cpu().numpy().tolist())
    
    test_bal_acc = balanced_accuracy_score(test_true_list, test_pred_list)
    test_roc_auc = roc_auc_score(test_true_list, test_score_list)
    test_mcc = matthews_corrcoef(test_true_list, test_pred_list)
    test_cm = confusion_matrix(test_true_list, test_pred_list)
    test_sensitivity = test_cm[1,1] / (test_cm[1,0] + test_cm[1,1])
    test_specificity = test_cm[0,0] / (test_cm[0,0] + test_cm[0,1])

    mlflow.log_metric("test_mcc", test_mcc)
    mlflow.log_metric("test_sensitivity", test_sensitivity)
    mlflow.log_metric("test_specificity", test_specificity)
    mlflow.log_metric("test_bal_acc", test_bal_acc)
    mlflow.log_metric("test_roc_auc", test_roc_auc)    

if __name__ == "__main__":    

    cnn_dict = {
        "fold_0_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/6b5e7d28c10f45bb9764249674071194/artifacts/model_best_metric.pt",
        "fold_1_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/f8d20bc2b7f24ef59d32cd39bc450822/artifacts/model_best_metric.pt",
        "fold_2_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/c9e29e39047a4ecb875e9aefa11dde41/artifacts/model_best_metric.pt",
        "fold_3_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/8ca90dc5a545447ab92ad3558021bf9a/artifacts/model_best_metric.pt",
        "fold_4_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/d3122546d2584227bbebb1ac69099abe/artifacts/model_best_metric.pt",
    }

    cnn_lstm_dict = {
        "fold_0_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/0a21a08722934f0c9828562f28c40228/artifacts/model_best_metric.pt",
        "fold_1_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/365c1881d08842c991f946842dd50099/artifacts/model_best_metric.pt",
        "fold_2_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/9001a9be12e748109ca1733bf94b25ff/artifacts/model_best_metric.pt",
        "fold_3_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/db5f8ab18a6c43f3ac0683ae7b0afe08/artifacts/model_best_metric.pt",
        "fold_4_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/15990b1778f048e6af52e03a94d2632c/artifacts/model_best_metric.pt",
    }    

    checkpoints_dict = {
        "CNN": cnn_dict,
        "CNN_LSTM": cnn_lstm_dict,
    }    

    for method in checkpoints_dict.keys():  
        for timepoints in [3]:
            for fold in range(5):

                checkpoint_path = checkpoints_dict[method][f"fold_{fold}_best_metric"]

                mlflow.set_experiment("end-to-end_3_timepoints")

                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():
                    main(method, timepoints, fold, checkpoint_path)
