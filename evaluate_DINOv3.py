import torch
from tqdm import tqdm
import numpy as np
import os
from data.Dataloader import get_train_dataloaders, get_val_dataloaders, get_test_dataloaders
import argparse
import yaml
# from utils.pretraining_engine import train_epoch, eval_epoch
from models.builder import build_model
# from models.GNN import TemporalGNN
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import mlflow
from omegaconf import OmegaConf
from utils.graph_utils import make_directed_complete_forward_graph
# from models.Janickova import ResNet18Encoder
from data.Dataset import ISPY2
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from glob import glob
import torch
import numpy as np
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from models.DINOv3 import DINOv3
from utils.utils import set_deterministic, log_all_python_files


def main(method, timepoints, fold):   

    # set_deterministic()  
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)      

      
    train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_2D=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)    

    val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_2D=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_2D=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)    

    model = DINOv3().to(device)
    model.eval()         
    

    # train data
    train_latents = []
    train_labels = []

    with torch.no_grad():
        for batch_data in tqdm(train_dl):

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            train_latents.append(latents.reshape(b, -1).cpu().numpy())            
            train_labels.extend(labels.cpu().numpy().tolist())            
    
    train_latents = np.concatenate(train_latents, axis=0)
    train_labels = np.array(train_labels)

    # val data         
    val_latents = []
    val_labels = []

    with torch.no_grad():
        for batch_data in tqdm(val_dl):   

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            val_latents.append(latents.reshape(b, -1).cpu().numpy())            
            val_labels.extend(labels.cpu().numpy().tolist())            
    
    val_latents = np.concatenate(val_latents, axis=0)
    val_labels = np.array(val_labels)
    
    # test data
    test_latents = []
    test_labels = []

    with torch.no_grad():
        for batch_data in tqdm(test_dl):      

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            test_latents.append(latents.reshape(b, -1).cpu().numpy())            
            test_labels.extend(labels.cpu().numpy().tolist())            
    
    test_latents = np.concatenate(test_latents, axis=0)
    test_labels = np.array(test_labels)

    # svm classifier

    X_train, y_train = train_latents, train_labels
    X_val, y_val = val_latents, val_labels
    X_test, y_test = test_latents, test_labels

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)    

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test) 

    clf = SVC(probability=True)
    # clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    train_probs = clf.predict_proba(X_train)[:,1]
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:,1]

    train_bacc = balanced_accuracy_score(y_train, train_preds)
    train_mcc = matthews_corrcoef(y_train, train_preds)
    train_cm = confusion_matrix(y_train, train_preds)   
    train_sensitivity = train_cm[1,1] / (train_cm[1,0] + train_cm[1,1])
    train_specificity = train_cm[0,0] / (train_cm[0,0] + train_cm[0,1])
    train_roc_auc = roc_auc_score(y_train, train_probs)
    
    test_bacc = balanced_accuracy_score(y_test, test_preds)
    test_mcc = matthews_corrcoef(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)   
    test_sensitivity = test_cm[1,1] / (test_cm[1,0] + test_cm[1,1])
    test_specificity = test_cm[0,0] / (test_cm[0,0] + test_cm[0,1])
    test_roc_auc = roc_auc_score(y_test, test_probs)
    
    mlflow.log_param(f'train_bacc', train_bacc)
    mlflow.log_param(f'train_mcc', train_mcc)
    mlflow.log_param(f'train_roc_auc', train_roc_auc)
    mlflow.log_param(f'train_sensitivity', train_sensitivity)
    mlflow.log_param(f'train_specificity', train_specificity)
    
    mlflow.log_param(f'test_bacc', test_bacc)
    mlflow.log_param(f'test_mcc', test_mcc)
    mlflow.log_param(f'test_roc_auc', test_roc_auc)    
    mlflow.log_param(f'test_sensitivity', test_sensitivity)
    mlflow.log_param(f'test_specificity', test_specificity)


if __name__ == '__main__':     

    set_deterministic()

    for method in ['DINOv3']:
        for timepoints in [4]:            
            for fold in range(5):
        
                mlflow.set_experiment("DINOv3")
                
                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():
                    main(method, timepoints, fold)
