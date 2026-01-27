import torch
from tqdm import tqdm
import numpy as np
import os
# from data.Dataloader import get_train_dataloaders, get_val_dataloaders, get_test_dataloaders
import argparse
import yaml
# from utils.pretraining_engine import train_epoch, eval_epoch
from models.builder import build_model
# from models.GNN import TemporalGNN
import os
import random
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
# import umap
from sklearn.preprocessing import StandardScaler
import mlflow
# from omegaconf import OmegaConf
from utils.graph_utils import make_directed_complete_forward_graph
# from models.Janickova import ResNet18Encoder
from data.Dataset import ISPY2
from sklearn.decomposition import PCA
# from imblearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from glob import glob
import torch
import numpy as np
from sklearn.decomposition import PCA
# from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import mlflow
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import itertools
# from models.DINOv3 import DINOv3
from models.Kaczmarek import ResNet18EncoderKaczmarek
from utils.utils import set_deterministic, log_all_python_files, seed_worker
from utils.pretraining_engine import inference_janickova
from utils.pretraining_engine import inference_kaczmarek
from utils.pretraining_engine import inference_kiechle
from models.Kiechle import ResNet18EncoderKiechle
from models.Janickova import ResNet18EncoderJanickova
from models.Kaczmarek import ResNet18EncoderKaczmarek
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

def main(method, timepoints, fold, checkpoint_path):
    
    set_deterministic()   
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)   

    # datasets
    train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints)
    val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints)
    test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints)    

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    if method == "kaczmarek":
        model = ResNet18EncoderKaczmarek(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif method == "janickova":
        model = ResNet18EncoderJanickova().to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif method == "kiechle":
        model = ResNet18EncoderKiechle(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)    

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))        

    if method == "kiechle":
        embeddings, labels = inference_kiechle(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device)
    
    elif method == "kaczmarek":
        embeddings, labels = inference_kaczmarek(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device)
    
    elif method == "janickova":
        embeddings, labels = inference_janickova(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device)          

    X_train, y_train = embeddings["train"], labels["train"]
    X_val, y_val = embeddings["val"], labels["val"]
    X_test, y_test = embeddings["test"], labels["test"]

    # sklearn

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])

    param_grid = {
        "pca__n_components": [0.90, 0.95, 0.99],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001]
    }

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    outer_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X_trainval, y_trainval), 1
    ):
        X_tr, X_va = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_va = y_trainval[train_idx], y_trainval[val_idx]

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_tr, y_tr)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_va)

        score = balanced_accuracy_score(y_va, y_pred)
        outer_scores.append(score)

        print(f"Outer fold {fold} | Balanced Acc: {score:.3f}")

    print(
        f"\nNested CV Balanced Accuracy: "
        f"{np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}"
    )

    final_grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=-1
    )

    final_grid.fit(X_trainval, y_trainval)
    final_model = final_grid.best_estimator_

    print("Best parameters:", final_grid.best_params_)

    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]

    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    # MCC
    mcc = matthews_corrcoef(y_test, y_test_pred)

    # AUROC
    auroc = roc_auc_score(y_test, y_test_prob)

    # Sensitivity / Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\nTest set performance")
    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"MCC               : {mcc:.3f}")
    print(f"AUROC             : {auroc:.3f}")
    print(f"Sensitivity       : {sensitivity:.3f}")
    print(f"Specificity       : {specificity:.3f}")

    mlflow.log_metric('test_balanced_accuracy', bal_acc)
    mlflow.log_metric('test_mcc', mcc)
    mlflow.log_metric('test_auroc', auroc)
    mlflow.log_metric('test_sensitivity', sensitivity)
    mlflow.log_metric('test_specificity', specificity)
        

if __name__ == '__main__':  

    janickova_dict = {
        "fold_0_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/6b5e7d28c10f45bb9764249674071194/artifacts/model_best_metric.pt",
        "fold_1_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/f8d20bc2b7f24ef59d32cd39bc450822/artifacts/model_best_metric.pt",
        "fold_2_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/c9e29e39047a4ecb875e9aefa11dde41/artifacts/model_best_metric.pt",
        "fold_3_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/8ca90dc5a545447ab92ad3558021bf9a/artifacts/model_best_metric.pt",
        "fold_4_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/d3122546d2584227bbebb1ac69099abe/artifacts/model_best_metric.pt",
    }

    kaczmarek_dict = {
        "fold_0_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/0a21a08722934f0c9828562f28c40228/artifacts/model_best_metric.pt",
        "fold_1_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/365c1881d08842c991f946842dd50099/artifacts/model_best_metric.pt",
        "fold_2_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/9001a9be12e748109ca1733bf94b25ff/artifacts/model_best_metric.pt",
        "fold_3_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/db5f8ab18a6c43f3ac0683ae7b0afe08/artifacts/model_best_metric.pt",
        "fold_4_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/15990b1778f048e6af52e03a94d2632c/artifacts/model_best_metric.pt",
    }
    
    kiechle_dict = {
        "fold_0_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/a00da610e4ec4ba786ef63abbdac5ab7/artifacts/model_best_metric.pt",
        "fold_1_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/c724a86d33284fc298283f32ff207a61/artifacts/model_best_metric.pt",
        "fold_2_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/c724a86d33284fc298283f32ff207a61/artifacts/model_best_metric.pt",
        "fold_3_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/ab7a282b007a44dcb8abfd041b2db7ac/artifacts/model_best_metric.pt",
        "fold_4_best_metric": "/home/johannes/Code/GNN_pCR/mlruns/3/90b66fc6c2d94dbea2e90681b54b218e/artifacts/model_best_metric.pt",
    }

    checkpoints_dict = {
        "janickova": janickova_dict,
        "kaczmarek": kaczmarek_dict,
        "kiechle": kiechle_dict,
    }    

    for method in checkpoints_dict.keys():        
        for timepoints in [4]:
            for fold in range(5):
                
                checkpoint_path = checkpoints_dict[method][f"fold_{fold}_best_metric"]
                mlflow.set_experiment("self-supervised-pretraining_evaluation")
                
                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():          
                    main(method, timepoints, fold, checkpoint_path)
