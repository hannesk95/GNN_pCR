import os
import random
import numpy as np
import torch
import mlflow
from pathlib import Path
import umap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_all_python_files(parent_dir="."):

    parent_dir = Path(parent_dir)
    py_files = list(parent_dir.rglob("*.py"))
    for py_file in py_files:
        if not "conda_env" in str(py_file):
            if not "mlruns" in str(py_file):
                mlflow.log_artifact(str(py_file))

def plot_umap(z, y, epoch, split):
    reducer = umap.UMAP()
    emb = reducer.fit_transform(z)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[y==0, 0], emb[y==0, 1], s=10, alpha=0.5, label="NR")
    plt.scatter(emb[y==1, 0], emb[y==1, 1], s=10, alpha=0.8, label="CR")
    plt.legend()
    plt.title("Latent space (UMAP)")
    plt.savefig(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png", dpi=300)
    plt.close()
    mlflow.log_artifact(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png")
    os.remove(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png")


def linear_probe(z_train, y_train, z_val, y_val):
    
    clf = LogisticRegression(max_iter=500)
    clf.fit(z_train, y_train)

    preds = clf.predict(z_val)
    probs = clf.predict_proba(z_val)[:, 1]    

    auc = roc_auc_score(y_val, probs)
    bacc = balanced_accuracy_score(y_val, preds)
    mcc = matthews_corrcoef(y_val, preds)

    return auc, bacc, mcc