import os
import mlflow
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def pairwise_similarity_stats(z, y):
    """
    z: [N, D] normalized embeddings
    y: [N] binary labels (1=CR, 0=NR)
    """

    sim = z @ z.T
    y = y.bool()

    def extract(mask_i, mask_j):
        return sim[mask_i][:, mask_j].flatten()

    cr_cr = extract(y, y)
    cr_nr = extract(y, ~y)
    nr_nr = extract(~y, ~y)

    return {
        "CR–CR": cr_cr,
        "CR–NR": cr_nr,
        "NR–NR": nr_nr
    }

def fisher_ratio(z, y):
    y = y.bool()
    mu_cr = z[y].mean(dim=0)
    mu_nr = z[~y].mean(dim=0)

    var_cr = z[y].var(dim=0).mean()
    var_nr = z[~y].var(dim=0).mean()

    return ((mu_cr - mu_nr).pow(2).mean()) / (var_cr + var_nr + 1e-8)

def linear_probe(z, y):
    clf = LogisticRegression(max_iter=500)
    clf.fit(z.numpy(), y.numpy())
    probs = clf.predict_proba(z.numpy())[:, 1]
    preds = clf.predict(z.numpy())

    auc = roc_auc_score(y.numpy(), probs)
    bacc = balanced_accuracy_score(y.numpy(), preds)

    return auc, bacc

def class_compactness(z, y):
    y = y.bool()

    def intra_class_dist(zc):
        sim = zc @ zc.T
        return (1 - sim).mean().item()

    return {
        "CR_compactness": intra_class_dist(z[y]),
        "NR_compactness": intra_class_dist(z[~y])
    }

def class_compactness(z, y):
    y = y.bool()

    def intra_class_dist(zc):
        sim = zc @ zc.T
        return (1 - sim).mean().item()

    return {
        "CR_compactness": intra_class_dist(z[y]),
        "NR_compactness": intra_class_dist(z[~y])
    }

def plot_umap(z, y, epoch, split):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(z.numpy())

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[y==0, 0], emb[y==0, 1], s=10, alpha=0.5, label="NR")
    plt.scatter(emb[y==1, 0], emb[y==1, 1], s=10, alpha=0.8, label="CR")
    plt.legend()
    plt.title("Latent space (UMAP)")
    plt.savefig(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png", dpi=300)
    plt.close()
    mlflow.log_artifact(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png")
    os.remove(f"./latent_space_umap_{split}_epoch_{str(epoch).zfill(3)}.png")

def analyze_latent_space(z, y, epoch, split):
    results = {}

    results["fisher_ratio"] = fisher_ratio(z, y)
    auc, bacc = linear_probe(z, y)
    results["linear_probe_auc"] = auc
    results["linear_probe_bacc"] = bacc
    results["compactness"] = class_compactness(z, y)

    sim_stats = pairwise_similarity_stats(z, y)
    results["similarity_means"] = {
        k: v.mean().item() for k, v in sim_stats.items()
    }

    plot_umap(z, y, epoch=epoch, split=split)

    # print(f"Latent space analysis results: {results}")
    mlflow.log_metric(f"fisher_ratio_{split}", results["fisher_ratio"].item(), step=epoch)
    mlflow.log_metric(f"linear_probe_auc_{split}", results["linear_probe_auc"], step=epoch)
    mlflow.log_metric(f"linear_probe_bacc_{split}", results["linear_probe_bacc"], step=epoch)
    mlflow.log_metric(f"CR_compactness_{split}", results["compactness"]["CR_compactness"], step=epoch)
    mlflow.log_metric(f"NR_compactness_{split}", results["compactness"]["NR_compactness"], step=epoch)
    for k, v in results["similarity_means"].items():
        k = k.replace("–", "_")
        mlflow.log_metric(f"similarity_mean_{k}_{split}", v, step=epoch)
    

    return results
