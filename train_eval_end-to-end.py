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

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EPOCHS = 100
BATCH_SIZE = 32
SEED = 28
DTYPE = torch.float32

def seed_everything(seed: int):
    """Set seed for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_determinism(seed)
    # torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with a different but deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)    
    random.seed(worker_seed)


def main(fold: int, timepoints: int, architecture: str):

    seed_everything(seed=SEED)

    mlflow.log_param("fold", fold)
    mlflow.log_param("timepoints", timepoints)
    mlflow.log_param("architecture", architecture)
    
    # log all .py files in the project
    parent_dir = Path("/home/johannes/Data/SSD_2.0TB/GNN_pCR")
    py_files = list(parent_dir.rglob("*.py"))
    for py_file in py_files:
        if not "conda_env" in str(py_file):
            if not "mlruns" in str(py_file):
                mlflow.log_artifact(str(py_file))

    if architecture == "CNN_distLSTM":
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_time_dists=True)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_time_dists=True)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_time_dists=True)
    
    else:
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, generator=generator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, generator=generator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, generator=generator, pin_memory=True)

    if architecture == "CNN_LSTM":
        # model = CNNLSTM().cuda()
        model = CNNdistLSTM().cuda()
    elif architecture == "CNN_distLSTM":
        model = CNNdistLSTM().cuda()
    elif architecture == "CNN":
        model = CNN(num_timepoints=timepoints).cuda()    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # validation loss should decrease
        factor=0.1,          # reduce LR by a factor of 10
        patience=20,         # wait for 20 epochs without improvement
        threshold=1e-4,      # minimum change to qualify as improvement
        threshold_mode='rel',
        cooldown=0,          # no cooldown
        min_lr=1e-7,         # optional safety floor
    )

    best_val_roc_auc = 0.0
    for epoch in tqdm(range(EPOCHS)):

        train_loss_list = []
        train_true_list = []
        train_pred_list = []
        train_score_list = []

        model.train()
        for batch in tqdm(train_dataloader):
            # images = batch[:, :, :, :, :, :].cuda()  # (B, T, C, D, H, W)
            images = batch[0][:, timepoints:, :, :, :, :].cuda()  # (B, T, C, D, H, W)
            # labels = batch[:, 0, 0, 0, 0, 0].long().cuda()  # (B,)
            labels = batch[1].long().cuda()  # (B,)

            if architecture == "CNN_distLSTM":
                time_dists = batch[2].cuda()  # (B, T)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=DTYPE):
                if architecture == "CNN_distLSTM":
                    logits = model(images, time_dists)
                else:
                    logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logits = logits.to(torch.float32)
            preds = torch.argmax(logits, dim=1)
            train_true_list.extend(labels.cpu().numpy().tolist())
            train_pred_list.extend(preds.cpu().numpy().tolist())
            train_score_list.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
            train_loss_list.append(loss.item())

        val_loss_list = []
        val_true_list = []
        val_pred_list = []
        val_score_list = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                # images = batch[:, :, :, :, :, :].cuda()  # (B, T, C, D, H, W)
                images = batch[0][:, :timepoints, :, :, :, :].cuda()  # (B, T, C, D, H, W)
                # labels = batch[:, 0, 0, 0, 0, 0].long().cuda()  # (B,)
                labels = batch[1].long().cuda()  # (B,)

                if architecture == "CNN_distLSTM":
                    time_dists = batch[2].cuda()  # (B, T)

                with torch.amp.autocast("cuda"):
                    if architecture == "CNN_distLSTM":
                        logits = model(images, time_dists)
                    else:
                        logits = model(images)
                loss = loss_fn(logits, labels)

                logits = logits.to(torch.float32)
                preds = torch.argmax(logits, dim=1)
                val_true_list.extend(labels.cpu().numpy().tolist())
                val_pred_list.extend(preds.cpu().numpy().tolist())
                val_score_list.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
                val_loss_list.append(loss.item())
        
        train_bal_acc = balanced_accuracy_score(train_true_list, train_pred_list)
        try:
            train_roc_auc = roc_auc_score(train_true_list, train_score_list)
        except:
            train_roc_auc = 0.0
        train_loss = sum(train_loss_list) / len(train_loss_list)
        val_bal_acc = balanced_accuracy_score(val_true_list, val_pred_list)
        try:
            val_roc_auc = roc_auc_score(val_true_list, val_score_list)
        except:
            val_roc_auc = 0.0
        val_loss = sum(val_loss_list) / len(val_loss_list)

        mlflow.log_metric("train_bal_acc", train_bal_acc, step=epoch)
        mlflow.log_metric("train_roc_auc", train_roc_auc, step=epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_bal_acc", val_bal_acc, step=epoch)
        mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        scheduler.step(val_loss)
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            torch.save(model.state_dict(), "best_model.pt")
            mlflow.log_artifact("best_model.pt")
    
    # Test evaluation
    test_true_list = []
    test_pred_list = []
    test_score_list = []
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # images = batch[:, :, :, :, :, :].cuda()  # (B, T, C, D, H, W)
            images = batch[0][:, :timepoints, :, :, :, :].cuda()  # (B, T, C, D, H, W)
            # labels = batch[:, 0, 0, 0, 0, 0].long().cuda()  # (B,)
            labels = batch[1].long().cuda()  # (B,)

            if architecture == "CNN_distLSTM":
                time_dists = batch[2].cuda()  # (B, T)

            with torch.amp.autocast("cuda"):
                if architecture == "CNN_distLSTM":
                    logits = model(images, time_dists)
                else:
                    logits = model(images)

            logits = logits.to(torch.float32)
            preds = torch.argmax(logits, dim=1)
            test_true_list.extend(labels.cpu().numpy().tolist())
            test_pred_list.extend(preds.cpu().numpy().tolist())
            test_score_list.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())
    
    test_bal_acc = balanced_accuracy_score(test_true_list, test_pred_list)
    test_roc_auc = roc_auc_score(test_true_list, test_score_list)
    mlflow.log_metric("test_bal_acc", test_bal_acc)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    os.remove("best_model.pt")

if __name__ == "__main__":

    # architecture = "CNN_LSTM"j
    # architecture = "CNN"
    # architecture = "CNN_distLSTM"

    for architecture in ["CNN", "CNN_LSTM", "CNN_distLSTM"]:
        for timepoints in [3, 4]:
            for fold in range(5):
                mlflow.set_experiment("end-to-end")
                with mlflow.start_run():
                    main(fold, timepoints, architecture)
