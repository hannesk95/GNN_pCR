from glob import glob
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from data.Dataloader import get_train_dataloaders, get_val_dataloaders
import argparse
from utils.pretraining_engine import train_epoch, eval_epoch
from utils.pretraining_engine import train_epoch_kaczmarek, eval_epoch_kaczmarek
from utils.pretraining_engine import train_epoch_gnn, eval_epoch_gnn
from models.builder import build_model
import os
import random
from omegaconf import OmegaConf
import mlflow
from models.GNN import TemporalGNN, ResNet18EncoderGNN
from models.ResNet18 import ResNet18Encoder
from models.Kaczmarek import ResNet18EncoderKaczmarek
from data.Dataset import ISPY2
from torch.amp import autocast, GradScaler
import itertools
from sklearn.preprocessing import OneHotEncoder
from utils.AsymContrastiveLoss import AsymmetricContrastiveLoss, CRSupervisedContrastiveLoss

BATCH_SIZE = 16
ACCUMULATION_STEPS = 4

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config, fold, timepoints):
        
    # log params
    mlflow.log_param('fold', fold)   
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('epochs', config.epochs)
    
    # get dataloaders
    # train_dl = get_train_dataloaders(data_path=config.data_path, 
    #                                  batch_size=config.batch_size, 
    #                                  timepoints=config.timepoints, 
    #                                  system=config.system,
    #                                  fold=fold)

    train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints)

    # Count samples per class
    patient_ids = torch.load(f"/home/johannes/Data/SSD_2.0TB/GNN_pCR/data/breast_cancer/data_splits_{timepoints}_timepoints.pt")[fold]["train"]
    
    labels = []
    for patient_id in patient_ids:
        files = sorted(glob(f"/home/johannes/Data/SSD_2.0TB/GNN_pCR/data/breast_cancer/data_processed/{patient_id}/*.pt"))
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


    # train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=sampler)

    # val_dl = get_val_dataloaders(data_path=config.data_path, 
    #                              val_batch_size=config.val_batch_size, 
    #                              timepoints=config.timepoints,
    #                              system=config.system,
    #                              fold=fold)

    val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints)
    # val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # model set-up
    # model = build_model(mtan_masking=config.mtan_masking, filters=config.filters, device=config.device)
    if config.kaczmarek:
        model = ResNet18EncoderKaczmarek(timepoints=timepoints).to(config.device)

        perms = list(itertools.permutations(range(timepoints), timepoints))
        perms = [''.join([str(item) for item in p]) for p in perms]
        enc = OneHotEncoder()
        enc.fit(np.array(perms).reshape(-1, 1))

        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    else:
        model = ResNet18Encoder().to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.use_gnn:        

        model = ResNet18EncoderGNN().to(config.device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('cnn_num_params', model_num_params)

        model_gnn = TemporalGNN(in_channels=512, hidden_channels=256, out_channels=128, num_layers=2, aggregation="mean").to(config.device)
        model_gnn_num_params = sum(p.numel() for p in model_gnn.parameters() if p.requires_grad)
        mlflow.log_param('gnn_num_params', model_gnn_num_params)

        optimizer = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": model_gnn.parameters()},
            ],
            lr=1e-2,
            nesterov=True,
            momentum=0.99,
            weight_decay=3e-5,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        align_loss = AsymmetricContrastiveLoss(margin=0.0, lambda_neg=1.0, timepoints=timepoints).to(config.device)
        supcon_loss = CRSupervisedContrastiveLoss(temperature=0.1).to(config.device)
    
    scaler = GradScaler()

    best_val_loss = np.inf
    best_val_metric = 0.0

    for epoch in tqdm(range(1,config.epochs+1)):

        if config.kaczmarek:
            train_losses = train_epoch_kaczmarek(
                model=model,
                loader=train_dl,
                optimizer=optimizer,
                timepoints=timepoints,
                device=config.device,
                mode=config.mode,
                align_labels=config.align_labels,
                scaler=scaler,
                one_hot_encoder=enc,
                ce_loss=ce_loss
            )

            val_losses = eval_epoch_kaczmarek(
                model=model,
                loader=val_dl,
                timepoints=timepoints,
                device=config.device,
                align_labels=config.align_labels,
                mode=config.mode,
                scaler=scaler,
                one_hot_encoder=enc,
                ce_loss=ce_loss
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)
            mlflow.log_metric('train_sup_loss', train_losses['sup'], step=epoch)
            mlflow.log_metric('train_ce_loss', train_losses['ce'], step=epoch)

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)
            mlflow.log_metric('val_ce_loss', val_losses['ce'], step=epoch)
            mlflow.log_metric('val_sup_loss', val_losses['sup'], step=epoch)

        elif config.use_gnn:                
            
            train_losses = train_epoch_gnn(
                    model_cnn=model,
                    model_gnn=model_gnn,                    
                    loader=train_dl,
                    optimizer=optimizer,
                    timepoints=timepoints,
                    device=config.device,
                    mode=config.mode,
                    align_labels=config.align_labels,
                    scaler=scaler,
                    align_loss=align_loss,
                    supcon_loss=supcon_loss,
                    epoch=epoch, 
                    accumulation_steps=ACCUMULATION_STEPS,
                    lr_scheduler=lr_scheduler
            )

            val_losses, val_metrics = eval_epoch_gnn(
                model_cnn=model,
                model_gnn=model_gnn,                    
                loader=val_dl,
                timepoints=timepoints,
                device=config.device,
                align_labels=config.align_labels,
                mode=config.mode,
                scaler=scaler,
                align_loss=align_loss,
                supcon_loss=supcon_loss,
                epoch=epoch,
                accumulation_steps=ACCUMULATION_STEPS,
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)
            mlflow.log_metric('train_supcon_loss', train_losses['supcon'], step=epoch) 
            mlflow.log_metric('train_align_loss', train_losses['align'], step=epoch)            

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)
            mlflow.log_metric('val_supcon_loss', val_losses['supcon'], step=epoch)
            mlflow.log_metric('val_align_loss', val_losses['align'], step=epoch)

        else:
            
            train_losses = train_epoch(
                model=model,
                loader=train_dl,
                optimizer=optimizer,
                timepoints=timepoints,
                device=config.device,
                mode=config.mode,
                align_labels=config.align_labels,
                scaler=scaler
            )

            val_losses = eval_epoch(
                model=model,
                loader=val_dl,
                timepoints=timepoints,
                device=config.device,
                align_labels=config.align_labels,
                mode=config.mode,
                scaler=scaler
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)
            mlflow.log_metric('train_sup_loss', train_losses['sup'], step=epoch)
            mlflow.log_metric('train_temp_loss', train_losses['temp'], step=epoch)

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)
            mlflow.log_metric('val_temp_loss', val_losses['temp'], step=epoch)
            mlflow.log_metric('val_sup_loss', val_losses['sup'], step=epoch)            
        
        torch.save(model.state_dict(), 'model_latest.pt')
        mlflow.log_artifact('model_latest.pt')
        os.remove('model_latest.pt')    

        if config.use_gnn:
                torch.save(model_gnn.state_dict(), 'model_gnn_latest.pt')
                mlflow.log_artifact('model_gnn_latest.pt')
                os.remove('model_gnn_latest.pt')

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), 'model_best.pt')
            mlflow.log_artifact('model_best.pt')
            os.remove('model_best.pt')

            if config.use_gnn:
                    torch.save(model_gnn.state_dict(), 'model_gnn_best.pt')
                    mlflow.log_artifact('model_gnn_best.pt')
                    os.remove('model_gnn_best.pt')
        
        if val_metrics['val_bacc'] > best_val_metric:
            best_val_metric = val_metrics['val_bacc']
            torch.save(model.state_dict(), 'model_best_metric.pt')
            mlflow.log_artifact('model_best_metric.pt')
            os.remove('model_best_metric.pt')

            if config.use_gnn:
                    torch.save(model_gnn.state_dict(), 'model_gnn_best_metric.pt')
                    mlflow.log_artifact('model_gnn_best_metric.pt')
                    os.remove('model_gnn_best_metric.pt')
    
    mlflow.log_param('best_val_loss', best_val_loss)
    mlflow.log_param('best_val_bacc', best_val_metric)

if __name__ == '__main__': 
    set_deterministic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='/home/johannes/Data/SSD_2.0TB/GNN_pCR/configs/pretraining_sweep.yaml',
                        help='path to config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)    

    config.use_gnn = True
    config.kaczmarek = False
    config.janickova = False
    
    for timepoints in [4]:
        for fold in range(5):

            if config.use_gnn:
                mlflow.set_experiment('pretraining_GNN')
            elif config.kaczmarek:
                mlflow.set_experiment('pretraining_Kaczmarek')
            elif config.janickova:
                mlflow.set_experiment('pretraining_Janickova')
            else:
                raise ValueError("No model type selected!")

            with mlflow.start_run():
                main(config, fold, timepoints)