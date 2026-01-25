import mlflow
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.ARTLoss import compute_alignment_loss, compute_temporal_loss, compute_reconstruction_loss
from utils.graph_utils import make_directed_complete_forward_graph, compute_graph_alignment_loss
from utils.latent_space_analysis import analyze_latent_space
from utils.AsymContrastiveLoss import AsymmetricContrastiveLoss, CRSupervisedContrastiveLoss


def train_epoch_janickova(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
        align_labels: list = [1.0],
        scaler = None,
        epoch = None,
        accumulation_steps = None,
        lr_scheduler = None
        
):
    """
    Train the model for one epoch.

    This function trains the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch and updates the model parameters.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): DataLoader providing the batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (total, temporal, supervised, reconstruction).
    """
    model.train()

    loss_temporal_epoch = []
    loss_align_epoch = []
    loss_total_epoch = []    

    temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]
    optimizer.zero_grad()

    for step, batch_data in tqdm(enumerate(loader), total=len(loader)):        
        
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device)        
        
        with torch.amp.autocast("cuda"):
            latents = model(images)
            latents = nn.functional.normalize(latents)

            latents_original = latents[:, :timepoints, :]
            latents_transformed = latents[:, timepoints:, :]

            loss_temporal_running = 0.0
            loss_align_running = 0.0
            for i in range(timepoints):
                for j in range(timepoints):
                    if i != j:                        
                        
                        latent_1 = latents_original[:, i, :]
                        latent_2 = latents_transformed[:, i, :]
                        latent_3 = latents_original[:, j, :]
                        margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

                        # Compute losses
                        loss_temporal = compute_temporal_loss(latent_1, latent_2, latent_3, margin)                        
                        loss_align = compute_alignment_loss(latent_1, latent_2, labels, align_labels)   

                        loss_temporal_running += loss_temporal
                        loss_align_running += loss_align                     
                                               

        loss_temporal = loss_temporal_running / accumulation_steps
        loss_align = loss_align_running / accumulation_steps
        loss = loss_temporal + loss_align
        scaler.scale(loss).backward()

        loss_temporal_epoch.append(loss_temporal.item())
        loss_align_epoch.append(loss_align.item()) 
        loss_total_epoch.append(loss.item())
           
        
        # perform optimizer step every accum_steps
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    if lr_scheduler is not None:
        mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=epoch)        
        lr_scheduler.step()                

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch),
        'align': sum(loss_align_epoch) / len(loss_align_epoch),
    }

def eval_epoch_janickova(
        model: nn.Module,
        loader: DataLoader,
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
        align_labels: list = [1.0],
        epoch = None,
        accumulation_steps = None,
):
    """
    Evaluate the model for one epoch.

    This function evaluates the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch without 
    updating the model parameters.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader providing the batches of data.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (temporal, supervised, reconstruction).
    """
    model.eval()

    loss_temporal_epoch = []
    loss_align_epoch = []
    loss_total_epoch = []    

    zs, ys = [], []

    temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]

    with torch.no_grad():
        for batch_data in tqdm(loader):
        
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)     

            B, T, C, D, H, W = images.shape    
            
            with torch.amp.autocast("cuda"):
                latents = model(images)
                latents = nn.functional.normalize(latents)

                latents_original = latents[:, :timepoints, :]
                latents_transformed = latents[:, timepoints:, :]

                loss_temporal_running = 0.0
                loss_align_running = 0.0
                for i in range(timepoints):
                    for j in range(timepoints):
                        if i != j:                        
                            
                            latent_1 = latents_original[:, i, :]
                            latent_2 = latents_transformed[:, i, :]
                            latent_3 = latents_original[:, j, :]
                            margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

                            # Compute losses
                            loss_temporal = compute_temporal_loss(latent_1, latent_2, latent_3, margin)                        
                            loss_align = compute_alignment_loss(latent_1, latent_2, labels, align_labels)   

                            loss_temporal_running += loss_temporal
                            loss_align_running += loss_align                                                

            loss_temporal = loss_temporal_running / accumulation_steps
            loss_align = loss_align_running / accumulation_steps
            loss = loss_temporal + loss_align

            loss_temporal_epoch.append(loss_temporal.item())
            loss_align_epoch.append(loss_align.item()) 
            loss_total_epoch.append(loss.item())     

            latents_original = latents_original.view(B, -1)
            zs.append(latents_original[:B, :].cpu())
            ys.append(labels.cpu()) 
        
        results = analyze_latent_space(torch.cat(zs), torch.cat(ys), epoch=epoch, split="val")
        
        losses = {'total': sum(loss_total_epoch) / len(loss_total_epoch),
                  'align': sum(loss_align_epoch) / len(loss_align_epoch),
                  'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch)}
        
        metrics = {'val_auc': results['linear_probe_auc'],
                   'val_bacc': results['linear_probe_bacc']}
    
    return losses, metrics        

def inference_janickova(
        model: nn.Module,                
        loader: DataLoader,
        device: torch.device = 'cuda'
):
    model.eval()
    
    z_train, y_train = [], []
    
    with torch.no_grad():
        for batch_data in tqdm(loader[0]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)

            B, T, C, D, H, W = images.shape      
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)   
            latents = latents[:B, :T, :].reshape(B, -1)             

            z_train.append(latents[:B, :].cpu())
            y_train.append(labels.cpu())
    
    z_train = torch.cat(z_train)
    y_train = torch.cat(y_train)
    
    z_val, y_val = [], []
    
    with torch.no_grad():
        for batch_data in tqdm(loader[1]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)
            latents = latents[:B, :T, :].reshape(B, -1)                     

            z_val.append(latents[:B, :].cpu())
            y_val.append(labels.cpu())
    
    z_val = torch.cat(z_val)
    y_val = torch.cat(y_val)
    
    z_test, y_test = [], []

    with torch.no_grad():
        for batch_data in tqdm(loader[2]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)   
            latents = latents[:B, :T, :].reshape(B, -1)                  

            z_test.append(latents[:B, :].cpu())
            y_test.append(labels.cpu())
    
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    
    embeddings = {
        'train': z_train,
        'val': z_val,
        'test': z_test
    }
    
    labels = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    return embeddings, labels

def train_epoch_kaczmarek(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device = 'cuda',
        align_labels: list = [1.0],        
        scaler = None,
        one_hot_encoder = None,
        epoch = None,
        accumulation_steps = None,
        lr_scheduler = None
):
    """
    Train the model for one epoch.

    This function trains the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch and updates the model parameters.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): DataLoader providing the batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (total, temporal, supervised, reconstruction).
    """
    model.train()

    loss_temporal_epoch = []
    loss_align_epoch = []
    loss_total_epoch = []

    ce_loss = torch.nn.CrossEntropyLoss()    

    optimizer.zero_grad()

    for step, batch_data in tqdm(enumerate(loader), total=len(loader)):
        
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device)        
        
        with torch.amp.autocast("cuda"):
            logits, latents_original, latents_transformed, labels_oh = model(images)
            latents_original = nn.functional.normalize(latents_original)
            latents_transformed = nn.functional.normalize(latents_transformed)           

            # Compute losses
            labels_oh = torch.tensor(one_hot_encoder.transform(labels_oh).toarray()).to(logits.device)
            loss_temporal = ce_loss(logits, labels_oh)
            loss_align = compute_alignment_loss(latents_original, latents_transformed, labels, align_labels) 
        
        loss_temporal = loss_temporal / accumulation_steps 
        loss_align = loss_align / accumulation_steps
        loss = loss_temporal + loss_align
        scaler.scale(loss).backward()

        loss_temporal_epoch.append(loss_temporal.item())
        loss_align_epoch.append(loss_align.item()) 
        loss_total_epoch.append(loss.item())       

        # perform optimizer step every accum_steps
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if lr_scheduler is not None:
        mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=epoch)        
        lr_scheduler.step()    

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch),
        'align': sum(loss_align_epoch) / len(loss_align_epoch),
    }

def eval_epoch_kaczmarek(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device = 'cuda',
        align_labels: list = [1.0],        
        one_hot_encoder = None,
        epoch = None,
        accumulation_steps = None,
):
    """
    Evaluate the model for one epoch.

    This function evaluates the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch without 
    updating the model parameters.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader providing the batches of data.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (temporal, supervised, reconstruction).
    """
    model.eval()

    loss_temporal_epoch = []
    loss_align_epoch = []
    loss_total_epoch = []

    ce_loss = torch.nn.CrossEntropyLoss() 

    zs, ys = [], []

    with torch.no_grad():
        for batch_data in tqdm(loader):

            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)        
        
            with torch.amp.autocast("cuda"):
                logits, latents_original, latents_transformed, labels_oh = model(images)
                latents_original = nn.functional.normalize(latents_original)
                latents_transformed = nn.functional.normalize(latents_transformed)           

                # Compute losses
                labels_oh = torch.tensor(one_hot_encoder.transform(labels_oh).toarray()).to(logits.device)
                loss_temporal = ce_loss(logits, labels_oh)
                loss_align = compute_alignment_loss(latents_original, latents_transformed, labels, align_labels) 
            
            loss_temporal = loss_temporal / accumulation_steps
            loss_align = loss_align / accumulation_steps
            loss = loss_temporal + loss_align

            loss_temporal_epoch.append(loss_temporal.item())
            loss_align_epoch.append(loss_align.item()) 
            loss_total_epoch.append(loss.item())   

            zs.append(latents_original.cpu())
            ys.append(labels.cpu()) 

    results = analyze_latent_space(torch.cat(zs), torch.cat(ys), epoch=epoch, split="val")
    losses = {'total': sum(loss_total_epoch) / len(loss_total_epoch),
              'align': sum(loss_align_epoch) / len(loss_align_epoch),
              'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch)}
    metrics = {'val_auc': results['linear_probe_auc'],
               'val_bacc': results['linear_probe_bacc']}


    return losses, metrics

def inference_kaczmarek(
        model: nn.Module,                
        loader: DataLoader,
        device: torch.device = 'cuda'
):
    model.eval()
    z_train, y_train = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[0]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)

            B, T, C, D, H, W = images.shape      
            
            _, latents_original, _, _ = model(images)                
            latents_original = nn.functional.normalize(latents_original)                   

            z_train.append(latents_original[:B, :].cpu())
            y_train.append(labels.cpu())
    
    z_train = torch.cat(z_train)
    y_train = torch.cat(y_train)

    z_val, y_val = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[1]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            _, latents_original, _, _ = model(images)                
            latents_original = nn.functional.normalize(latents_original)                   

            z_val.append(latents_original[:B, :].cpu())
            y_val.append(labels.cpu())
    
    z_val = torch.cat(z_val)
    y_val = torch.cat(y_val)    

    z_test, y_test = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[2]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            _, latents_original, _, _ = model(images)                
            latents_original = nn.functional.normalize(latents_original)                   

            z_test.append(latents_original[:B, :].cpu())
            y_test.append(labels.cpu())
    
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    
    embeddings = {
        'train': z_train,
        'val': z_val,
        'test': z_test
    }
    
    labels = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    return embeddings, labels
    
def train_epoch_kiechle(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        timepoints: int,       
        device: torch.device = 'cuda',
        scaler = None, 
        epoch = None,
        accumulation_steps = None,
        lr_scheduler = None,
):
    """
    Train the GNN model for one epoch.

    This function trains the GNN model over all batches in the provided data loader. It calculates 
    the loss for each batch and updates the model parameters.

    Args:
        model (nn.Module): The GNN model to be trained.
        loader (DataLoader): DataLoader providing the batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the model on.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()

    loss_fn = AsymmetricContrastiveLoss(margin=0.0, lambda_neg=1.0, timepoints=timepoints).to(device)

    # Lists to track different losses for the epoch
    loss_align_epoch = []
    loss_temporal_epoch = []   
    loss_orthogonal_epoch = []
    loss_total_epoch = []

    optimizer.zero_grad()

    # Iterate over batches in the data loader
    for step, batch_data in tqdm(enumerate(loader), total=len(loader)):
            
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device)      
        
        with torch.amp.autocast("cuda"):
            latents = model(images)            
            latents = nn.functional.normalize(latents, dim=-1)  
        
            # Compute losses     
            labels_twice = torch.concat([labels, labels], dim=0)
            losses = loss_fn(latents, labels_twice)        
        
        loss_align = losses['align'] / accumulation_steps
        loss_temporal = losses['temporal'] / accumulation_steps
        loss_orthogonal = losses['orthogonal'] / accumulation_steps
        loss = loss_align + loss_temporal + loss_orthogonal
        scaler.scale(loss).backward()

        loss_align_epoch.append(loss_align.item())
        loss_temporal_epoch.append(loss_temporal.item())
        loss_orthogonal_epoch.append(loss_orthogonal.item())
        loss_total_epoch.append(loss.item())      
    
        # perform optimizer step every accum_steps
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()   

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    if lr_scheduler is not None:
        mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=epoch)        
        lr_scheduler.step()

    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'align': sum(loss_align_epoch) / len(loss_align_epoch),
        'orthogonal': sum(loss_orthogonal_epoch) / len(loss_orthogonal_epoch),
        'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch)
    }

def eval_epoch_kiechle(
        model: nn.Module,
        loader: DataLoader,
        timepoints: int,
        device: torch.device = 'cuda',
        epoch = None,
        accumulation_steps = None,
):
    """
    Evaluate the GNN model for one epoch.

    This function evaluates the GNN model over all batches in the provided data loader. It calculates 
    the loss for each batch without updating the model parameters.

    Args:
        model (nn.Module): The GNN model to be evaluated.
        loader (DataLoader): DataLoader providing the batches of data.
        device (torch.device): Device to run the model on.

    Returns:
        float: The average loss for the epoch.
    """
    model.eval()

    loss_fn = AsymmetricContrastiveLoss(margin=0.0, lambda_neg=1.0, timepoints=timepoints).to(device)

    # Lists to track different losses for the epoch
    loss_align_epoch = []
    loss_temporal_epoch = []   
    loss_orthogonal_epoch = []
    loss_total_epoch = []

    zs, ys = [], []

    with torch.no_grad():
        for batch_data in tqdm(loader):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)

            B, T, C, D, H, W = images.shape      
            
            with torch.amp.autocast("cuda"):
                latents = model(images)            
                latents = nn.functional.normalize(latents, dim=-1)  
            
                # Compute losses     
                labels_twice = torch.concat([labels, labels], dim=0)
                losses = loss_fn(latents, labels_twice)        
            
            loss_align = losses['align'] / accumulation_steps
            loss_temporal = losses['temporal'] / accumulation_steps
            loss_orthogonal = losses['orthogonal'] / accumulation_steps
            loss = loss_align + loss_temporal + loss_orthogonal

            loss_align_epoch.append(loss_align.item())
            loss_temporal_epoch.append(loss_temporal.item())
            loss_orthogonal_epoch.append(loss_orthogonal.item())
            loss_total_epoch.append(loss.item())       

            zs.append(latents[:B, :].cpu())
            ys.append(labels.cpu())
        
        results = analyze_latent_space(torch.cat(zs), torch.cat(ys), epoch=epoch, split="val")
        
        losses = {'total': sum(loss_total_epoch) / len(loss_total_epoch),
                  'align': sum(loss_align_epoch) / len(loss_align_epoch),
                  'orthogonal': sum(loss_orthogonal_epoch) / len(loss_orthogonal_epoch),
                  'temporal': sum(loss_temporal_epoch) / len(loss_temporal_epoch)}

        metrics = {'val_auc': results['linear_probe_auc'],
                   'val_bacc': results['linear_probe_bacc']}
            
    return losses, metrics

def inference_kiechle(
        model: nn.Module,                
        loader: DataLoader,
        device: torch.device = 'cuda'
):
    
    model.eval()

    z_train, y_train = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[0]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)

            B, T, C, D, H, W = images.shape      
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)                   

            z_train.append(latents[:B, :].cpu())
            y_train.append(labels.cpu())
    
    z_train = torch.cat(z_train)
    y_train = torch.cat(y_train)

    z_val, y_val = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[1]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)                   

            z_val.append(latents[:B, :].cpu())
            y_val.append(labels.cpu())
    
    z_val = torch.cat(z_val)
    y_val = torch.cat(y_val)

    z_test, y_test = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader[2]):
            
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            B, T, C, D, H, W = images.shape     
            
            latents = model(images)                
            latents = nn.functional.normalize(latents, dim=-1)                   

            z_test.append(latents[:B, :].cpu())
            y_test.append(labels.cpu())
    
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)

    embeddings = {
        'train': z_train,
        'val': z_val,
        'test': z_test
    }

    labels = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }

    return embeddings, labels

    