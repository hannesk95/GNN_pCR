import mlflow
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.ARTLoss import compute_alignment_loss, compute_temporal_loss, compute_reconstruction_loss
from utils.graph_utils import make_directed_complete_forward_graph, compute_graph_alignment_loss
from utils.latent_space_analysis import analyze_latent_space


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,
        
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

    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_total_epoch = []
    

    temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]

    for batch_data in tqdm(loader):

        optimizer.zero_grad()
        
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device)        
        
        with torch.amp.autocast("cuda"):
            latents = model(images)
            latents = nn.functional.normalize(latents)

            latents_original = latents[:, :timepoints, :]
            latents_transformed = latents[:, timepoints:, :]

            loss = 0.0
            for i in range(timepoints):
                for j in range(timepoints):
                    if i != j:                        
                        
                        latent_1 = latents_original[:, i, :]
                        latent_2 = latents_transformed[:, i, :]
                        latent_3 = latents_original[:, j, :]
                        margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

                        # Compute the temporal loss
                        loss_temp = compute_temporal_loss(latent_1, latent_2, latent_3, margin)
                        loss_temp_epoch.append(loss_temp.item())

                        # Compute the alignment loss for the given labels
                        loss_align_total = compute_alignment_loss(latent_1, latent_2, labels, align_labels)                        
                        loss_sup_epoch.append(loss_align_total.item())                        

                        # Combine losses
                        loss = loss + loss_temp + loss_align_total
                        loss_total_epoch.append(loss.item())
           
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'temp': sum(loss_temp_epoch) / len(loss_temp_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
    }

def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,
        
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

    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_total_epoch = []

    temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]

    with torch.no_grad():
        for batch_data in tqdm(loader):
        
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 

            with torch.amp.autocast("cuda"):
                latents = model(images)
                latents = nn.functional.normalize(latents)

                latents_original = latents[:, :timepoints, :]
                latents_transformed = latents[:, timepoints:, :]

                loss = 0.0
                for i in range(timepoints):
                    for j in range(timepoints):
                        if i != j:                        
                            
                            latent_1 = latents_original[:, i, :]
                            latent_2 = latents_transformed[:, i, :]
                            latent_3 = latents_original[:, j, :]
                            margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

                            # Compute the temporal loss
                            loss_temp = compute_temporal_loss(latent_1, latent_2, latent_3, margin)
                            loss_temp_epoch.append(loss_temp.item())

                            # Compute the alignment loss for the given labels
                            loss_align_total = compute_alignment_loss(latent_1, latent_2, labels, align_labels)                        
                            loss_sup_epoch.append(loss_align_total.item())                        

                            # Combine losses
                            loss = loss + loss_temp + loss_align_total
                            loss_total_epoch.append(loss.item())    

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'temp': sum(loss_temp_epoch) / len(loss_temp_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
    }

def train_epoch_kaczmarek(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,
        one_hot_encoder = None,
        ce_loss = None,
        
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

    loss_ce_epoch = []
    loss_sup_epoch = []
    loss_total_epoch = []

    # ce_loss = nn.CrossEntropyLoss()    

    # temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]

    for batch_data in tqdm(loader):

        optimizer.zero_grad()
        
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device)        
        
        with torch.amp.autocast("cuda"):
            logits, latents_original, latents_transformed, labels_oh = model(images)
            latents_original = nn.functional.normalize(latents_original)
            latents_transformed = nn.functional.normalize(latents_transformed)

            

            # loss = 0.0
            # for i in range(timepoints):     
            #     for j in range(timepoints):
            #         if i != j:                        
                        
            # latent_1 = latents_original[:, i, :]
            # latent_2 = latents_transformed[:, i, :]
            # latent_3 = latents_original[:, j, :]
            # margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

            # Compute the CE loss
            labels_oh = torch.tensor(one_hot_encoder.transform(labels_oh).toarray()).to(logits.device)
            loss_ce = ce_loss(logits, labels_oh)
            loss_ce_epoch.append(loss_ce.item())

            # Compute the alignment loss for the given labels
            loss_align_total = compute_alignment_loss(latents_original, latents_transformed, labels, align_labels)                        
            loss_sup_epoch.append(loss_align_total.item())                        

            # Combine losses
            loss = loss_ce + loss_align_total
            loss_total_epoch.append(loss.item())
           
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'ce': sum(loss_ce_epoch) / len(loss_ce_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
    }

def eval_epoch_kaczmarek(
        model: nn.Module,
        loader: DataLoader,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,
        one_hot_encoder = None,
        ce_loss = None,
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

    loss_ce_epoch = []
    loss_sup_epoch = []
    loss_total_epoch = []

    # ce_loss = nn.CrossEntropyLoss()

    # temporal_distances_mapping = [1.0, 0.75, 0.5, 0.25]

    with torch.no_grad():
        for batch_data in tqdm(loader):

            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device)        
            
            with torch.amp.autocast("cuda"):
                logits, latents_original, latents_transformed, labels_oh = model(images)
                latents_original = nn.functional.normalize(latents_original)
                latents_transformed = nn.functional.normalize(latents_transformed)
            
            # Compute the CE loss
            labels_oh = torch.tensor(one_hot_encoder.transform(labels_oh).toarray()).to(logits.device)
            loss_ce = ce_loss(logits, labels_oh)
            loss_ce_epoch.append(loss_ce.item())

            # Compute the alignment loss for the given labels
            loss_align_total = compute_alignment_loss(latents_original, latents_transformed, labels, align_labels)                        
            loss_sup_epoch.append(loss_align_total.item())                        

            # Combine losses
            loss = loss_ce + loss_align_total
            loss_total_epoch.append(loss.item())
        
            # images = batch_data[0].float().to(device)
            # labels = batch_data[1].long().to(device) 

            # with torch.amp.autocast("cuda"):
            #     latents = model(images)
            #     latents = nn.functional.normalize(latents)

            #     latents_original = latents[:, :timepoints, :]
            #     latents_transformed = latents[:, timepoints:, :]

            #     loss = 0.0
            #     for i in range(timepoints):
            #         for j in range(timepoints):
            #             if i != j:                        
                            
            #                 latent_1 = latents_original[:, i, :]
            #                 latent_2 = latents_transformed[:, i, :]
            #                 latent_3 = latents_original[:, j, :]
            #                 margin = np.abs(temporal_distances_mapping[i] - temporal_distances_mapping[j])

            #                 # Compute the temporal loss
            #                 loss_temp = compute_temporal_loss(latent_1, latent_2, latent_3, margin)
            #                 loss_temp_epoch.append(loss_temp.item())

            #                 # Compute the alignment loss for the given labels
            #                 loss_align_total = compute_alignment_loss(latent_1, latent_2, labels, align_labels)                        
            #                 loss_sup_epoch.append(loss_align_total.item())                        

            #                 # Combine losses
            #                 loss = loss + loss_temp + loss_align_total
            #                 loss_total_epoch.append(loss.item())    

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'ce': sum(loss_ce_epoch) / len(loss_ce_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
    }

def train_epoch_gnn(
        model_cnn: nn.Module,
        model_gnn: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,
        align_loss = None,      
        supcon_loss = None,  
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
    model_cnn.train()
    model_gnn.train()

    # Lists to track different losses for the epoch
    loss_align_epoch = []
    loss_supcon_epoch = []   
    loss_total_epoch = []

    optimizer.zero_grad()

    # Iterate over batches in the data loader
    for step, batch_data in tqdm(enumerate(loader), total=len(loader)):

        # optimizer.zero_grad()
            
        images = batch_data[0].float().to(device)
        labels = batch_data[1].long().to(device) 
        batch_size = images.size(0)          
        
        with torch.amp.autocast("cuda"):
            latents = model_cnn(images)
            # latents = nn.functional.normalize(latents, dim=-1)

            latents_original = latents[:, :timepoints, :]
            latents_transformed = latents[:, timepoints:, :]        

        # Create graph from latent embeddings and forward pass through GNN
        graph_data_original = make_directed_complete_forward_graph(latents_original, batch_size=batch_size)
        graph_data_original = graph_data_original.to(device)
        graph_data_original = graph_data_original.sort(sort_by_row=False)
        with torch.amp.autocast("cuda"):
            latents_original = model_gnn(graph_data_original.x, graph_data_original.edge_index)
        latents_original = latents_original.view(batch_size, -1, latents_original.size(-1))
        latents_original = latents_original.flatten(start_dim=1)     

        graph_data_transformed = make_directed_complete_forward_graph(latents_transformed, batch_size=batch_size)        
        graph_data_transformed = graph_data_transformed.to(device)
        graph_data_transformed = graph_data_transformed.sort(sort_by_row=False)
        with torch.amp.autocast("cuda"):
            latents_transformed = model_gnn(graph_data_transformed.x, graph_data_transformed.edge_index)
        latents_transformed = latents_transformed.view(batch_size, -1, latents_transformed.size(-1))
        latents_transformed = latents_transformed.flatten(start_dim=1)      
        
        latents = torch.concat([latents_original, latents_transformed], dim=0)
        labels_twice = torch.concat([labels, labels], dim=0)
        latents = nn.functional.normalize(latents, dim=-1)  
        
        # Compute losses     
        loss_align = align_loss(latents, labels_twice)  
        loss_align_epoch.append(loss_align.item())

        # loss_supcon = supcon_loss(torch.concat([latents_original, latents_transformed], dim=0), torch.concat([labels, labels], dim=0))
        # loss_supcon_epoch.append(loss_supcon.item())
        loss_supcon = 0.0
        loss_supcon_epoch.append(loss_supcon)

        loss = loss_align + loss_supcon        
        loss = loss / accumulation_steps # IMPORTANT: normalize loss for gradient accumulation
        loss_total_epoch.append(loss.item())

        # scaled backward
        scaler.scale(loss).backward()
    
        # perform optimizer step every accum_steps
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()     
    
    
    if lr_scheduler is not None:
        mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=epoch)        
        lr_scheduler.step()

    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'align': sum(loss_align_epoch) / len(loss_align_epoch),
        'supcon': sum(loss_supcon_epoch) / len(loss_supcon_epoch),
    }

def eval_epoch_gnn(
        model_cnn: nn.Module,
        model_gnn: nn.Module,
        loader: DataLoader,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        scaler = None,        
        align_loss = None,
        supcon_loss = None,
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
    model_cnn.eval()
    model_gnn.eval()

    # Lists to track different losses for the epoch
    loss_align_epoch = []
    loss_supcon_epoch = []   
    loss_total_epoch = []

    zs, ys = [], []

    with torch.no_grad():
        for batch_data in tqdm(loader):
         
            images = batch_data[0].float().to(device)
            labels = batch_data[1].long().to(device) 
            batch_size = images.size(0)        
            
            with torch.amp.autocast("cuda"):
                latents = model_cnn(images)
                latents = nn.functional.normalize(latents)

                latents_original = latents[:, :timepoints, :]
                latents_transformed = latents[:, timepoints:, :]            

            # Create graph from latent embeddings and forward pass through GNN
            graph_data_original = make_directed_complete_forward_graph(latents_original, batch_size=batch_size)
            graph_data_original = graph_data_original.to(device)
            graph_data_original = graph_data_original.sort(sort_by_row=False)
            with torch.amp.autocast("cuda"):
                latents_original = model_gnn(graph_data_original.x, graph_data_original.edge_index)
            latents_original = latents_original.view(batch_size, -1, latents_original.size(-1))
            latents_original = latents_original.flatten(start_dim=1)     

            graph_data_transformed = make_directed_complete_forward_graph(latents_transformed, batch_size=batch_size)        
            graph_data_transformed = graph_data_transformed.to(device)
            graph_data_transformed = graph_data_transformed.sort(sort_by_row=False)
            with torch.amp.autocast("cuda"):
                latents_transformed = model_gnn(graph_data_transformed.x, graph_data_transformed.edge_index)
            latents_transformed = latents_transformed.view(batch_size, -1, latents_transformed.size(-1))
            latents_transformed = latents_transformed.flatten(start_dim=1)      
            
            latents = torch.concat([latents_original, latents_transformed], dim=0)
            labels_twice = torch.concat([labels, labels], dim=0)
            latents = nn.functional.normalize(latents, dim=-1)  
            
            # Compute losses     
            loss_align = align_loss(latents, labels_twice)  
            loss_align_epoch.append(loss_align.item())

            # loss_supcon = supcon_loss(torch.concat([latents_original, latents_transformed], dim=0), torch.concat([labels, labels], dim=0))
            # loss_supcon_epoch.append(loss_supcon.item())
            loss_supcon = 0.0
            loss_supcon_epoch.append(loss_supcon)

            loss = loss_align + loss_supcon        
            loss = loss / accumulation_steps # IMPORTANT: normalize loss for gradient accumulation
            loss_total_epoch.append(loss.item())      

            zs.append(latents_original.cpu())
            ys.append(labels.cpu())
        
        results = analyze_latent_space(torch.cat(zs), torch.cat(ys), epoch=epoch, split="val")
        # print(f"Latent space analysis results: {results}")
        # mlflow.log_dict(results, 'latent_space_analysis_eval_gnn.json')
        # mlflow.log_artifact('/home/johannes/Data/SSD_2.0TB/GNN_pCR/latent_space_umap.png', artifact_path='.')
            
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'align': sum(loss_align_epoch) / len(loss_align_epoch),
        'supcon': sum(loss_supcon_epoch) / len(loss_supcon_epoch),
    }, {'val_bacc': results['linear_probe_bacc'], 'val_auc': results['linear_probe_auc']}

