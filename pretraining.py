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

import torch
import mlflow
import itertools
import numpy as np

from tqdm import tqdm
from glob import glob
from utils.pretraining_engine import train_epoch_janickova, eval_epoch_janickova, inference_janickova
from utils.pretraining_engine import train_epoch_kaczmarek, eval_epoch_kaczmarek, inference_kaczmarek
from utils.pretraining_engine import train_epoch_kiechle, eval_epoch_kiechle, inference_kiechle
from models.Kiechle import ResNet18EncoderKiechle
from models.Janickova import ResNet18EncoderJanickova
from models.Kaczmarek import ResNet18EncoderKaczmarek
from data.Dataset import ISPY2
from torch.amp import GradScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.utils import set_deterministic, log_all_python_files

BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 100
ALIGN_LABELS = [1.0]

def main(method, timepoints, fold):
    
    set_deterministic()   
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

    # datasets
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

    if method == "kaczmarek":
        model = ResNet18EncoderKaczmarek(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

        perms = list(itertools.permutations(range(timepoints), timepoints))
        perms = [''.join([str(item) for item in p]) for p in perms]
        enc = OneHotEncoder()
        enc.fit(np.array(perms).reshape(-1, 1))

    elif method == "janickova":
        model = ResNet18EncoderJanickova().to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif method == "kiechle":
        model = ResNet18EncoderKiechle(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)        
    
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, nesterov=True, momentum=0.99, weight_decay=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = np.inf
    best_val_metric = -np.inf

    for epoch in tqdm(range(1,EPOCHS+1)):

        if method == "kaczmarek":
            train_losses = train_epoch_kaczmarek(
                model=model,
                loader=train_dl,
                optimizer=optimizer,
                device=device,
                align_labels=ALIGN_LABELS,
                scaler=scaler,
                one_hot_encoder=enc,
                epoch=epoch, 
                accumulation_steps=ACCUMULATION_STEPS,
                lr_scheduler=lr_scheduler
            )

            val_losses, val_metrics = eval_epoch_kaczmarek(
                model=model,
                loader=val_dl,
                device=device,
                align_labels=ALIGN_LABELS,
                one_hot_encoder=enc,
                epoch=epoch,
                accumulation_steps=ACCUMULATION_STEPS
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)
            mlflow.log_metric('train_align_loss', train_losses['align'], step=epoch)
            mlflow.log_metric('train_temporal_loss', train_losses['temporal'], step=epoch)

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)
            mlflow.log_metric('val_align_loss', val_losses['align'], step=epoch)
            mlflow.log_metric('val_temporal_loss', val_losses['temporal'], step=epoch)
            
        elif method == "kiechle":                
            
            train_losses = train_epoch_kiechle(
                    model=model,                    
                    loader=train_dl,
                    optimizer=optimizer,
                    timepoints=timepoints,
                    device=device,                   
                    scaler=scaler,                
                    epoch=epoch, 
                    accumulation_steps=ACCUMULATION_STEPS,
                    lr_scheduler=lr_scheduler
            )

            val_losses, val_metrics = eval_epoch_kiechle(
                model=model,                   
                loader=val_dl,
                timepoints=timepoints,
                device=device,           
                epoch=epoch,
                accumulation_steps=ACCUMULATION_STEPS
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)            
            mlflow.log_metric('train_align_loss', train_losses['align'], step=epoch)         
            mlflow.log_metric('train_temporal_loss', train_losses['temporal'], step=epoch)        
            mlflow.log_metric('train_orthogonal_loss', train_losses['orthogonal'], step=epoch)  

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)            
            mlflow.log_metric('val_align_loss', val_losses['align'], step=epoch)
            mlflow.log_metric('val_temporal_loss', val_losses['temporal'], step=epoch)
            mlflow.log_metric('val_orthogonal_loss', val_losses['orthogonal'], step=epoch)

        elif method == "janickova":
            
            train_losses = train_epoch_janickova(
                model=model,
                loader=train_dl,
                optimizer=optimizer,
                timepoints=timepoints,
                device=device,          
                align_labels=ALIGN_LABELS,
                scaler=scaler,
                epoch=epoch, 
                accumulation_steps=ACCUMULATION_STEPS,
                lr_scheduler=lr_scheduler
            )

            val_losses, val_metrics = eval_epoch_janickova(
                model=model,
                loader=val_dl,
                timepoints=timepoints,
                device=device,
                align_labels=ALIGN_LABELS,
                epoch=epoch,
                accumulation_steps=ACCUMULATION_STEPS,
            )

            mlflow.log_metric('train_total_loss', train_losses['total'], step=epoch)
            mlflow.log_metric('train_align_loss', train_losses['align'], step=epoch)
            mlflow.log_metric('train_temporal_loss', train_losses['temporal'], step=epoch)

            mlflow.log_metric('val_total_loss', val_losses['total'], step=epoch)            
            mlflow.log_metric('val_align_loss', val_losses['align'], step=epoch)  
            mlflow.log_metric('val_temporal_loss', val_losses['temporal'], step=epoch)  

        if epoch == 1:
            torch.save(model.state_dict(), 'model_best_loss.pt') 
            torch.save(model.state_dict(), 'model_best_metric.pt') 
            mlflow.log_artifact('model_best_loss.pt')               
            mlflow.log_artifact('model_best_metric.pt')

        if val_losses['total'] <= best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), 'model_best_loss.pt')
            mlflow.log_artifact('model_best_loss.pt')               
        
        if val_metrics['val_bacc'] >= best_val_metric:
            best_val_metric = val_metrics['val_bacc']
            torch.save(model.state_dict(), 'model_best_metric.pt')
            mlflow.log_artifact('model_best_metric.pt')               

        torch.save(model.state_dict(), 'model_latest_epoch.pt')
        mlflow.log_artifact('model_latest_epoch.pt') 

    mlflow.log_param('best_val_loss', best_val_loss)  
    mlflow.log_param('best_val_bacc', best_val_metric)

    # Test evaluation

    for checkpoint in ['model_best_loss.pt', 'model_best_metric.pt', 'model_latest_epoch.pt']:
        checkpoint_name = checkpoint.replace('.pt','')        

        model.load_state_dict(torch.load(checkpoint))        

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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.numpy())
        X_val = scaler.transform(X_val.numpy())
        X_test = scaler.transform(X_test.numpy())    

        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test) 

        clf = SVC(probability=True)
        # clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train.numpy())

        train_preds = clf.predict(X_train)
        train_probs = clf.predict_proba(X_train)[:,1]
        test_preds = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)[:,1]

        train_bacc = balanced_accuracy_score(y_train.numpy(), train_preds)
        train_mcc = matthews_corrcoef(y_train.numpy(), train_preds)
        train_cm = confusion_matrix(y_train.numpy(), train_preds)   
        train_sensitivity = train_cm[1,1] / (train_cm[1,0] + train_cm[1,1])
        train_specificity = train_cm[0,0] / (train_cm[0,0] + train_cm[0,1])
        train_roc_auc = roc_auc_score(y_train.numpy(), train_probs)
        mlflow.log_param(f'train_bacc_{checkpoint_name}', train_bacc)
        mlflow.log_param(f'train_mcc_{checkpoint_name}', train_mcc)
        mlflow.log_param(f'train_roc_auc_{checkpoint_name}', train_roc_auc)
        mlflow.log_param(f'train_sensitivity_{checkpoint_name}', train_sensitivity)
        mlflow.log_param(f'train_specificity_{checkpoint_name}', train_specificity)
        test_bacc = balanced_accuracy_score(y_test.numpy(), test_preds)
        test_mcc = matthews_corrcoef(y_test.numpy(), test_preds)
        test_cm = confusion_matrix(y_test.numpy(), test_preds)   
        test_sensitivity = test_cm[1,1] / (test_cm[1,0] + test_cm[1,1])
        test_specificity = test_cm[0,0] / (test_cm[0,0] + test_cm[0,1])
        test_roc_auc = roc_auc_score(y_test.numpy(), test_probs)
        mlflow.log_param(f'test_bacc_{checkpoint_name}', test_bacc)
        mlflow.log_param(f'test_mcc_{checkpoint_name}', test_mcc)
        mlflow.log_param(f'test_roc_auc_{checkpoint_name}', test_roc_auc)    
        mlflow.log_param(f'test_sensitivity_{checkpoint_name}', test_sensitivity)
        mlflow.log_param(f'test_specificity_{checkpoint_name}', test_specificity)
    
    os.remove('model_best_metric.pt')
    os.remove('model_best_loss.pt')  
    os.remove('model_latest_epoch.pt')

if __name__ == '__main__':   
    
    for method in ["kiechle", "kaczmarek", "janickova"]:
        for timepoints in [4]:
            for fold in range(5):

                mlflow.set_experiment("self-supervised-pretraining")

                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():
                    main(method, timepoints, fold)
