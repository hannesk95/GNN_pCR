import argparse
import os
import itertools
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (e.g. 0 or 1)")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for cross-validation")
    parser.add_argument("--skip_loss", type=str, default=None, help="Specify which loss to skip: alignment_loss, temporal_loss, orthogonality_loss, or None")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import mlflow
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import GradScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from data.Dataset import ISPY2

from models.self_supervised.Janickova import ResNet18EncoderJanickova, ResNet18EncoderJanickova_new
from models.self_supervised.Kaczmarek import ResNet18EncoderKaczmarek
from models.self_supervised.Kiechle import ResNet18EncoderKiechle

from utils.pretraining_engine import (
    train_epoch_janickova,
    train_epoch_kaczmarek,
    train_epoch_kiechle,
    eval_epoch_janickova,
    eval_epoch_kaczmarek,
    eval_epoch_kiechle,
    inference_janickova,
    inference_kaczmarek,
    inference_kiechle,    
)
from utils.utils import log_all_python_files, seed_worker, set_deterministic, plot_umap, linear_probe


BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 100
ALIGN_LABELS = [1.0]

def main(method, timepoints, fold, skip_loss, feature_sim, temperature, use_gnn):
    
    set_deterministic()   
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)   
    mlflow.log_param('skip_loss', skip_loss)
    mlflow.log_param('feature_sim', feature_sim)
    mlflow.log_param('temperature', temperature)
    mlflow.log_param('use_gnn', use_gnn)

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

    g = torch.Generator()
    g.manual_seed(42)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=sampler, persistent_workers=True, worker_init_fn=seed_worker, generator=g)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, worker_init_fn=seed_worker, generator=g)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    if method == "kaczmarek":
        model = ResNet18EncoderKaczmarek(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

        perms = list(itertools.permutations(range(timepoints), timepoints))
        perms = [''.join([str(item) for item in p]) for p in perms]
        enc = OneHotEncoder()
        enc.fit(np.array(perms).reshape(-1, 1))

    elif method == "janickova":
        # model = ResNet18EncoderJanickova().to(device)
        model = ResNet18EncoderJanickova_new().to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif method == "kiechle":
        model = ResNet18EncoderKiechle(timepoints=timepoints, use_gnn=use_gnn).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)        
    
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, nesterov=True, momentum=0.99, weight_decay=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = np.inf
    best_val_metric = -np.inf

    for epoch in tqdm(range(1,EPOCHS+1)):

        if method == "kaczmarek":
            train_losses, train_z, train_y = train_epoch_kaczmarek(
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

            val_losses, val_z, val_y = eval_epoch_kaczmarek(
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
            
            train_losses, train_z, train_y = train_epoch_kiechle(
                    model=model,                    
                    loader=train_dl,
                    optimizer=optimizer,
                    timepoints=timepoints,
                    device=device,                   
                    scaler=scaler,                
                    epoch=epoch, 
                    accumulation_steps=ACCUMULATION_STEPS,
                    lr_scheduler=lr_scheduler,
                    skip_loss=skip_loss,
                    feature_sim=feature_sim,
                    temperature=temperature
            )

            val_losses, val_z, val_y = eval_epoch_kiechle(
                model=model,                   
                loader=val_dl,
                timepoints=timepoints,
                device=device,           
                epoch=epoch,
                accumulation_steps=ACCUMULATION_STEPS,
                skip_loss=skip_loss,
                feature_sim=feature_sim,
                temperature=temperature
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
            
            train_losses, train_z, train_y = train_epoch_janickova(
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

            val_losses, val_z, val_y = eval_epoch_janickova(
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

        plot_umap(z=train_z, y=train_y, epoch=epoch, split="train", fold=fold)
        plot_umap(z=val_z, y=val_y, epoch=epoch, split="val", fold=fold)
        linear_probe_auc, linear_probe_bacc, linear_probe_mcc = linear_probe(z_train=train_z, y_train=train_y, z_val=val_z, y_val=val_y)

        mlflow.log_metric('linear_probe_auc', linear_probe_auc, step=epoch)
        mlflow.log_metric('linear_probe_bacc', linear_probe_bacc, step=epoch)
        mlflow.log_metric('linear_probe_mcc', linear_probe_mcc, step=epoch)
        
        if epoch == 1:
            best_val_loss = val_losses['total']
            best_val_metric = linear_probe_auc
            torch.save(model.state_dict(), f'{method}_fold{fold}_best_loss.pt') 
            torch.save(model.state_dict(), f'{method}_fold{fold}_best_metric.pt') 
            mlflow.log_artifact(f'{method}_fold{fold}_best_loss.pt')               
            mlflow.log_artifact(f'{method}_fold{fold}_best_metric.pt')

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), f'{method}_fold{fold}_best_loss.pt')
            mlflow.log_artifact(f'{method}_fold{fold}_best_loss.pt')               
        
        if linear_probe_auc > best_val_metric:
            best_val_metric = linear_probe_auc
            torch.save(model.state_dict(), f'{method}_fold{fold}_best_metric.pt')
            mlflow.log_artifact(f'{method}_fold{fold}_best_metric.pt')               

        torch.save(model.state_dict(), f'{method}_fold{fold}_latest_epoch.pt')
        mlflow.log_artifact(f'{method}_fold{fold}_latest_epoch.pt') 

    mlflow.log_param('best_val_loss', best_val_loss)  
    mlflow.log_param('best_val_metric', best_val_metric)

    # Test evaluation
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # for checkpoint in [f'{method}_best_loss.pt', f'{method}_best_metric.pt', f'{method}_latest_epoch.pt']:
    for checkpoint in [f'{method}_fold{fold}_best_loss.pt', f'{method}_fold{fold}_best_metric.pt', f'{method}_fold{fold}_latest_epoch.pt']:
        checkpoint_name = checkpoint.replace('.pt','').replace(f'{method}_fold{fold}_', '')        

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

        for fold_sklearn, (train_idx, val_idx) in enumerate(
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

            print(f"Outer fold {fold_sklearn} | Balanced Acc: {score:.3f}")

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

        mlflow.log_metric(f'test_balanced_accuracy_{checkpoint_name}', bal_acc)
        mlflow.log_metric(f'test_mcc_{checkpoint_name}', mcc)
        mlflow.log_metric(f'test_auroc_{checkpoint_name}', auroc)
        mlflow.log_metric(f'test_sensitivity_{checkpoint_name}', sensitivity)
        mlflow.log_metric(f'test_specificity_{checkpoint_name}', specificity)
    
    os.remove(f'{method}_fold{fold}_best_metric.pt')
    os.remove(f'{method}_fold{fold}_best_loss.pt')  
    os.remove(f'{method}_fold{fold}_latest_epoch.pt')

if __name__ == '__main__':

    # If no specific fold is provided, run all folds and all experiments (this will take a long time!)
    if args.fold is None:
        # train and evaluate our method and perform loss function ablations        
        for use_gnn in [True]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None, "alignment_loss", "temporal_loss", "orthogonality_loss"]:
                        for method in ["kiechle"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(5):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:./mlruns")                    
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)
        
        # train and evaluate our method without GNN but all losses
        for use_gnn in [False]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None]:
                        for method in ["kiechle"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(5):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:./mlruns")                    
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)
        
        # train and evaluate SSL comparison methods
        for use_gnn in [None]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None]:
                        for method in ["kaczmarek", "janickova"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(5):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:./mlruns")                    
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)
    
    # If a specific fold is provided, only run that fold (useful for parallelization)
    else:
        # train and evaluate our method and perform loss function ablations        
        for use_gnn in [True]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None, "alignment_loss", "temporal_loss", "orthogonality_loss"]:
                        for method in ["kiechle"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(args.fold, args.fold+1):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:/dss/dssmcmlfs01/pn39hu/pn39hu-dss-0000/hannes/GNN_pCR/mlruns")                    
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)
        
        # train and evaluate our method without GNN but all losses
        for use_gnn in [False]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None]:
                        for method in ["kiechle"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(args.fold, args.fold+1):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:/dss/dssmcmlfs01/pn39hu/pn39hu-dss-0000/hannes/GNN_pCR/mlruns")                    
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)
        
        # train and evaluate SSL comparison methods
        for use_gnn in [None]:
            for feature_sim in [None]:
                for temperature in [None]:
                    for skip_loss in [None]:
                        for method in ["kaczmarek", "janickova"]: # ["kiechle", "kaczmarek", "janickova"]:
                            for timepoints in [4]:
                                for fold in range(args.fold, args.fold+1):
                                    
                                    args.skip_loss = skip_loss

                                    mlflow.set_tracking_uri("file:/dss/dssmcmlfs01/pn39hu/pn39hu-dss-0000/hannes/GNN_pCR/mlruns")                   
                                    mlflow.set_experiment(f"miccai_2026_auc")

                                    mlflow.end_run()  # end previous run if any
                                    with mlflow.start_run(run_name=f"{method}_fold_{fold}"):
                                        main(method, timepoints, fold, args.skip_loss, feature_sim, temperature, use_gnn)

