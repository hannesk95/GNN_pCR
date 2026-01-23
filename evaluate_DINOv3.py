import torch
from tqdm import tqdm
import numpy as np
import os
from data.Dataloader import get_train_dataloaders, get_val_dataloaders, get_test_dataloaders
import argparse
import yaml
from utils.pretraining_engine import train_epoch, eval_epoch
from models.builder import build_model
from models.GNN import TemporalGNN
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
from models.Janickova import ResNet18Encoder
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

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config, fold, feature_aggregation, timepoints):   

    feature_aggregation = feature_aggregation  # 'mean' or 'concat'
    mlflow.log_param("feature_aggregation", feature_aggregation)
    mlflow.log_param("timepoints", timepoints)
    # timepoints=['T0', 'T1', 'T2', 'T3']
    # timepoints = 3
    device = config.device      

    # train_dl = get_train_dataloaders(data_path=config.data_path, 
    #                                  batch_size=1, 
    #                                  timepoints=timepoints, 
    #                                  system=config.system,
    #                                  fold=fold)      
    train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_2D=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # val_dl = get_val_dataloaders(data_path=config.data_path, 
    #                              val_batch_size=1, 
    #                              timepoints=timepoints,
    #                              system=config.system,
    #                              fold=fold) 
    val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_2D=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # test_dl = get_test_dataloaders(data_path=config.data_path, 
    #                                test_batch_size=1, 
    #                                timepoints=timepoints,
    #                                system=config.system,
    #                                fold=fold)
    test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_2D=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # model set-up
    # model = build_model(mtan_masking=config.mtan_masking, filters=config.filters, device=config.device)
    model = DINOv3().to(config.device)
    # pretrained_model_path = artifact_path
    # model.load_state_dict(torch.load(pretrained_model_path, map_location=config.device))
    model.eval()         
    

    # train data
    train_latents = []
    train_labels = []
    with torch.no_grad():
        for batch_data in tqdm(train_dl):        

            if not config.use_gnn:       

                # latent_list = []
                # for i in range(len(timepoints)):
                #     # img = batch_data[timepoints[i]][0].float().to(device)
                #     img = batch_data[f'target_{timepoints[i]}'].float().to(device)
                #     _, latent = model(img)
                #     latent = nn.functional.normalize(latent)
                #     latent_list.append(latent.cpu().numpy())

                
                images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
                labels = batch_data[1].long().to(device)

                b, t, c, h, w = images.shape
                images = images.view(b * t, c, h, w)
                latents = model(images)
                latents = latents.view(b, t, -1)
                latents = latents.clone().detach()

                if feature_aggregation == 'mean':
                    train_latents.append(np.mean(latents.cpu().numpy(), axis=1))
                elif feature_aggregation == 'concat':
                    # train_latents.append(np.concatenate(latents.cpu().numpy(), axis=2))
                    train_latents.append(latents.view(latents.size(0), -1).cpu().numpy())
                else:
                    raise ValueError('Invalid feature aggregation method')
                
                train_labels.extend(list(labels.cpu().numpy()))            
    
    train_latents = np.concatenate(train_latents, axis=0)
    train_labels = np.array(train_labels)

    # val data         
    val_latents = []
    val_labels = []
    with torch.no_grad():
        for batch_data in tqdm(val_dl):   

            if not config.use_gnn:            

                # latent_list = []
                # for i in range(len(timepoints)):
                #     # img = batch_data[timepoints[i]][0].float().to(device)
                #     img = batch_data[f'target_{timepoints[i]}'].float().to(device)
                #     _, latent = model(img)
                #     latent = nn.functional.normalize(latent)
                #     latent_list.append(latent.cpu().numpy())
                images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
                labels = batch_data[1].long().to(device)

                b, t, c, h, w = images.shape
                images = images.view(b * t, c, h, w)
                latents = model(images)
                latents = latents.view(b, t, -1)
                latents = latents.clone().detach()

                if feature_aggregation == 'mean':
                    val_latents.append(np.mean(latents.cpu().numpy(), axis=1))
                elif feature_aggregation == 'concat':
                    # val_latents.append(np.concatenate(latents.cpu().numpy(), axis=2))
                    val_latents.append(latents.view(latents.size(0), -1).cpu().numpy())
                else:
                    raise ValueError('Invalid feature aggregation method')
                
                val_labels.extend(list(labels.cpu().numpy()))
            
            elif config.use_gnn:

                imgs_list = []
                for i in range(len(timepoints)):
                    # img = batch_data[timepoints[i]][0].float().to(device)
                    img = batch_data[f'target_{timepoints[i]}'].float().to(device)
                    imgs_list.append(img)

                imgs = torch.cat(imgs_list, dim=0)  # (T, C, H, W)
                _, latent = model(imgs)
                latent = nn.functional.normalize(latent)

                batch_size = val_dl.batch_size
                graph_data = make_directed_complete_forward_graph(latent, batch_size=batch_size)
                graph_data = graph_data.to(device)
                out = model_gnn(graph_data.x, graph_data.edge_index)
                out = out.view(batch_size, -1, out.size(-1))

                if feature_aggregation == 'mean':
                    out_agg = torch.mean(out, dim=1)
                elif feature_aggregation == 'concat':
                    out_agg = out.flatten(start_dim=1)

                val_latents.append(out_agg.cpu().numpy())
                label = batch_data['pcr'].float().to(device)
                val_labels.append(int(label.cpu().numpy()[0]))
    
    val_latents = np.concatenate(val_latents, axis=0)
    val_labels = np.array(val_labels)

    # test data
    test_latents = []
    test_labels = []
    with torch.no_grad():
        for batch_data in tqdm(test_dl):      

            if not config.use_gnn:         

                # latent_list = []
                # for i in range(len(timepoints)):
                #     # img = batch_data[timepoints[i]][0].float().to(device)
                #     img = batch_data[f'target_{timepoints[i]}'].float().to(device)
                #     _, latent = model(img)
                #     latent = nn.functional.normalize(latent)
                #     latent_list.append(latent.cpu().numpy())
                images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
                labels = batch_data[1].long().to(device)

                b, t, c, h, w = images.shape
                images = images.view(b * t, c, h, w)
                latents = model(images)
                latents = latents.view(b, t, -1)
                latents = latents.clone().detach()

                if feature_aggregation == 'mean':
                    test_latents.append(np.mean(latents.cpu().numpy(), axis=1))
                elif feature_aggregation == 'concat':
                    # test_latents.append(np.concatenate(latents.cpu().numpy(), axis=2))
                    test_latents.append(latents.view(latents.size(0), -1).cpu().numpy())
                else:
                    raise ValueError('Invalid feature aggregation method')
                
                test_labels.extend(list(labels.cpu().numpy()))
            
            elif config.use_gnn:
                imgs_list = []
                for i in range(len(timepoints)):
                    # img = batch_data[timepoints[i]][0].float().to(device)
                    img = batch_data[f'target_{timepoints[i]}'].float().to(device)
                    imgs_list.append(img)

                imgs = torch.cat(imgs_list, dim=0)  # (T, C, H, W)
                _, latent = model(imgs)
                latent = nn.functional.normalize(latent)

                batch_size = test_dl.batch_size
                graph_data = make_directed_complete_forward_graph(latent, batch_size=batch_size)
                graph_data = graph_data.to(device)
                out = model_gnn(graph_data.x, graph_data.edge_index)
                out = out.view(batch_size, -1, out.size(-1))

                if feature_aggregation == 'mean':
                    out_agg = torch.mean(out, dim=1)
                elif feature_aggregation == 'concat':
                    out_agg = out.flatten(start_dim=1)

                test_latents.append(out_agg.cpu().numpy())
                label = batch_data['pcr'].float().to(device)
                test_labels.append(int(label.cpu().numpy()[0]))
    
    test_latents = np.concatenate(test_latents, axis=0)
    test_labels = np.array(test_labels)

    # logistic regression with 10 runs random seeds and balanced accuracy evaulation

    pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("clf", SVC(probability=True, random_state=42))
        ])
    
    param_grid = {
        # --- Core model type ---
        'clf__kernel': ['linear'],

        # --- Regularization strength ---
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],

        # --- PCA components ---
        "pca__n_components": [0.95, 0.99]
    }

    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    outer_results_bacc = list()
    outer_results_f1 = list()
    outer_results_mcc = list()
    outer_results_auroc = list()

    outer_true = list()
    outer_pred = list()

    X = np.concatenate([train_latents, val_latents], axis=0)
    y = np.concatenate([train_labels, val_labels], axis=0)

    for i, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):
        # split data
        # X_train, X_test = X[train_ix, :], X[test_ix, :]
        # y_train, y_test = y[train_ix], y[test_ix]

        X_train = X
        y_train = y
        X_test = test_latents
        y_test = test_labels
        
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # define the model
        # pipeline, param_grid = get_model_and_param_grid(model_type=model_type)
        
        # define search
        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv_inner, refit=True)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)

        outer_true.extend(y_test)
        outer_pred.extend(yhat)
        
        # evaluate the model        
        bacc = balanced_accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average='weighted')
        mcc = matthews_corrcoef(y_test, yhat)
        auroc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        mlflow.log_metric(f"bacc_fold_{i}", bacc)
        mlflow.log_metric(f"f1_fold_{i}", f1)
        mlflow.log_metric(f"mcc_fold_{i}", mcc)
        mlflow.log_metric(f"auroc_fold_{i}", auroc)

        outer_results_bacc.append(bacc)
        outer_results_f1.append(f1)
        outer_results_mcc.append(mcc)
        outer_results_auroc.append(auroc)
        
        # report progress
        print('>bacc=%.3f, est=%.3f, cfg=%s' % (bacc, result.best_score_, result.best_params_))

    # summarize the estimated performance of the model
    print('Balanced Accuracy:   %.3f (%.3f)' % (mean(outer_results_bacc), std(outer_results_bacc)))
    print('F1 Score:            %.3f (%.3f)' % (mean(outer_results_f1), std(outer_results_f1)))
    print('MCC:                 %.3f (%.3f)' % (mean(outer_results_mcc), std(outer_results_mcc)))
    print('AUROC:               %.3f (%.3f)' % (mean(outer_results_auroc), std(outer_results_auroc)))

    mlflow.log_metric("bacc_mean", mean(outer_results_bacc))
    mlflow.log_metric("bacc_std", std(outer_results_bacc))
    mlflow.log_metric("f1_mean", mean(outer_results_f1))
    mlflow.log_metric("f1_std", std(outer_results_f1))
    mlflow.log_metric("mcc_mean", mean(outer_results_mcc))
    mlflow.log_metric("mcc_std", std(outer_results_mcc))
    mlflow.log_metric("auroc_mean", mean(outer_results_auroc))
    mlflow.log_metric("auroc_std", std(outer_results_auroc))

    
    # param_grid = {
    #     "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    #     "penalty": ['l2'],
    #     "solver": ["lbfgs"],  # must be compatible with penalty
    #     "max_iter": [500],
    #     "class_weight": [None, 'balanced']
    # }

    # accuracies = []
    # aucs = []

    # for seed in tqdm(range(10)):
    #     best_acc = -np.inf
    #     best_params = None

    #     # ----- GRID SEARCH ON VALIDATION SET -----
    #     for C in param_grid["C"]:
    #         for penalty in param_grid["penalty"]:
    #             for solver in param_grid["solver"]:
    #                 for max_iter in param_grid["max_iter"]:
    #                     for class_weight in param_grid["class_weight"]:
    #                         clf = LogisticRegression(
    #                             C=C,
    #                             penalty=penalty,
    #                             solver=solver,
    #                             max_iter=max_iter,
    #                             random_state=seed
    #                         )
    #                         clf.fit(train_latents, train_labels)
    #                         val_preds = clf.predict(val_latents)
    #                         val_acc = balanced_accuracy_score(val_labels, val_preds)                                

    #                         if val_acc > best_acc:
    #                             best_acc = val_acc
    #                             best_params = {
    #                                 "C": C,
    #                                 "penalty": penalty,
    #                                 "solver": solver,
    #                                 "max_iter": max_iter,
    #                                 "class_weight": class_weight
    #                             }

    #     # ----- TRAIN FINAL MODEL WITH BEST PARAMS -----
    #     clf = LogisticRegression(
    #         **best_params,
    #         random_state=seed
    #     )
    #     clf.fit(train_latents, train_labels)

    #     # ----- TEST PERFORMANCE -----
    #     preds = clf.predict(test_latents)
    #     scores = clf.predict_proba(test_latents)[:, 1]
    #     acc = balanced_accuracy_score(test_labels, preds)
    #     auc = roc_auc_score(test_labels, scores)
    #     accuracies.append(acc)
    #     aucs.append(auc)

    # mean_acc = np.mean(accuracies)
    # std_acc = np.std(accuracies)
    # mean_auc = np.mean(aucs)
    # std_auc = np.std(aucs)
    # mlflow.log_metric("mean_accuracy", mean_acc)
    # mlflow.log_metric("std_accuracy", std_acc)
    # mlflow.log_metric("mean_auc", mean_auc)
    # mlflow.log_metric("std_auc", std_auc)

    # print(f'Logistic Regression Accuracy over 10 runs: {mean_acc:.4f} +/- {std_acc:.4f}')       

    # scaled = StandardScaler().fit_transform(test_latents)

    # # UMAP to 2D
    # reducer = umap.UMAP(
    #     n_components=2,
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     metric="euclidean",
    #     random_state=42
    # )
    # emb_2d = reducer.fit_transform(scaled)

    # # Plot
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(
    #     emb_2d[:, 0],
    #     emb_2d[:, 1],
    #     c=test_labels,
    #     s=15,
    #     alpha=0.8
    # )
    # plt.colorbar(scatter, label="Label")
    # plt.xlabel("UMAP-1")
    # plt.ylabel("UMAP-2")
    # plt.title("UMAP Projection of Embeddings")
    # plt.tight_layout()
    # plt.savefig("umap_projection.png", dpi=300)          

    # mlflow.log_artifact("umap_projection.png")
    # os.remove("umap_projection.png")

if __name__ == '__main__':     

    set_deterministic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='/home/johannes/Data/SSD_2.0TB/GNN_pCR/configs/pretraining_sweep.yaml',
                        help='path to wadb sweep config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    config.use_gnn = False  # evaluate DINOv3 only

    for fold in range(5):
        for feature_aggregation in ['concat', 'mean']:
            for timepoints in [3, 4]:
                # print(f'Evaluating artifact: {artifact_name} with feature aggregation: {feature_aggregation}')

                # fold = int(artifact_name.split("_")[1])
                # model_checkpoint = artifact_name.split("_")[2]
                mlflow.set_experiment("evaluation_DINOv3")
                mlflow.start_run()
                mlflow.log_param("fold", fold)
                # mlflow.log_param("model_checkpoint", model_checkpoint)
            
                main(config, fold, feature_aggregation, timepoints)
                mlflow.end_run()
