import mlflow
import numpy as np
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data.Dataset import ISPY2
from models.DINOv3 import DINOv3
from utils.utils import log_all_python_files, set_deterministic

def main(method, timepoints, fold):   

    # set_deterministic()  
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)      

      
    train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_2D=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)    

    val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_2D=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_2D=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)    

    model = DINOv3().to(device)
    model.eval()         
    

    # train data
    train_latents = []
    train_labels = []

    with torch.no_grad():
        for batch_data in tqdm(train_dl):

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            train_latents.append(latents.reshape(b, -1).cpu().numpy())            
            train_labels.extend(labels.cpu().numpy().tolist())            
    
    train_latents = np.concatenate(train_latents, axis=0)
    train_labels = np.array(train_labels)

    # val data         
    val_latents = []
    val_labels = []

    with torch.no_grad():
        for batch_data in tqdm(val_dl):   

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            val_latents.append(latents.reshape(b, -1).cpu().numpy())            
            val_labels.extend(labels.cpu().numpy().tolist())            
    
    val_latents = np.concatenate(val_latents, axis=0)
    val_labels = np.array(val_labels)
    
    # test data
    test_latents = []
    test_labels = []

    with torch.no_grad():
        for batch_data in tqdm(test_dl):      

            images = batch_data[0][:, :timepoints, :, : ,:].float().to(device)
            labels = batch_data[1].long().to(device)

            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            latents = model(images)
            latents = latents.detach().view(b, t, -1)
            
            test_latents.append(latents.reshape(b, -1).cpu().numpy())            
            test_labels.extend(labels.cpu().numpy().tolist())            
    
    test_latents = np.concatenate(test_latents, axis=0)
    test_labels = np.array(test_labels)

    # svm classifier

    X_train, y_train = train_latents, train_labels
    X_val, y_val = val_latents, val_labels
    X_test, y_test = test_latents, test_labels

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

    mlflow.log_metric('test_bal_acc', bal_acc)
    mlflow.log_metric('test_mcc', mcc)
    mlflow.log_metric('test_roc_auc', auroc)
    mlflow.log_metric('test_sensitivity', sensitivity)
    mlflow.log_metric('test_specificity', specificity)


if __name__ == '__main__':     

    set_deterministic()

    for method in ['DINOv3']:
        for timepoints in [4]:            
            for fold in range(5):
        
                mlflow.set_tracking_uri("file:./mlruns")
                mlflow.set_experiment("miccai_2026")
                
                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():
                    main(method, timepoints, fold)
