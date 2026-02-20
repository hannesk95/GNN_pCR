import mlflow
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.Dataset import ISPY2

from models.self_supervised.Janickova import ResNet18EncoderJanickova_new
from models.self_supervised.Kaczmarek import ResNet18EncoderKaczmarek
from models.self_supervised.Kiechle import ResNet18EncoderKiechle

from utils.pretraining_engine import (
    inference_janickova,
    inference_kaczmarek,
    inference_kiechle,
)
from utils.utils import log_all_python_files, set_deterministic


def main(method, timepoints, fold, checkpoint_path):
    
    set_deterministic()   
    log_all_python_files()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # log params
    mlflow.log_param('method', method)    
    mlflow.log_param('timepoints', timepoints)
    mlflow.log_param('fold', fold)   

    # datasets
    if "dist" in method:
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints, output_time_dists=True)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints, output_time_dists=True)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints, output_time_dists=True)
    
    else:
        train_dataset = ISPY2(split='train', fold=fold, timepoints=timepoints)
        val_dataset = ISPY2(split='val', fold=fold, timepoints=timepoints)
        test_dataset = ISPY2(split='test', fold=fold, timepoints=timepoints)    

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    if "kaczmarek" in method:
        model = ResNet18EncoderKaczmarek(timepoints=timepoints).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif "janickova" in method:
        model = ResNet18EncoderJanickova_new().to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)

    elif "kiechle" in method:
        model = ResNet18EncoderKiechle(timepoints=timepoints, use_gnn=True, method=method).to(device)
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param('model_num_params', model_num_params)        
        

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))        

    if "kiechle" in method:
        embeddings, labels = inference_kiechle(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device,
                method=method)
    
    elif "kaczmarek" in method:
        embeddings, labels = inference_kaczmarek(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device)
    
    elif "janickova" in method:
        embeddings, labels = inference_janickova(
                model=model,                
                loader=[train_dl, val_dl, test_dl],
                device=device,
                method=method)          

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

    mlflow.log_metric('test_balanced_accuracy', bal_acc)
    mlflow.log_metric('test_mcc', mcc)
    mlflow.log_metric('test_auroc', auroc)
    mlflow.log_metric('test_sensitivity', sensitivity)
    mlflow.log_metric('test_specificity', specificity)
        

if __name__ == '__main__':  

    janickova_dict = {
        "fold_0_best_metric": "",
        "fold_1_best_metric": "",
        "fold_2_best_metric": "",
        "fold_3_best_metric": "",
        "fold_4_best_metric": "",
    }

    janickova_dist_dict = {
        "fold_0_best_metric": "",
        "fold_1_best_metric": "",
        "fold_2_best_metric": "",
        "fold_3_best_metric": "",
        "fold_4_best_metric": "",
    }
    
    kiechle_dict = {
        "fold_0_best_metric": "",
        "fold_1_best_metric": "",
        "fold_2_best_metric": "",
        "fold_3_best_metric": "",
        "fold_4_best_metric": "",
    }
    
    kiechle_dist_dict = {
        "fold_0_best_metric": "",
        "fold_1_best_metric": "",
        "fold_2_best_metric": "",
        "fold_3_best_metric": "",
        "fold_4_best_metric": "",
    }

    checkpoints_dict = {
        "janickova": janickova_dict,
        "janickova_dist": janickova_dist_dict,
        "kiechle": kiechle_dict,
        "kiechle_dist": kiechle_dist_dict,
    }    

    for method in checkpoints_dict.keys():        
        for timepoints in [4]:
            for fold in range(5):
                
                checkpoint_path = checkpoints_dict[method][f"fold_{fold}_best_metric"]
                
                mlflow.set_tracking_uri("file:./mlruns")
                mlflow.set_experiment("self-supervised-pretraining_evaluation")
                
                mlflow.end_run()  # end previous run if any
                with mlflow.start_run():          
                    main(method, timepoints, fold, checkpoint_path)
