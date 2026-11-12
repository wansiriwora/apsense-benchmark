
import pandas as pd
import argparse
       
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from sklearn.utils.class_weight import compute_class_weight
def get_class_weights(_labels, n_class = 2, log_message=lambda msg: None):
    if isinstance(_labels, torch.Tensor):
        _labels = _labels.cpu().numpy()
    _labels = _labels.astype(int)
    unique, counts = np.unique(_labels, return_counts=True)
    freq = dict((zip(unique, counts)))
    n_samples = sum(list(freq.values()))
    print('len of labels:', n_samples)
    log_message('len of labels: {}'.format(n_samples))
    print(freq)
    
    print('label counts:', unique, counts)
    log_message('label counts: {} {}'.format(unique, counts))
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(_labels), y=_labels)
    print(type(class_weights))
    # ind = np.where(class_weights < 1)
    # class_weights[ind] = 1
    print('class_weights:', class_weights)
    log_message('class_weights: {}'.format(class_weights))
    return class_weights

import numpy as np
import torch
import pickle
def load_data(test_subjects, path):
    X_list, Y_list = [], []

    for idx in test_subjects:
        # print(idx)
        path_feature = f'{path}/s_{idx}.pkl'
        # print(f"📂 Loading {path_feature}...")
        if not os.path.exists(path_feature):
            print(f"⚠️ Warning: {path_feature} not found! Skipping...")
            continue
        
        with open(path_feature, 'rb') as f:
            feature = pickle.load(f)
        # Labels: HB uses 0/1, MESA uses 'Normal'/'Apnea'/'Hypopnea'
        label_map = {'Apnea': 1, 'Hypopnea': 1, 'Normal': 0, 0: 0, 1: 1}
        feat_cols = ['pwa', 'rr', 'diff_pwa', 'diff_rr', 'systole', 'diastole', 'area']
        y_mapped = feature['label'].map(label_map)
        feature = feature[y_mapped.notna()].reset_index(drop=True)
        y_mapped = y_mapped.dropna().reset_index(drop=True)
        if len(feature) == 0:
            print(f"⚠️ Warning: {path_feature} has no labelled windows! Skipping...")
            continue
        # -> (N, 7, winsize)
        x_array = np.stack([np.stack(feature[c].values) for c in feat_cols], axis=1)
        y_array = y_mapped.values.astype(np.int64)
        
        X_list.append(x_array)
        Y_list.append(y_array)
    # print(X_list)
    # print(Y_list)
    X_array = np.vstack(X_list)
    Y_array = np.hstack(Y_list)
    print(f"✅ X_array.shape: {X_array.shape}")
    print(f"✅ Y_array.shape: {Y_array.shape}")
    return torch.tensor(X_array, dtype=torch.float32), torch.tensor(Y_array, dtype=torch.float32)
    

def scaler_all_channel(train_x=None, test_x=None, scalers=None, ch=-1):
    prescaled = scalers is not None

    if (train_x is None and not prescaled) or (train_x is not None and prescaled):
        raise ValueError("Only train_x or scalers is required.")

    if test_x is None:
        test_x = train_x

    scalers = scalers if scalers else []
    scaled_X = []
    num_ch = train_x.shape[1] if train_x is not None else test_x.shape[1]

    if ch == -1:  # SCALER ALL CHANNEL
        for i in range(num_ch):
            if prescaled:
                scaler = scalers[i]
            else:
                scaler = StandardScaler()
                scaler.fit(train_x[:, i, :])

            scaled_X.append(scaler.transform(test_x[:, i, :]))
            scalers.append(scaler)

    else:
        if prescaled:
            scaler = scalers[i]
        else:
            scaler.fit(train_x[:, ch, :])

        scaled_X.append(scaler.transform(test_x[:, ch, :]))

    scaled_X = np.array(scaled_X)
    scaled_X = np.swapaxes(scaled_X, 0, 1)
    scaled_X = np.swapaxes(scaled_X, 1, 2)

    return scaled_X, scalers


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from models import MODELS
from models._dsep_block import _RDSepBlock

import torch.nn.functional as F


import collections.abc as container_abc
from collections import OrderedDict
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torch.optim as optim

# Apsense MESA
model = MODELS["DSepST15Net"](num_channels=7, winsize=60)
# summary(model, input_size=(1, 120, 5))

def subsample(X, y):
    rus = RandomUnderSampler(random_state=42)
    x_shape = X.shape
    X_res = X.reshape(x_shape[0], -1)
    X_res, y_res = rus.fit_resample(X_res, y)
    X_res = X_res.reshape(-1, *(x_shape[1:]))
    print(f"Remaining samples: {np.unique(y_res, return_counts=True)}")
    return X_res, y_res
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, f1_score, confusion_matrix, r2_score, roc_auc_score
)
import torch
import torch.nn as nn
import copy
import gc
import os

# %%
def run_model(output_dir, path_feature, meta_file, harmon_file, dataset):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = f'{output_dir}/model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    output_path = f'{output_dir}/test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_path = f'{output_dir}/model'
    output_path = f'{output_dir}/test'
    log_file = f'{output_dir}/baseline.txt'
    if dataset == 'MESA':
        metadata = pd.read_csv(f'{meta_file}')
        harm = pd.read_csv(f'{harmon_file}')
        SKIP_SUBJECTS = [2934,2416]
        id_l = []
        ahi_l = []
        for i in range(len(metadata)):
            if metadata['overall5'][i] == 7.0 and metadata['mesaid'][i] not in SKIP_SUBJECTS:
                id_o = metadata['mesaid'][i]
                id_l.append(id_o)
                ahi_l.append(harm[harm['mesaid']==id_o]['nsrr_ahi_hp3r_aasm15'].values[0])
    elif dataset == 'HB':
        metadata = pd.read_csv(f'{meta_file}')
        skip = [700045, 700217, 700173, 700144, 700065, 700129, 700151, 700267, 700053, 700174]
        id_l = []
        ahi_l = []
        for i in range(len(metadata)):
            if metadata['studyqual'][i] >= 5 and metadata['nsrrid'][i] not in skip:
                id_o = metadata['nsrrid'][i]
                ahi = metadata['aphypi'][i]
                id_l.append(id_o)
                ahi_l.append(ahi)
    meta = pd.DataFrame({'idx': id_l, 'ahi': ahi_l})
    meta['severity'] = meta['ahi'].apply(lambda x: 1 if x > 15 else 0)
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model Parameters
    batch_size = 256
    num_epochs = 1000
    # lr = 0.001
    lr = 3e-4
    early_stopping_patience = 30
    WEIGHTED_CLASS = True
    LR_SCHEDULER_PATIENCE = 5
    thres = 0.0
    decay = 1e-4

    confusion_matrix_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def log_message(message, log_file=log_file):
        with open(log_file, "a") as file:
            file.write(message + "\n")

    # Outer Fold loop
    for outer_fold, (train_val_idx, test_idx) in enumerate(skf_outer.split(meta['idx'], meta['severity'])):
        if outer_fold >= 0:       
            best_fold = None
            best_model = None
            best_model_state_dict = None
            optimizer = None
            train_loss_lists = []
            train_acc_lists = []
            val_loss_lists = []
            val_acc_lists = []
            val_f1_lists = []
            
            print(f'Outer Fold {outer_fold+1} (Test Set)')
            log_message(f'Outer Fold {outer_fold+1} (Test Set)')
            
            inner_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            test_subjects = meta['idx'].iloc[test_idx]
            train_val_subjects = meta['idx'].iloc[train_val_idx]
            train_val_severity = meta['severity'].iloc[train_val_idx]   # ← เพิ่มบรรทัดนี้
            
            for inner_fold, (train_idx, val_idx) in enumerate(inner_kfold.split(train_val_subjects, train_val_severity)):
                if inner_fold >= 0:
                    print(f'Inner Fold {inner_fold+1} (Validation Set)')
                    log_message(f'Inner Fold {inner_fold+1} (Validation Set)')
                    print("train subjects:", train_val_subjects.iloc[train_idx].unique())
                    log_message(f'train subjects: {train_val_subjects.iloc[train_idx].unique()}')
                    print("  val subjects:", train_val_subjects.iloc[val_idx].unique())
                    log_message(f'val subjects:{train_val_subjects.iloc[val_idx].unique()}')    
                    
                    train_subjects = train_val_subjects.iloc[train_idx].unique()
                    val_subjects = train_val_subjects.iloc[val_idx].unique()
                    
                    X_train_b, Y_train_b = load_data(train_subjects, path_feature)
                    X_val, Y_val = load_data(val_subjects, path_feature)
                    
                    X_train_s, scalers = scaler_all_channel(X_train_b, X_train_b)
                    X_val_s, _ = scaler_all_channel(X_train_b, X_val)    
                    
                    # Convert to Torch
                    X_train, Y_train = torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(Y_train_b, dtype=torch.long)
                    X_val, Y_val = torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long)

                    # Model Initialization
                    model = MODELS["DSepST15Net"](num_channels=7, winsize=60)
                    model.to(device)
                    
                    # Calculate class weights
                    class_weights_train = get_class_weights(Y_train, log_message=log_message)
                    print(f'Class weights train : {class_weights_train}')
                    log_message(f'Class weights train : {class_weights_train}')
                    class_weights_train = torch.tensor(class_weights_train, dtype=torch.float32).to(device)
                    
                    class_weights_val = get_class_weights(Y_val, log_message=log_message)
                    print(f'Class weights val : {class_weights_val}')
                    log_message(f'Class weights val : {class_weights_val}')
                    class_weights_val = torch.tensor(class_weights_val, dtype=torch.float32).to(device)
                    
                    # Data loader
                    dataset = torch.utils.data.TensorDataset(
                        torch.tensor(X_train), Y_train)
                    train_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=True)

                    dataset = torch.utils.data.TensorDataset(
                        torch.tensor(X_val), Y_val)
                    val_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=True)
                    
                    start = time.time()
                    # Training loop
                    train_loss_list = []
                    train_acc_list = []
                    val_loss_list = []
                    val_acc_list = []
                    val_f1_list = []
                    
                    # Testing loop
                    fold_pred = []
                    fold_true = []
                    fold_prob = []
                    
                    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    # lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='min', verbose=True)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.5,
                        patience=10,
                        threshold=1e-3,
                        cooldown=3,
                        min_lr=1e-6,
                        verbose=True
                    )
                    best_loss = np.inf
                    early_stopping_counter = 0
                    start_time = time.time()
                    ce_criterion = nn.BCEWithLogitsLoss()
                    # train loop
                    for epoch in range(1, num_epochs+1):
                        model.train()
                        train_loss = 0.0
                        train_acc = 0.0
                        for i, data in enumerate(train_loader):
                            x_input, apneic_groundtruth = data
                            x_input, apneic_groundtruth = x_input.to(device), apneic_groundtruth.to(device)
                            x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
                            apneic_stage = model(x_input).reshape(-1)
                            apneic_groundtruth = apneic_groundtruth.float()
                            data_weight = torch.where(apneic_groundtruth == 0, class_weights_train[0], class_weights_train[1])
                            apneic_loss = ce_criterion(apneic_stage, apneic_groundtruth.reshape(-1))
                            optimizer.zero_grad()
                            apneic_loss.backward()
                            optimizer.step()
                            train_loss += apneic_loss.item() * x_input.size(0)
                        train_loss /= len(train_loader.dataset)
                        train_loss_list.append(train_loss)
                        
                        with torch.no_grad():
                            for i, data in enumerate(train_loader):
                                x_input, apneic_groundtruth = data
                                x_input, apneic_groundtruth = x_input.to(device), apneic_groundtruth.to(device)
                                x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
                                apneic_stage = model(x_input).reshape(-1)
                                
                                # Set if the result is less than threshold, it will be 0
                                threshold = torch.tensor([thres], dtype=torch.float32).to(device)
                                train_pred = (apneic_stage > threshold).float() * 1
                                train_acc += (train_pred == apneic_groundtruth).sum().item()
                        train_acc /= len(train_loader.dataset)
                        train_acc_list.append(train_acc)
                        
                        model.eval()
                        val_loss = 0.0
                        val_acc = 0.0
                        y_true_list = []
                        y_pred_list = []
                        with torch.no_grad():
                            for i, data in enumerate(val_loader):
                                x_input, apneic_groundtruth = data
                                x_input, apneic_groundtruth = x_input.to(device), apneic_groundtruth.to(device)
                                x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
                                apneic_stage = model(x_input).reshape(-1)
                                # Set if the result is less than threshold, it will be 0
                                threshold = torch.tensor([thres], dtype=torch.float32).to(device)
                                val_pred = (apneic_stage > threshold).float() * 1
                                y_true_list.extend(apneic_groundtruth.cpu().numpy())
                                y_pred_list.extend(val_pred.cpu().numpy())
                                val_acc += (val_pred == apneic_groundtruth).sum().item()
                                apneic_groundtruth = apneic_groundtruth.float()
                                data_weight = torch.where(apneic_groundtruth == 0, class_weights_val[0], class_weights_val[1])
                                apneic_loss = ce_criterion(apneic_stage, apneic_groundtruth.reshape(-1))
                                val_loss += apneic_loss.item() * x_input.size(0)
                        val_f1 = f1_score(y_true_list, y_pred_list)
                        val_loss /= len(val_loader.dataset)
                        val_loss_list.append(val_loss)
                        val_acc /= len(val_loader.dataset)
                        val_acc_list.append(val_acc)
                        val_f1_list.append(val_f1)
                        scheduler.step(val_loss)
                        # lr_scheduler.step(val_loss)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                            
                        if early_stopping_counter == early_stopping_patience:
                            print(f'Early stopping at epoch {epoch}')
                            log_message(f'Early stopping at epoch {epoch}')
                            break
                            
                        print(f'Epoch {epoch:02d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f}')
                    log_message(f'Training complete in: {time.time()-start_time} seconds')
                    train_loss_lists.append(train_loss_list)
                    train_acc_lists.append(train_acc_list)
                    val_loss_lists.append(val_loss_list)
                    val_acc_lists.append(val_acc_list)
                    val_f1_lists.append(val_f1_list)
                    optimizer_state_dict = copy.deepcopy(optimizer)
                    # model_state_dict = copy.deepcopy(model.state_dict())
                    # model_l = copy.deepcopy(model)
                    log_message(f'---fold {inner_fold}---average accuracy {np.mean(val_acc_list)}')
                    torch.cuda.empty_cache()
                    
                    torch.save(best_model_state_dict, f'{model_path}/model_outer_{outer_fold}_fold_{inner_fold}.pth')
                    log_message(f'save model: {model_path}/model_outer_{outer_fold}_fold_{inner_fold}.pth')
                    log_message(f'Train time: {start-time.time()}')

                    # Test set
                    # Test set download data
                    test_subjects = meta['idx'].iloc[test_idx]
                    test_severity = meta['severity'].iloc[test_idx]
                    print(pd.Series(test_severity).value_counts())
                    test_sever = pd.Series(test_severity).value_counts()
                    # log_message(test_sever.to_string())
                    test_id = list(test_subjects.unique())
                    X_test, Y_test = load_data(test_id,path_feature)
                    X_test, _ = scaler_all_channel(X_train_b, X_test)
                    print(f'shape of X_test, shape of Y_test: {X_test.shape, Y_test.shape}')
                    log_message(f'shape of X_test, shape of Y_test: {X_test.shape, Y_test.shape}')
                    test_dataset = TensorDataset(torch.tensor(X_test), Y_test)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                    log_message(f'Test subjects: {test_id}')
                    log_message(f'Test: {pd.Series(test_severity).value_counts()}')
                    log_message(f'shape of X_test, shape of Y_test: {X_test.shape, Y_test.shape}')
                    model.load_state_dict(best_model_state_dict)
                    model.eval()
                    correct, total = 0.0, 0.0
                    with torch.no_grad():
                        for i, data in enumerate(test_loader):
                            x_input, apneic_groundtruth = data
                            x_input, apneic_groundtruth = x_input.to(device), apneic_groundtruth.to(device)
                            x_input, apneic_groundtruth = x_input.float(), apneic_groundtruth.long()
                            apneic_stage = model(x_input).reshape(-1)
                            if apneic_stage.cpu().numpy().shape == ():
                                fold_prob.append(1/(1+np.exp(-apneic_stage.cpu().numpy())))
                            else:
                                fold_prob.extend(1/(1+np.exp(-apneic_stage.cpu().numpy())))
                            threshold = torch.tensor([thres], dtype=torch.float32).to(device)
                            test_pred = (apneic_stage > threshold).float() * 1
                            fold_pred.extend(test_pred.cpu().numpy())
                            fold_true.extend(apneic_groundtruth.cpu().numpy())
                            total += apneic_groundtruth.size(0)
                            correct += (test_pred == apneic_groundtruth).sum().item()
                    acc = 100 * correct / total
                    print('Test Accuracy: ', acc)
                    log_message(f'Test Accuracy : {acc}')
                    print()
                    torch.save({
                        'outer fold': outer_fold,
                        'inner fold': inner_fold,
                        'optimizer': optimizer_state_dict,
                        'train_loss': train_loss_list,
                        'train_acc': train_acc_list,
                        'val_loss': val_loss_list,
                        'val_acc': val_acc_list,
                        'val_f1': val_f1_list,
                        'y_true': fold_true,
                        'y_pred': fold_pred,
                        'test_pred_prob': fold_prob,
                    }, f'{output_path}/output_outer_{outer_fold}_fold_{inner_fold}.pth')
                    print('-'*10)
                    log_message('-'*10)
                    gc.collect()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="batchsample method")
    p.add_argument("--output", required=True, help='output directory')
    p.add_argument("--feature", required=True, help='feature directory')
    p.add_argument("--meta", required=True, help='meta directory')
    p.add_argument("--harmon", required=False, help='harmonize only in MESA')
    p.add_argument("--dataset", required=True, help="'MESA' or 'HB'")
    args = p.parse_args()
    
    run_model(args.output, args.feature, args.meta, args.harmon, args.dataset)

# python imbalance.py \
#     --output '/home/work/Apnea/ApSense/benchmark/imbalance/HB' \
#     --feature '/home/work/Apnea/ApSense/benchmark/preprocess/HB/feature' \
#     --meta '/home/Data/heartbeat/datasets/heartbeat-baseline-dataset-0.4.0.csv' \
#     --dataset 'HB'