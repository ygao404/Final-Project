import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

class TCGADataset(Dataset):
    def __init__(self, features, stage_labels, domain_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.stage_labels = torch.tensor(stage_labels, dtype=torch.float32).unsqueeze(1)
        self.domain_labels = torch.tensor(domain_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.stage_labels[idx], self.domain_labels[idx]

def load_and_prepare_data(config):
    data_dir = config['data']['data_dir']
    k_features = config['data']['k_features']
    test_size = config['data']['test_size']
    cal_size = config['data']['cal_size']
    random_state = config['data']['random_state']
    
    ordered_files = glob.glob(os.path.join(data_dir, '*_ordered.csv'))
    if not ordered_files:
        raise FileNotFoundError(f"No *_ordered.csv files found in {data_dir}.")
    
    X_list, y_raw_list = [], []
    expected_rows = 4367

    print("="*50)
    print(" 1. Data Validation and Loading")
    print("="*50)
    for file in ordered_files:
        filename = os.path.basename(file)
        df = pd.read_csv(file, index_col=0, low_memory=False)
        if df.shape[0] == expected_rows:
            print(f"[Pass] {filename}")
        else:
            continue
        X_list.append(df.iloc[1:].T.apply(pd.to_numeric, errors='coerce'))
        y_raw_list.append(df.iloc[0])

    X_all = pd.concat(X_list, axis=0).values
    y_raw_all = pd.concat(y_raw_list, axis=0).values

    domains, stages, valid_indices = [], [], []
    for i, label in enumerate(y_raw_all):
        if pd.isna(label) or label == "Unknown" or "-" not in str(label):
            continue
        domain, stage = label.split('-', 1)
        if domain == 'READ':
            domain = 'COAD'
        domains.append(domain)
        stages.append(stage)
        valid_indices.append(i)

    domain_counts = pd.Series(domains).value_counts()
    filtered_domains = [d if domain_counts.get(d, 0) >= 50 else 'Other' for d in domains]
    X_valid = X_all[valid_indices]
    print(f"\n[Data Cleaning] Valid samples: {len(X_valid)}")

    domain_encoder = LabelEncoder()
    y_domain_encoded = domain_encoder.fit_transform(filtered_domains)
    stage_encoder = LabelEncoder()
    y_stage_encoded = stage_encoder.fit_transform(stages)

    n_domains = len(domain_encoder.classes_)
    domain_onehot = np.eye(n_domains)[y_domain_encoded]

    X_tmp, X_test, y_stage_tmp, y_stage_test, y_domain_tmp, y_domain_test, doh_tmp, doh_test = train_test_split(
        X_valid, y_stage_encoded, y_domain_encoded, domain_onehot,
        test_size=test_size, random_state=random_state, stratify=y_domain_encoded
    )
    X_train, X_cal, y_stage_train, y_stage_cal, y_domain_train, y_domain_cal, doh_train, doh_cal = train_test_split(
        X_tmp, y_stage_tmp, y_domain_tmp, doh_tmp,
        test_size=cal_size, random_state=random_state, stratify=y_domain_tmp 
    )
    print(f"[Split] Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")

    print("\n>>> Evaluating features with Random Forest (may take tens of seconds)...")
    rf_estimators = config['data']['rf_estimators']
    rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_stage_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k_features]
    
    X_train_sel = X_train[:, indices]
    X_cal_sel   = X_cal[:, indices]
    X_test_sel  = X_test[:, indices]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_cal_s   = scaler.transform(X_cal_sel)
    X_test_s  = scaler.transform(X_test_sel)

    X_train = np.hstack([X_train_s, doh_train])
    X_cal   = np.hstack([X_cal_s,   doh_cal])
    X_test  = np.hstack([X_test_s,  doh_test])
    print(f"[Features] {k_features} genes + {n_domains} domain = {X_train.shape[1]} total")

    num_early = np.sum(y_stage_train == 0)
    num_late  = np.sum(y_stage_train == 1)
    pos_weight_val = (num_early / num_late) * config['data']['pos_weight_multiplier']
    print(f"[Weighting] Early={num_early}, Late={num_late}, pos_weight={pos_weight_val:.2f}")

    raw_domain_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_domain_train), y=y_domain_train)
    smoothed_domain_weights = np.sqrt(raw_domain_weights)

    train_dataset = TCGADataset(X_train, y_stage_train, y_domain_train)
    cal_dataset   = TCGADataset(X_cal,   y_stage_cal,   y_domain_cal)
    test_dataset  = TCGADataset(X_test,  y_stage_test,  y_domain_test)

    actual_input_dim = X_train.shape[1]
    return (train_dataset, cal_dataset, test_dataset,
            actual_input_dim, len(domain_encoder.classes_),
            stage_encoder.classes_, domain_encoder.classes_,
            pos_weight_val, smoothed_domain_weights,
            y_stage_cal, y_stage_test, y_domain_test)