import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, mean_absolute_error 
from sklearn.preprocessing import label_binarize 
import matplotlib.pyplot as plt 

from tcga_loss import BinaryFocalLoss
from tcga_model import DANN_MLP
from tcga_dataset import load_and_prepare_data

def masked_mse_loss(predictions, targets, mask):
    valid = mask.view(-1) > 0
    if not torch.any(valid):
        return predictions.new_zeros(())
    return F.mse_loss(predictions.view(-1)[valid], targets.view(-1)[valid])

def train_one(train_loader, input_dim, num_domains, num_genders, pos_weight_val,
              domain_class_weights, gender_class_weights, device, seed, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    epochs = config['train']['epochs']
    lr = float(config['train']['lr'])
    weight_decay = float(config['train']['weight_decay'])
    dropout_rate = float(config['model']['dropout_rate'])
    focal_gamma = float(config['loss']['focal_gamma'])
    domain_loss_weight = float(config['loss']['domain_loss_weight'])
    gender_loss_weight = float(config['loss'].get('gender_loss_weight', 0.0))
    age_loss_weight = float(config['loss'].get('age_loss_weight', 0.0))
    max_alpha = float(config['train']['max_alpha'])

    model = DANN_MLP(
        input_dim=input_dim,
        num_domains=num_domains,
        num_genders=num_genders,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion_stage = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight_tensor)
    
    domain_weight_tensor = torch.tensor(domain_class_weights, dtype=torch.float32).to(device)
    criterion_domain = nn.CrossEntropyLoss(weight=domain_weight_tensor)
    gender_weight_tensor = torch.tensor(gender_class_weights, dtype=torch.float32).to(device)
    criterion_gender = nn.CrossEntropyLoss(weight=gender_weight_tensor)
    
    total_batches = epochs * len(train_loader)
    current_batch = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y_stage, batch_y_domain, batch_y_gender, batch_y_age, batch_y_age_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y_stage = batch_y_stage.to(device)
            batch_y_domain = batch_y_domain.to(device)
            batch_y_gender = batch_y_gender.to(device)
            batch_y_age = batch_y_age.to(device)
            batch_y_age_mask = batch_y_age_mask.to(device)
            
            p = float(current_batch) / total_batches
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * max_alpha
            
            optimizer.zero_grad()
            stage_logits, domain_logits, gender_logits, age_pred = model(batch_x, alpha=alpha)
            
            loss_stage = criterion_stage(stage_logits, batch_y_stage)
            loss_domain = criterion_domain(domain_logits, batch_y_domain)
            loss_gender = criterion_gender(gender_logits, batch_y_gender)
            loss_age = masked_mse_loss(age_pred, batch_y_age, batch_y_age_mask)
            loss = (
                loss_stage
                + domain_loss_weight * loss_domain
                + gender_loss_weight * loss_gender
                + age_loss_weight * loss_age
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_batch += 1
            
        scheduler.step()
    return model

def train_and_evaluate(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    (train_dataset, cal_dataset, test_dataset,
     input_dim, num_domains, num_genders,
     stage_classes, domain_classes, gender_classes,
     pos_weight_val, domain_class_weights, gender_class_weights,
     y_stage_cal, y_stage_test, y_domain_test, y_gender_test,
     y_age_test, age_mask_test, age_mean, age_std) = load_and_prepare_data(config)
     
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    cal_loader   = DataLoader(cal_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_seeds = config['train']['n_seeds']
    seeds = [42, 0, 123, 456, 789][:n_seeds]
    fixed_threshold = config['train']['fixed_threshold']

    print(f"\n{'='*50}")
    print(f" 3. Training {n_seeds}-seed ensemble on 75% data (device={device})")
    print(f"{'='*50}")
    
    test_probs_sum = np.zeros(len(test_dataset))
    cal_probs_sum  = np.zeros(len(cal_dataset))
    total_weight   = 0.0
    last_model     = None
    
    for seed in seeds:
        model = train_one(
            train_loader, input_dim, num_domains, num_genders,
            pos_weight_val, domain_class_weights, gender_class_weights, device, seed, config
        )
        last_model = model
        model.eval()
        cal_probs = []
        with torch.no_grad():
            for bx, _, _, _, _, _ in cal_loader:
                p = torch.sigmoid(model(bx.to(device), alpha=0.0)[0]).cpu().numpy().flatten()
                cal_probs.extend(p)
        cal_probs = np.array(cal_probs)
        best_acc = max(accuracy_score(y_stage_cal, (cal_probs > thr).astype(int))
                       for thr in np.arange(0.3, 0.75, 0.01))
        w = best_acc
        total_weight += w
        print(f"  seed={seed}: cal_acc={best_acc:.4f}")
        cal_probs_sum += w * cal_probs
        
        offset = 0
        with torch.no_grad():
            for bx, _, _, _, _, _ in test_loader:
                p = torch.sigmoid(model(bx.to(device), alpha=1.0)[0]).cpu().numpy().flatten()
                test_probs_sum[offset:offset+len(p)] += w * p
                offset += len(p)
                
    test_probs = test_probs_sum / total_weight
    cal_probs_avg = cal_probs_sum / total_weight
    best_thr, best_cal_acc = fixed_threshold, 0.0
    
    for thr in np.arange(0.3, 0.75, 0.01):
        a = accuracy_score(y_stage_cal, (cal_probs_avg > thr).astype(int))
        if a > best_cal_acc:
            best_cal_acc, best_thr = a, thr
            
    print(f"\n[Calibration] cal_acc={best_cal_acc:.4f}, threshold={best_thr:.2f}")
    stage_preds = (test_probs > best_thr).astype(int)
    
    last_model.eval()
    domain_preds = []
    domain_probs = []  
    gender_preds = []
    age_preds_scaled = []
    with torch.no_grad():
        for bx, _, _, _, _, _ in test_loader:
            _, dl, gl, al = last_model(bx.to(device), alpha=1.0)
            domain_prob = F.softmax(dl, dim=1)
            domain_probs.extend(domain_prob.cpu().numpy())
            domain_preds.extend(torch.argmax(dl, dim=1).cpu().numpy())
            gender_preds.extend(torch.argmax(gl, dim=1).cpu().numpy())
            age_preds_scaled.extend(al.cpu().numpy().flatten())
    domain_probs = np.array(domain_probs) 
    age_preds = np.array(age_preds_scaled) * age_std + age_mean

    print(f"\n{'='*50}")
    print(" 4. Test Set Evaluation Results")
    print(f"{'='*50}")
    print(">>> Main Task: Stage Classification Report")
    print(classification_report(y_stage_test, stage_preds,
                                labels=range(len(stage_classes)),
                                target_names=stage_classes, zero_division=0))
    print(">>> Adversarial Task: Domain Classification Report")
    print(classification_report(y_domain_test, domain_preds,
                                labels=range(len(domain_classes)),
                                target_names=domain_classes, zero_division=0))
    print(">>> Adversarial Task: Gender Classification Report")
    print(classification_report(y_gender_test, gender_preds,
                                labels=range(len(gender_classes)),
                                target_names=gender_classes, zero_division=0))

    known_age_mask = age_mask_test.astype(bool)
    age_mae = mean_absolute_error(y_age_test[known_age_mask], age_preds[known_age_mask])
    print(f">>> Adversarial Task: Age Regression MAE (years): {age_mae:.2f}")
                                
    acc = accuracy_score(y_stage_test, stage_preds)
    print(f"\n>>> FINAL TEST ACCURACY: {acc:.4f}")
    
    fpr_stage, tpr_stage, _ = roc_curve(y_stage_test, test_probs)
    roc_auc_stage = auc(fpr_stage, tpr_stage)
    print(f">>> FINAL TEST AUC (Stage): {roc_auc_stage:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_stage, tpr_stage, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_stage:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) - Stage Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_stage_classification.png', dpi=300, bbox_inches='tight')
    plt.close()

    y_domain_test_bin = label_binarize(y_domain_test, classes=range(num_domains))
    fpr_domain = dict()
    tpr_domain = dict()
    roc_auc_domain = dict()
    
    for i in range(num_domains):
        if np.sum(y_domain_test_bin[:, i]) > 0:
            fpr_domain[i], tpr_domain[i], _ = roc_curve(y_domain_test_bin[:, i], domain_probs[:, i])
            roc_auc_domain[i] = auc(fpr_domain[i], tpr_domain[i])
        else:
            roc_auc_domain[i] = np.nan
            
    plt.figure(figsize=(12, 8))
    valid_classes = [(i, roc_auc_domain[i]) for i in range(num_domains) if not np.isnan(roc_auc_domain[i])]
    valid_classes.sort(key=lambda x: x[1], reverse=True)
    cmap = plt.get_cmap('tab20') 
    
    for idx, (i, auc_val) in enumerate(valid_classes):
        plt.plot(fpr_domain[i], tpr_domain[i], color=cmap(idx % 20), lw=2, 
                 label=f'{domain_classes[i]} (AUC = {auc_val:.3f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC - Domain Classification (One-vs-Rest)')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_domain_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return acc

if __name__ == "__main__":
    train_and_evaluate(config_path='config.yaml')
