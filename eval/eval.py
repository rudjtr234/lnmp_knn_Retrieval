import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Read files
with open('answer_label.json', 'r') as f:
    ground_truth = json.load(f)

with open('lnmp_predictions_v0.1.1.json', 'r') as f:
    predictions = json.load(f)

# Create dictionaries
gt_dict = {}
for record in ground_truth:
    record_id = str(record.get('id', ''))
    if record_id.endswith('.tiff'):
        record_id = record_id[:-5]
    gt_dict[record_id] = record

pred_dict = {}
for record in predictions:
    record_id = str(record.get('id', ''))
    if record_id.endswith('.tiff'):
        record_id = record_id[:-5]
    pred_dict[record_id] = record

# Extract paired data
fields = ['metastasis']
for field in fields:
    y_true = []
    y_pred = []
    
    for record_id in gt_dict:
        if record_id in pred_dict:
            gt_value = gt_dict[record_id].get(field)
            pred_value = pred_dict[record_id].get(field)
            
            if gt_value is not None and pred_value is not None:
                y_true.append(gt_value)
                y_pred.append(pred_value)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        continue
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Bootstrap for 95% CI
    n_bootstrap = 1000
    metrics = {'accuracy': [], 'precision': [], 'sensitivity': [], 'specificity': [], 'f1': []}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            
            metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics['sensitivity'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics['specificity'].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
            metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        except:
            continue
    
    # Print results
    print(f"{field}:")
    print(f"  Accuracy: {accuracy:.3f} (95% CI: {np.percentile(metrics['accuracy'], 2.5):.3f}-{np.percentile(metrics['accuracy'], 97.5):.3f})")
    print(f"  Precision: {precision:.3f} (95% CI: {np.percentile(metrics['precision'], 2.5):.3f}-{np.percentile(metrics['precision'], 97.5):.3f})")
    print(f"  Sensitivity: {sensitivity:.3f} (95% CI: {np.percentile(metrics['sensitivity'], 2.5):.3f}-{np.percentile(metrics['sensitivity'], 97.5):.3f})")
    print(f"  Specificity: {specificity:.3f} (95% CI: {np.percentile(metrics['specificity'], 2.5):.3f}-{np.percentile(metrics['specificity'], 97.5):.3f})")
    print(f"  F1-score: {f1:.3f} (95% CI: {np.percentile(metrics['f1'], 2.5):.3f}-{np.percentile(metrics['f1'], 97.5):.3f})")
