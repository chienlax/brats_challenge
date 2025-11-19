import json
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

def load_data(filepath):
    """Loads the JSON summary file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def flatten_data(data):
    """
    Flattens the nested metric_per_case list into a Pandas DataFrame.
    Extracts ALL metrics present in the JSON (Dice, IoU, TP, TN, FP, FN, volumes).
    """
    cases = []
    for entry in data['metric_per_case']:
        # Extract Case ID from filename, handling potential path variations
        filename = os.path.basename(entry['prediction_file'])
        case_id = filename.replace('.nii.gz', '')
        
        row = {'case_id': case_id}
        
        valid_dice_scores = []
        
        for class_id, metrics in entry['metrics'].items():
            # metrics dictionary keys: 'Dice', 'FN', 'FP', 'IoU', 'TN', 'TP', 'n_pred', 'n_ref'
            
            # Helper to safely get float values, treating None as NaN or 0.0 where appropriate
            def get_val(key, default=np.nan):
                val = metrics.get(key)
                return float(val) if val is not None else default

            # Extract all raw metrics
            dice = get_val('Dice')
            iou = get_val('IoU')
            fn = get_val('FN', 0.0)
            fp = get_val('FP', 0.0)
            tn = get_val('TN', 0.0)
            tp = get_val('TP', 0.0)
            ref_vol = get_val('n_ref', 0.0)
            pred_vol = get_val('n_pred', 0.0)

            # Store in row with class prefix
            prefix = f'class_{class_id}'
            row[f'{prefix}_dice'] = dice
            row[f'{prefix}_iou'] = iou
            row[f'{prefix}_fn'] = fn
            row[f'{prefix}_fp'] = fp
            row[f'{prefix}_tn'] = tn
            row[f'{prefix}_tp'] = tp
            row[f'{prefix}_ref_vol'] = ref_vol
            row[f'{prefix}_pred_vol'] = pred_vol
            
            # Derived Metric: Relative Volume Error (RVE)
            # (Pred - Ref) / Ref. 
            if ref_vol > 0:
                row[f'{prefix}_rve'] = (pred_vol - ref_vol) / ref_vol
            elif pred_vol > 0:
                row[f'{prefix}_rve'] = 1.0 # Pure hallucination
            else:
                row[f'{prefix}_rve'] = 0.0 # Correctly empty

            # Collect valid Dice for calculating case average
            # Special handling: If reference is empty and model predicts empty, Dice is NaN in JSON
            # but effectively 1.0 for scoring purposes (Perfect TN).
            if pd.isna(dice):
                if ref_vol == 0 and pred_vol == 0:
                     valid_dice_scores.append(1.0)
            else:
                valid_dice_scores.append(dice)

        # Calculate average Dice for this case across available classes
        row['avg_dice'] = np.mean(valid_dice_scores) if valid_dice_scores else 0.0
        cases.append(row)
        
    return pd.DataFrame(cases)

def get_top_bottom_cases(df, sort_col, n=5):
    """Returns top and bottom N rows sorted by sort_col."""
    # Drop NaNs for sorting to avoid issues
    df_clean = df.dropna(subset=[sort_col])
    bottom = df_clean.sort_values(by=sort_col, ascending=True).head(n)
    top = df_clean.sort_values(by=sort_col, ascending=False).head(n)
    return top, bottom

def analyze_correlations(df, class_ids):
    """
    Analyzes correlations to find patterns:
    1. Volume vs Performance (Do small targets fail?)
    2. Inter-class interference (Does Class A volume affect Class B performance?)
    """
    correlations = []
    
    # 1. Volume Bias (Self-Correlation)
    for cid in class_ids:
        vol_col = f'class_{cid}_ref_vol'
        dice_col = f'class_{cid}_dice'
        
        # Only look at cases where the object actually exists (Ref Vol > 0)
        subset = df[df[vol_col] > 0].dropna(subset=[dice_col])
        
        if len(subset) > 5:
            corr, _ = pearsonr(subset[vol_col], subset[dice_col])
            correlations.append({
                'type': 'Volume Bias',
                'class': cid,
                'description': f'Correlation between Class {cid} Volume and Dice',
                'value': corr,
                'interpretation': 'Positive correlation implies model struggles with small lesions.' if corr > 0.3 else 'Performance is independent of size.'
            })

    # 2. Inter-class Interference (e.g., Does large Edema hurt Core segmentation?)
    # Checking correlation between Volume of X and Dice of Y
    for cid_x in class_ids:
        for cid_y in class_ids:
            if cid_x == cid_y: continue
            
            vol_col = f'class_{cid_x}_ref_vol'
            dice_col = f'class_{cid_y}_dice'
            
            subset = df[df[vol_col] > 0].dropna(subset=[dice_col])
            if len(subset) > 10:
                corr, _ = pearsonr(subset[vol_col], subset[dice_col])
                # Filter for significant correlations to reduce noise
                if abs(corr) > 0.25: 
                    correlations.append({
                        'type': 'Inter-class',
                        'class': f'{cid_x}->{cid_y}',
                        'description': f'Vol({cid_x}) vs Dice({cid_y})',
                        'value': corr,
                        'interpretation': 'Large Class ' + cid_x + ' degrades Class ' + cid_y + ' performance.' if corr < 0 else 'Large Class ' + cid_x + ' aids Class ' + cid_y + '.'
                    })
                    
    return correlations

def generate_markdown_report(json_data, df, output_file="comprehensive_analysis.md"):
    md = []
    md.append("# Comprehensive Segmentation Performance Analysis Report\n")
    
    # --- Section 1: Overall Statistics ---
    md.append("## 1. Global Performance Statistics")
    md.append("Analysis of aggregated metrics across the entire dataset.")
    
    # Foreground Mean Table
    fg = json_data['foreground_mean']
    md.append("\n### A. Foreground Averages (All Classes)")
    md.append("| Metric | Value | Description |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Dice** | {fg.get('Dice', 0):.4f} | Similarity Coefficient |")
    md.append(f"| **IoU** | {fg.get('IoU', 0):.4f} | Intersection over Union |")
    md.append(f"| **Precision** | {fg['TP'] / (fg['TP'] + fg['FP']):.4f} | PPV (TP / (TP + FP)) |")
    md.append(f"| **Recall** | {fg['TP'] / (fg['TP'] + fg['FN']):.4f} | Sensitivity (TP / (TP + FN)) |")
    
    # Per-Class Table
    md.append("\n### B. Per-Class Breakdown")
    class_ids = sorted(json_data['mean'].keys())
    
    # Header
    header = "| Class | Dice | IoU | Precision | Recall | FN (Avg) | FP (Avg) |"
    md.append(header)
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for cid in class_ids:
        m = json_data['mean'][cid]
        tp = m.get('TP', 0)
        fp = m.get('FP', 0)
        fn = m.get('FN', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        md.append(f"| {cid} | {m['Dice']:.4f} | {m['IoU']:.4f} | {precision:.4f} | {recall:.4f} | {fn:.1f} | {fp:.1f} |")

    # --- Section 2: Case Analysis ---
    md.append("\n## 2. Case-Level Analysis")
    md.append("identifying specific cases that represent the best and worst performance.")

    # General Best/Worst
    top_cases, bot_cases = get_top_bottom_cases(df, 'avg_dice', 50)

    def add_detailed_case_table(cases_df, title):
        md.append(f"\n### {title}")
        
        # Dynamic Header
        headers = ["Case ID", "Avg Dice"]
        metrics = ['dice', 'iou', 'rve', 'ref_vol', 'pred_vol', 'fn', 'fp', 'tp', 'tn']
        
        for cid in class_ids:
            for m in metrics:
                headers.append(f"C{cid}_{m}")
        
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join([":---"] * len(headers)) + " |")
        
        for _, row in cases_df.iterrows():
            line = [f"{row['case_id']}", f"{row['avg_dice']:.4f}"]
            for cid in class_ids:
                prefix = f'class_{cid}'
                
                def fmt(val, is_float=True):
                    if pd.isna(val): return "-"
                    return f"{val:.4f}" if is_float else f"{val:.1f}"

                line.append(fmt(row.get(f'{prefix}_dice'), True))
                line.append(fmt(row.get(f'{prefix}_iou'), True))
                line.append(fmt(row.get(f'{prefix}_rve'), True))
                line.append(fmt(row.get(f'{prefix}_ref_vol'), False))
                line.append(fmt(row.get(f'{prefix}_pred_vol'), False))
                line.append(fmt(row.get(f'{prefix}_fn'), False))
                line.append(fmt(row.get(f'{prefix}_fp'), False))
                line.append(fmt(row.get(f'{prefix}_tp'), False))
                line.append(fmt(row.get(f'{prefix}_tn'), False))
            md.append("| " + " | ".join(line) + " |")

    add_detailed_case_table(bot_cases, "A. 50 Overall Worst Cases (Lowest Average Dice)")
    add_detailed_case_table(top_cases, "B. 50 Overall Best Cases")

    # Worst per class
    md.append("\n### C. 50 Most Problematic Cases Per Class")
    for cid in class_ids:
        col = f'class_{cid}_dice'
        # Filter for existing objects only
        subset = df[df[f'class_{cid}_ref_vol'] > 0]
        if not subset.empty:
            worst_cases = subset.sort_values(by=col).head(50)
            add_detailed_case_table(worst_cases, f"Class {cid} Worst 50")

    # --- Section 3: Pattern Discovery ---
    md.append("\n## 3. Pattern Discovery & Correlations")
    
    corrs = analyze_correlations(df, class_ids)
    
    # Patterns in Problems
    md.append("\n### A. Patterns in Problematic Predictions")
    md.append("Correlation analysis reveals the following systematic issues:")
    
    vol_biases = [c for c in corrs if c['type'] == 'Volume Bias']
    md.append("\n**1. Sensitivity to Object Size (Volume Bias):**")
    md.append("| Class | Correlation (Vol vs Dice) | Interpretation |")
    md.append("| :--- | :--- | :--- |")
    for c in vol_biases:
        md.append(f"| {c['class']} | {c['value']:.3f} | {c['interpretation']} |")
    
    inter_class = [c for c in corrs if c['type'] == 'Inter-class']
    if inter_class:
        md.append("\n**2. Inter-Class Interference:**")
        for c in inter_class:
             md.append(f"* **{c['description']}:** {c['value']:.3f} ({c['interpretation']})")

    # Common patterns in False Positives
    md.append("\n**3. False Positive Tendencies:**")
    for cid in class_ids:
        fp_col = f'class_{cid}_fp'
        mean_fp = df[fp_col].mean()
        mean_ref = df[f'class_{cid}_ref_vol'].mean()
        fp_ratio = (mean_fp / mean_ref) * 100 if mean_ref > 0 else 0
        if fp_ratio > 10:
            md.append(f"* **Class {cid}:** High background noise. False Positives Avg Volume is {fp_ratio:.1f}% of the average object size.")

    # Patterns in Best Cases
    md.append("\n### B. Patterns in Best Cases")
    md.append("What defines a 'easy' case?")
    
    # Compare volume of best cases vs average
    mean_vol_all = df[[f'class_{c}_ref_vol' for c in class_ids]].sum(axis=1).mean()
    mean_vol_best = top_cases[[f'class_{c}_ref_vol' for c in class_ids]].sum(axis=1).mean()
    
    if mean_vol_best > mean_vol_all * 1.2:
        md.append(f"* **Larger Tumors:** The best cases tend to have significantly larger tumor volumes ({mean_vol_best:.0f} voxels) compared to the dataset average ({mean_vol_all:.0f}).")
    elif mean_vol_best < mean_vol_all * 0.8:
        md.append(f"* **Smaller Tumors:** The best cases tend to have smaller tumor volumes ({mean_vol_best:.0f} voxels) compared to average.")
    else:
        md.append("* **Volume Neutral:** Success does not strictly depend on total tumor burden.")

    with open(output_file, 'w') as f:
        f.write("\n".join(md))
    
    print(f"Report generated: {os.path.abspath(output_file)}")

def main():
    input_file = 'summary.json'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading data...")
    json_data = load_data(input_file)
    
    print("Processing statistics...")
    df = flatten_data(json_data)
    
    print("Analyzing patterns and generating report...")
    generate_markdown_report(json_data, df)

if __name__ == "__main__":
    main()