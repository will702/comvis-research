"""
evaluate_saved_models.py
Load all saved .pkl models, evaluate on saved test features,
then print ranked summary tables (overall + per-class for acne2 & acne3)
and save comparison charts.

Run with:
    python scripts/evaluate_saved_models.py
"""

import os
import sys
import glob

# Force UTF-8 output on Windows so box-drawing chars print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(MODELS_DIR, 'eval_results')
CM_DIR      = os.path.join(RESULTS_DIR, 'confusion_matrices')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

CLASSES = ['acne1_1024', 'acne2_1024', 'acne3_1024']

# Friendly display names mapped from pkl filenames
PKL_NAME_MAP = {
    'acne_svm_rbf_model':             'SVM (RBF)',
    'acne_svm_polynomial_model':      'SVM (Polynomial)',
    'acne_svm_linear_model':          'SVM (Linear)',
    'acne_logistic_regression_model': 'Logistic Regression',
    'acne_knn_model':                 'KNN',
    'acne_random_forest_model':       'Random Forest',
    'acne_extra_trees_model':         'Extra Trees',
    'acne_adaboost_model':            'AdaBoost',
    'acne_gradient_boosting_model':   'Gradient Boosting',
    'acne_histgradientboosting_model':'HistGradientBoosting',
    'acne_mlp_model':                 'MLP',
    'acne_lda_model':                 'LDA',
    'acne_qda_model':                 'QDA',
    'acne_gaussian_nb_model':         'Gaussian NB',
    'acne_calibrated_linear_svm_model': 'Calibrated Linear SVM',
    'acne_bagging_svm_model':         'Bagging SVM',
    'acne_sgd_classifier_model':      'SGD Classifier',
    'acne_ridge_classifier_model':    'Ridge Classifier',
    'acne_catboost_model':            'CatBoost',
    'acne_voting_ensemble_model':     'Voting Ensemble (top-3)',
    'acne_stacking_ensemble_model':   'Stacking Ensemble (top-5)',
    # XGBoost / LightGBM (if they were trained)
    'acne_xgboost_model':             'XGBoost',
    'acne_lightgbm_model':            'LightGBM',
}


# ── Load test data ─────────────────────────────────────────────────────────────
def load_test_data():
    npy_path = os.path.join(MODELS_DIR, 'test_data_features.npy')
    if not os.path.exists(npy_path):
        print(f"ERROR: test_data_features.npy not found at {npy_path}")
        print("       Run train.py first to generate test features.")
        sys.exit(1)
    data = np.load(npy_path, allow_pickle=True).item()
    return data['X_test'], data['y_test']


# ── Discover pkl files ─────────────────────────────────────────────────────────
def discover_models():
    pkl_files = sorted(glob.glob(os.path.join(MODELS_DIR, 'acne_*_model.pkl')))
    if not pkl_files:
        print(f"ERROR: No .pkl model files found in {MODELS_DIR}")
        sys.exit(1)
    return pkl_files


# ── Evaluate one model ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, display_name):
    y_pred    = model.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec       = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_cls    = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec_cls  = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_cls   = recall_score(y_test, y_pred, average=None, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    row = {
        'Model':     display_name,
        'Accuracy':  acc,
        'Precision': prec,
        'Recall':    rec,
        'F1':        f1,
        **{f'P_{c}':  prec_cls[i] for i, c in enumerate(CLASSES)},
        **{f'R_{c}':  rec_cls[i]  for i, c in enumerate(CLASSES)},
        **{f'F1_{c}': f1_cls[i]   for i, c in enumerate(CLASSES)},
    }

    print(f"\n{'─' * 56}")
    print(f"  {display_name}")
    print(f"{'─' * 56}")
    print(f"  Accuracy : {acc:.4f}   Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}   F1-Score : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix — {display_name}', fontsize=10)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    safe = display_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    plt.savefig(os.path.join(CM_DIR, f'cm_{safe}.png'), dpi=120)
    plt.close()

    return row


# ── Print box table ────────────────────────────────────────────────────────────
def print_box_table(df, title, cols, headers, highlight_col=None):
    col_w = 28
    num_w = 9
    sep = '+' + '-' * (col_w + 2) + ('+' + '-' * (num_w + 2)) * len(cols) + '+' + '-' * 8 + '+'
    width = len(sep)

    print(f"\n{'=' * width}")
    print(title.center(width, '='))
    print('=' * width)
    print(sep)
    hdr = f"| {'Model':<{col_w}} " + ''.join(f"| {h:^{num_w}} " for h in headers) + f"| {'Rank':^6} |"
    print(hdr)
    print(sep)

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        star = ' ★' if rank == 1 else '  '
        vals = ''.join(f"| {row[c]:^{num_w}.4f} " for c in cols)
        print(f"| {row['Model']:<{col_w}} {vals}| {rank:^4}{star} |")
    print(sep)


# ── Summary charts ─────────────────────────────────────────────────────────────
def save_comparison_chart(summary_df):
    plot_df   = summary_df.sort_values('F1', ascending=True)
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(11, max(7, len(plot_df) * 0.46)))
    y     = np.arange(len(plot_df))
    bar_h = 0.18

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * bar_h
        bars = ax.barh(y + offset, plot_df[metric], bar_h,
                       label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, plot_df[metric]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Model'], fontsize=8)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel('Score')
    ax.set_title('Model Comparison — Acne Severity Classification\n(sorted by F1, ascending)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

    # Highlight overall best
    best_model = summary_df.iloc[0]['Model']
    labels = [t.get_text() for t in ax.get_yticklabels()]
    if best_model in labels:
        idx = labels.index(best_model)
        ax.get_yticklabels()[idx].set_color('#C44E52')
        ax.get_yticklabels()[idx].set_fontweight('bold')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def save_per_class_chart(summary_df):
    target_classes = CLASSES[1:]
    cls_colors = ['#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, len(target_classes),
                             figsize=(18, max(7, len(summary_df) * 0.44)))
    if len(target_classes) == 1:
        axes = [axes]

    fig.suptitle('Per-Class F1 Ranking — Acne2 vs Acne3',
                 fontsize=13, fontweight='bold')

    for ax, cls_name, col in zip(axes, target_classes, cls_colors):
        f1_col = f'F1_{cls_name}'
        p_col  = f'P_{cls_name}'
        r_col  = f'R_{cls_name}'
        if f1_col not in summary_df.columns:
            continue

        cls_sorted = summary_df[['Model', p_col, r_col, f1_col]] \
            .sort_values(f1_col, ascending=True).reset_index(drop=True)

        best_model = cls_sorted['Model'].iloc[-1]
        best_val   = cls_sorted[f1_col].iloc[-1]
        bar_colors = [col if m == best_model else '#cccccc'
                      for m in cls_sorted['Model']]

        y     = np.arange(len(cls_sorted))
        bar_h = 0.25
        b_f1 = ax.barh(y + bar_h,  cls_sorted[f1_col], bar_h,
                        label='F1',        color=bar_colors, alpha=0.9)
        ax.barh(y,            cls_sorted[p_col],  bar_h,
                label='Precision', color=col, alpha=0.45)
        ax.barh(y - bar_h,  cls_sorted[r_col],  bar_h,
                label='Recall',    color=col, alpha=0.25)

        for bar, val in zip(b_f1, cls_sorted[f1_col]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=6.5)

        ax.set_yticks(y)
        ax.set_yticklabels(cls_sorted['Model'], fontsize=7.5)
        ax.set_xlim(0, 1.16)
        ax.set_xlabel('Score')
        ax.set_title(f'{cls_name}\n(★ best F1: {best_model})',
                     fontsize=9, fontweight='bold', color=col)
        ax.axvline(x=best_val, color=col, linestyle='--',
                   linewidth=1.2, alpha=0.7)
        ax.legend(loc='lower right', fontsize=7)

        n = len(cls_sorted)
        for i, (_, row) in enumerate(cls_sorted.iterrows()):
            rank = n - i
            ax.text(1.13, y[i], f'#{rank}', va='center',
                    fontsize=6.5, color='#444444')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'per_class_ranking.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def save_best_per_metric_chart(summary_df, best_per_metric):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)
    fig.suptitle('Best Model per Metric', fontsize=12, fontweight='bold')

    for ax, metric, color in zip(axes, metrics, colors):
        s = summary_df[['Model', metric]].sort_values(metric, ascending=False).reset_index(drop=True)
        bar_colors = [color if m == best_per_metric[metric] else '#cccccc' for m in s['Model']]
        ax.barh(s['Model'], s[metric], color=bar_colors, edgecolor='white')
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='y', labelsize=7)
        ax.invert_yaxis()
        best_val = s[metric].iloc[0]
        ax.axvline(x=best_val, color=color, linestyle='--', linewidth=1, alpha=0.6)
        ax.text(best_val - 0.01, -0.5, f'{best_val:.4f}',
                color=color, fontsize=8, fontweight='bold', ha='right')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'best_per_metric.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ACNE SEVERITY — EVALUATE SAVED MODELS")
    print("=" * 60)

    X_test, y_test = load_test_data()
    print(f"\nTest set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes : {CLASSES}\n")

    pkl_files = discover_models()
    print(f"Found {len(pkl_files)} model file(s).\n")

    results = []
    failed  = []

    for pkl_path in pkl_files:
        stem = os.path.splitext(os.path.basename(pkl_path))[0]
        display_name = PKL_NAME_MAP.get(stem, stem)
        print(f"Loading  {display_name}  ({os.path.basename(pkl_path)}) ...")

        try:
            model = joblib.load(pkl_path)
            row   = evaluate_model(model, X_test, y_test, display_name)
            results.append(row)
        except Exception as e:
            print(f"  !! FAILED: {e}")
            failed.append(display_name)

    if not results:
        print("\nERROR: All models failed to load/evaluate.")
        sys.exit(1)

    # ── Build summary DataFrame ───────────────────────────────────────────────
    summary_df = pd.DataFrame(results).sort_values('F1', ascending=False).reset_index(drop=True)
    summary_df['Rank'] = summary_df.index + 1

    overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    best_per_metric = {
        m: summary_df.loc[summary_df[m].idxmax(), 'Model']
        for m in overall_metrics
    }

    # ── Overall ranking table ─────────────────────────────────────────────────
    print_box_table(
        summary_df,
        title=' OVERALL RANKING (sorted by weighted F1) ',
        cols=['Accuracy', 'Precision', 'Recall', 'F1'],
        headers=['Accuracy', 'Precision', 'Recall  ', 'F1-Score'],
        highlight_col='F1',
    )

    # ── Best per metric summary ───────────────────────────────────────────────
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│          BEST MODEL PER METRIC (OVERALL)                 │")
    print("├───────────────┬──────────────────────────────────────────┤")
    print(f"│ {'Metric':<13} │ {'Best Model':<40} │")
    print("├───────────────┼──────────────────────────────────────────┤")
    for m in overall_metrics:
        bm  = best_per_metric[m]
        bv  = summary_df.loc[summary_df['Model'] == bm, m].values[0]
        print(f"│ {m:<13} │ {bm:<32} ({bv:.4f}) │")
    print("└───────────────┴──────────────────────────────────────────┘")

    # ── Per-class ranking: acne2 & acne3 ─────────────────────────────────────
    for cls in CLASSES[1:]:   # skip acne1
        f1_col = f'F1_{cls}'
        p_col  = f'P_{cls}'
        r_col  = f'R_{cls}'
        if f1_col not in summary_df.columns:
            continue

        cls_df = summary_df[['Model', p_col, r_col, f1_col]] \
            .sort_values(f1_col, ascending=False).reset_index(drop=True)

        best_f1   = cls_df.iloc[0]['Model']
        best_prec = cls_df.loc[cls_df[p_col].idxmax(), 'Model']
        best_rec  = cls_df.loc[cls_df[r_col].idxmax(), 'Model']

        print_box_table(
            cls_df,
            title=f' PER-CLASS RANKING  ›  {cls} ',
            cols=[p_col, r_col, f1_col],
            headers=['Precision', 'Recall  ', 'F1-Score'],
        )
        print(f"  Best F1        → {best_f1}")
        print(f"  Best Precision → {best_prec}")
        print(f"  Best Recall    → {best_rec}")

    # ── Overall champion ──────────────────────────────────────────────────────
    champ = summary_df.iloc[0]
    print(f"\n{'★' * 60}")
    print(f"  OVERALL CHAMPION (weighted F1): {champ['Model']}")
    print(f"  Accuracy={champ['Accuracy']:.4f}  Precision={champ['Precision']:.4f}"
          f"  Recall={champ['Recall']:.4f}  F1={champ['F1']:.4f}")
    for cls in CLASSES:
        f1_c = summary_df.iloc[0].get(f'F1_{cls}', float('nan'))
        print(f"  {cls}: F1={f1_c:.4f}")
    print(f"{'★' * 60}")

    if failed:
        print(f"\n  Skipped (load error): {failed}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

    # ── Save charts ───────────────────────────────────────────────────────────
    save_comparison_chart(summary_df)
    save_best_per_metric_chart(summary_df, best_per_metric)
    save_per_class_chart(summary_df)
    print("\nDone.")


if __name__ == '__main__':
    main()
