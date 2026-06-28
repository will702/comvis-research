"""
train_6535.py
=============
Alternative training pipeline with two key differences vs src/train.py:

  1. 65 / 35 train-test split  (vs 80/20)
       → larger test set gives a more robust evaluation of generalisation.

  2. Balanced test set
       → after splitting, the test set is downsampled so each class has the
         same number of samples (= count of the smallest class).
       → removes the majority-class bias that inflates weighted metrics when
         one class dominates the test set.

All outputs are isolated so the original pipeline is never touched:
  data_split_6535/   — train & test image directories
  models_6535/       — trained .pkl models + evaluation outputs
"""

import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    VotingClassifier, BaggingClassifier, StackingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BASE_DIR, DATA_DIR, CLASSES
from src.preprocessing import FaceProcessor
from src.features import FeatureExtractor
from src.data_prep import apply_augmentation

# ── Output directories (isolated from original pipeline) ────────────────────
SPLIT_DIR   = os.path.join(BASE_DIR, 'data_split_6535')
OUT_DIR     = os.path.join(BASE_DIR, 'models_6535')
EVAL_DIR    = os.path.join(OUT_DIR,  'evaluation_test')

for d in [OUT_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Optional boosting libraries ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier;      _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier;   _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier; _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ============================================================================
# STEP 1 — Data split + augmentation
# ============================================================================

def prepare_split(test_size=0.35, random_state=42, target_count=300):
    """
    65/35 stratified split, then:
      - Train  : balance to target_count per class (downsample or augment)
      - Test   : copy originals only — balanced AFTER feature extraction
    """
    train_dir = os.path.join(SPLIT_DIR, 'train')
    test_dir  = os.path.join(SPLIT_DIR, 'test')

    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        for cls in CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)

    # Collect all source images
    data = []
    for cls_name in CLASSES:
        cls_path = os.path.join(DATA_DIR, cls_name)
        if not os.path.exists(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.endswith('.jpg'):
                data.append((os.path.join(cls_path, f), cls_name))

    if not data:
        raise FileNotFoundError("No images found in DATA_DIR.")

    paths, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # ── Copy test set as-is (no augmentation) ───────────────────────────────
    print(f"\nCopying {len(X_test)} images to test set (untouched)...")
    for path, cls in zip(X_test, y_test):
        shutil.copy(path, os.path.join(test_dir, cls, os.path.basename(path)))

    # ── Build balanced train set ─────────────────────────────────────────────
    train_classes = {cls: [] for cls in CLASSES}
    for path, cls in zip(X_train, y_train):
        train_classes[cls].append(path)

    print(f"\nOriginal train counts : { {c: len(train_classes[c]) for c in CLASSES} }")
    print(f"Target per class      : {target_count}")

    for cls in CLASSES:
        cls_paths   = train_classes[cls]
        target_path = os.path.join(train_dir, cls)
        current     = len(cls_paths)

        if current >= target_count:
            print(f"  [{cls}] Downsample {current} → {target_count}")
            for p in random.sample(cls_paths, target_count):
                shutil.copy(p, os.path.join(target_path, os.path.basename(p)))
        else:
            print(f"  [{cls}] Oversample {current} → {target_count} "
                  f"(+{target_count - current} augmented)")
            for p in cls_paths:
                shutil.copy(p, os.path.join(target_path, os.path.basename(p)))
            needed = target_count - current
            for i in range(needed):
                src = random.choice(cls_paths)
                img = cv2.imread(src)
                if img is None:
                    continue
                aug_img = apply_augmentation(img)
                cv2.imwrite(
                    os.path.join(target_path, f"aug_{i}_{os.path.basename(src)}"),
                    aug_img,
                )

    print("\nData split complete.")
    for split, d in [('Train', train_dir), ('Test', test_dir)]:
        print(f"\n{split} set:")
        for cls in CLASSES:
            n = len([f for f in os.listdir(os.path.join(d, cls))
                     if f.endswith('.jpg')])
            print(f"  {cls}: {n}")


# ============================================================================
# STEP 2 — Feature extraction
# ============================================================================

def extract_features_from_dir(directory, processor, extractor):
    X, y, paths = [], [], []
    for label, cls_name in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls_name)
        if not os.path.exists(cls_path):
            continue
        for f in sorted(os.listdir(cls_path)):
            if not f.endswith('.jpg'):
                continue
            img_path = os.path.join(cls_path, f)
            img_resized, skin_mask = processor.preprocess_image(img_path)
            if img_resized is not None and skin_mask is not None:
                feat = extractor.extract(img_resized, skin_mask)
            else:
                feat = np.zeros(extractor.FEATURE_DIM)
            X.append(feat)
            y.append(label)
            paths.append(img_path)
    return np.array(X), np.array(y)


def balance_test_set(X_test, y_test, random_state=42):
    """
    Downsample test set so each class has the same number of samples
    (= count of the minority class). This removes majority-class bias
    from weighted metrics and makes per-class evaluation fair.
    """
    rng = np.random.default_rng(random_state)
    class_indices = {c: np.where(y_test == c)[0] for c in np.unique(y_test)}
    min_count = min(len(idx) for idx in class_indices.values())

    print(f"\nTest set before balancing: { {c: len(v) for c, v in class_indices.items()} }")
    print(f"Downsampling each class to: {min_count} samples")

    selected = []
    for cls, idx in class_indices.items():
        chosen = rng.choice(idx, size=min_count, replace=False)
        selected.extend(chosen)

    selected = sorted(selected)
    X_bal = X_test[selected]
    y_bal = y_test[selected]

    print(f"Test set after  balancing: { {c: int(np.sum(y_bal == c)) for c in np.unique(y_bal)} }")
    print(f"Total test samples        : {len(y_bal)}")
    return X_bal, y_bal


# ============================================================================
# STEP 3 — Training & evaluation (mirrors src/train.py, different output dirs)
# ============================================================================

def train_and_evaluate():
    train_dir = os.path.join(SPLIT_DIR, 'train')
    test_dir  = os.path.join(SPLIT_DIR, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("data_split_6535 not found — running prepare_split() first...")
        prepare_split()

    processor = FaceProcessor()
    extractor = FeatureExtractor()

    print(f"\nExtracting features — Train ({train_dir}) ...")
    X_train, y_train = extract_features_from_dir(train_dir, processor, extractor)

    print(f"\nExtracting features — Test ({test_dir}) ...")
    X_test_raw, y_test_raw = extract_features_from_dir(test_dir, processor, extractor)

    # Balance the test set
    X_test, y_test = balance_test_set(X_test_raw, y_test_raw)

    print(f"\nFeature vector size : {X_train.shape[1]}")
    print(f"X_train shape       : {X_train.shape}")
    print(f"X_test  shape       : {X_test.shape}  (balanced)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'SVM (RBF)': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='rbf', probability=True, random_state=42))]),
            {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 0.001, 0.01, 0.1]},
        ),
        'SVM (Polynomial)': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='poly', probability=True, random_state=42))]),
            {'clf__C': [0.1, 1, 10], 'clf__degree': [2, 3], 'clf__gamma': ['scale', 0.01]},
        ),
        'SVM (Linear)': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='linear', probability=True, random_state=42))]),
            {'clf__C': [0.01, 0.1, 1, 10]},
        ),
        'Logistic Regression': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=2000, solver='lbfgs',
                                                  multi_class='multinomial', random_state=42))]),
            {'clf__C': [0.01, 0.1, 1, 10, 100]},
        ),
        'KNN': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', KNeighborsClassifier(metric='minkowski'))]),
            {'clf__n_neighbors': [3, 5, 7, 11],
             'clf__weights': ['uniform', 'distance'],
             'clf__p': [1, 2]},
        ),
        'Random Forest': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
            {'clf__n_estimators': [100, 200, 300],
             'clf__max_depth': [None, 15, 30],
             'clf__min_samples_split': [2, 5],
             'clf__max_features': ['sqrt', 'log2']},
        ),
        'Extra Trees': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', ExtraTreesClassifier(random_state=42, n_jobs=-1))]),
            {'clf__n_estimators': [100, 200, 300],
             'clf__max_depth': [None, 15, 30],
             'clf__min_samples_split': [2, 5]},
        ),
        'AdaBoost': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', AdaBoostClassifier(algorithm='SAMME', random_state=42))]),
            {'clf__n_estimators': [50, 100, 200], 'clf__learning_rate': [0.5, 1.0, 1.5]},
        ),
        'Gradient Boosting': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', GradientBoostingClassifier(random_state=42))]),
            {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.05, 0.1],
             'clf__max_depth': [3, 5], 'clf__subsample': [0.8, 1.0]},
        ),
        'HistGradientBoosting': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', HistGradientBoostingClassifier(random_state=42))]),
            {'clf__max_iter': [100, 200], 'clf__learning_rate': [0.05, 0.1, 0.2],
             'clf__max_depth': [None, 5, 10], 'clf__min_samples_leaf': [10, 20]},
        ),
        'MLP': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', MLPClassifier(max_iter=500, early_stopping=True,
                                             validation_fraction=0.1, random_state=42))]),
            {'clf__hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128), (256, 128, 64)],
             'clf__activation': ['relu', 'tanh'],
             'clf__alpha': [1e-4, 1e-3, 1e-2],
             'clf__learning_rate_init': [0.001, 0.01]},
        ),
        'LDA': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', LinearDiscriminantAnalysis())]),
            {'clf__solver': ['svd', 'lsqr'], 'clf__shrinkage': [None, 'auto', 0.1, 0.5]},
        ),
        'QDA': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', QuadraticDiscriminantAnalysis())]),
            {'clf__reg_param': [0.0, 0.1, 0.3, 0.5]},
        ),
        'Gaussian NB': (
            Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
            {'clf__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        'Calibrated Linear SVM': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', CalibratedClassifierCV(
                          LinearSVC(max_iter=5000, random_state=42), cv=3))]),
            {'clf__estimator__C': [0.01, 0.1, 1, 10]},
        ),
        'Bagging SVM': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', BaggingClassifier(
                          estimator=SVC(kernel='rbf', probability=True, random_state=42),
                          random_state=42, n_jobs=-1))]),
            {'clf__n_estimators': [10, 20, 30],
             'clf__max_samples': [0.7, 0.85, 1.0],
             'clf__max_features': [0.7, 1.0]},
        ),
        'SGD Classifier': (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1))]),
            {'clf__loss': ['hinge', 'log_loss', 'modified_huber'],
             'clf__alpha': [1e-5, 1e-4, 1e-3],
             'clf__penalty': ['l2', 'elasticnet'],
             'clf__l1_ratio': [0.1, 0.5]},
        ),
        'Ridge Classifier': (
            Pipeline([('scaler', StandardScaler()), ('clf', RidgeClassifier())]),
            {'clf__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
    }

    if _HAS_XGB:
        models['XGBoost'] = (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', XGBClassifier(eval_metric='mlogloss',
                                             use_label_encoder=False,
                                             random_state=42, n_jobs=-1))]),
            {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.05, 0.1],
             'clf__max_depth': [3, 5, 7], 'clf__subsample': [0.8, 1.0]},
        )

    if _HAS_LGB:
        models['LightGBM'] = (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))]),
            {'clf__n_estimators': [100, 200, 300], 'clf__learning_rate': [0.03, 0.05, 0.1],
             'clf__num_leaves': [31, 63, 127], 'clf__subsample': [0.8, 1.0],
             'clf__colsample_bytree': [0.8, 1.0],
             'clf__reg_alpha': [0, 0.1], 'clf__reg_lambda': [1, 5]},
        )

    if _HAS_CAT:
        models['CatBoost'] = (
            Pipeline([('scaler', StandardScaler()),
                      ('clf', CatBoostClassifier(random_seed=42, verbose=0,
                                                   thread_count=-1))]),
            {'clf__iterations': [100, 200, 300], 'clf__learning_rate': [0.03, 0.05, 0.1],
             'clf__depth': [4, 6, 8], 'clf__l2_leaf_reg': [1, 3, 5]},
        )

    # ── Train loop ───────────────────────────────────────────────────────────
    results_summary = []

    for name, (pipeline, param_grid) in models.items():
        print(f"\nTraining {name} ...")
        gs = GridSearchCV(pipeline, param_grid, cv=cv,
                          scoring='f1_weighted', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"  Best params : {gs.best_params_}")
        print(f"  Best CV F1  : {gs.best_score_:.4f}")

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        joblib.dump(gs.best_estimator_,
                    os.path.join(OUT_DIR, f'acne_{safe_name}_model.pkl'))

        y_pred   = gs.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        prec     = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec      = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1       = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_cls   = f1_score(y_test, y_pred, average=None, zero_division=0)
        prec_cls = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_cls  = recall_score(y_test, y_pred, average=None, zero_division=0)
        cm       = confusion_matrix(y_test, y_pred)

        results_summary.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'CV_F1': gs.best_score_,
            **{f'P_{c}': prec_cls[i] for i, c in enumerate(CLASSES)},
            **{f'R_{c}': rec_cls[i]  for i, c in enumerate(CLASSES)},
            **{f'F1_{c}': f1_cls[i]  for i, c in enumerate(CLASSES)},
        })

        print(f"\n--- {name} ---")
        print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f'Confusion Matrix — {name}\n(65/35 split, balanced test)')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'cm_{safe_name}.png'))
        plt.close()

    # ── Voting Ensemble (top-3 by CV F1) ────────────────────────────────────
    print("\n\nBuilding Voting Ensemble (top-3 by CV F1)...")
    sorted_res  = sorted(results_summary, key=lambda r: r['CV_F1'], reverse=True)
    top3_names  = [r['Model'] for r in sorted_res[:3]]
    print(f"  Top-3: {top3_names}")

    top3_est = []
    for mn in top3_names:
        safe = mn.replace(' ', '_').replace('(', '').replace(')', '').lower()
        top3_est.append((safe, joblib.load(os.path.join(OUT_DIR, f'acne_{safe}_model.pkl'))))

    voting_clf = VotingClassifier(estimators=top3_est, voting='soft', n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    joblib.dump(voting_clf, os.path.join(OUT_DIR, 'acne_voting_ensemble_model.pkl'))

    y_pred_v = voting_clf.predict(X_test)
    _append_result(results_summary, 'Voting Ensemble (top-3)',
                   y_test, y_pred_v, cv_f1=float('nan'))

    # ── Stacking Ensemble (top-5 + LR meta) ─────────────────────────────────
    print("\n\nBuilding Stacking Ensemble (top-5 + LR meta)...")
    top5_names = [r['Model'] for r in sorted_res[:5]]
    print(f"  Top-5: {top5_names}")

    stacking_est = []
    for mn in top5_names:
        safe = mn.replace(' ', '_').replace('(', '').replace(')', '').lower()
        stacking_est.append((safe, joblib.load(
            os.path.join(OUT_DIR, f'acne_{safe}_model.pkl'))))

    stacking_clf = StackingClassifier(
        estimators=stacking_est,
        final_estimator=LogisticRegression(max_iter=2000, solver='lbfgs',
                                           multi_class='multinomial',
                                           random_state=42, C=1.0),
        stack_method='auto', cv=5, n_jobs=-1, passthrough=False,
    )
    stacking_clf.fit(X_train, y_train)
    joblib.dump(stacking_clf, os.path.join(OUT_DIR, 'acne_stacking_ensemble_model.pkl'))

    y_pred_s = stacking_clf.predict(X_test)
    _append_result(results_summary, 'Stacking Ensemble (top-5)',
                   y_test, y_pred_s, cv_f1=float('nan'))

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(results_summary)
    _save_outputs(results_summary)

    np.save(os.path.join(OUT_DIR, 'test_data_features.npy'),
            {'X_test': X_test, 'y_test': y_test})
    print("\nAll done. Models saved to models_6535/")


# ============================================================================
# Helpers
# ============================================================================

def _append_result(results_summary, name, y_test, y_pred, cv_f1):
    f1_cls   = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec_cls = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_cls  = recall_score(y_test, y_pred, average=None, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)

    results_summary.append({
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'CV_F1':     cv_f1,
        **{f'P_{c}': prec_cls[i] for i, c in enumerate(CLASSES)},
        **{f'R_{c}': rec_cls[i]  for i, c in enumerate(CLASSES)},
        **{f'F1_{c}': f1_cls[i]  for i, c in enumerate(CLASSES)},
    })

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    safe = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix — {name}\n(65/35 split, balanced test)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f'cm_{safe}.png'))
    plt.close()


def _print_summary(results_summary):
    summary_df = (pd.DataFrame(results_summary)
                  .sort_values('F1', ascending=False)
                  .reset_index(drop=True))
    summary_df['Rank'] = summary_df.index + 1

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'CV_F1']
    col_w, num_w = 28, 9
    sep = ('+' + '-' * (col_w + 2)
           + ('+' + '-' * (num_w + 2)) * len(metrics)
           + '+' + '-' * 6 + '+')

    print("\n\n" + "=" * len(sep))
    print(" MODEL COMPARISON SUMMARY — 65/35 BALANCED TEST ".center(len(sep), "="))
    print("=" * len(sep))
    print(sep)
    print(f"| {'Model':<{col_w}} "
          + ''.join(f"| {m:^{num_w}} " for m in metrics)
          + f"| {'Rank':^4} |")
    print(sep)
    for _, row in summary_df.iterrows():
        cv_val = f"{row['CV_F1']:.4f}" if not pd.isna(row['CV_F1']) else "  —   "
        print(f"| {row['Model']:<{col_w}} "
              f"| {row['Accuracy']:^{num_w}.4f} "
              f"| {row['Precision']:^{num_w}.4f} "
              f"| {row['Recall']:^{num_w}.4f} "
              f"| {row['F1']:^{num_w}.4f} "
              f"| {cv_val:^{num_w}} "
              f"| {int(row['Rank']):^4} |")
    print(sep)

    # Best per metric
    best = {m: summary_df.loc[summary_df[m].idxmax(), 'Model'] for m in metrics}
    print("\n┌──────────────────┬──────────────────────────────────┐")
    print("│ Metric           │ Best Model                       │")
    print("├──────────────────┼──────────────────────────────────┤")
    labels = {'Accuracy': 'Accuracy ', 'Precision': 'Precision',
              'Recall': 'Recall   ', 'F1': 'F1-Score ', 'CV_F1': 'CV F1    '}
    for m, lbl in labels.items():
        score = summary_df.loc[summary_df['Model'] == best[m], m].values[0]
        print(f"│ {lbl:<16} │ {best[m]:<24} ({score:.4f}) │")
    print("└──────────────────┴──────────────────────────────────┘")

    # Per-class ranking (acne2 & acne3)
    for cls_name in CLASSES[1:]:
        f1_col, p_col, r_col = f'F1_{cls_name}', f'P_{cls_name}', f'R_{cls_name}'
        if f1_col not in summary_df.columns:
            continue
        cls_df = (summary_df[['Model', p_col, r_col, f1_col]]
                  .sort_values(f1_col, ascending=False)
                  .reset_index(drop=True))
        cls_df['ClsRank'] = cls_df.index + 1
        sep2 = ('+' + '-' * 30 + ('+' + '-' * 11) * 3 + '+' + '-' * 8 + '+')
        print(f"\n{'=' * len(sep2)}")
        print(f" PER-CLASS RANKING › {cls_name} ".center(len(sep2), '='))
        print('=' * len(sep2))
        print(sep2)
        print(f"| {'Model':<28} | {'Precision':^9} | {'Recall':^9} | {'F1-Score':^9} | {'ClsRank':^6} |")
        print(sep2)
        for _, row in cls_df.iterrows():
            tag = ' ★' if row['ClsRank'] == 1 else '  '
            print(f"| {row['Model']:<28} "
                  f"| {row[p_col]:^9.4f} "
                  f"| {row[r_col]:^9.4f} "
                  f"| {row[f1_col]:^9.4f} "
                  f"| {int(row['ClsRank']):^4}{tag} |")
        print(sep2)

    champion = summary_df.iloc[0]
    print(f"\n★  OVERALL BEST (65/35 balanced): {champion['Model']}")
    print(f"   Accuracy={champion['Accuracy']:.4f}  F1={champion['F1']:.4f}")

    summary_df.to_csv(os.path.join(EVAL_DIR, 'model_comparison.csv'), index=False)
    print(f"\nFull results → {EVAL_DIR}/model_comparison.csv")

    # Save text summary
    summary_path = os.path.join(OUT_DIR, 'Evaluation_Summary_6535.txt')
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    _print_summary_to_buf(summary_df, metrics, labels, best)
    sys.stdout = old_stdout
    with open(summary_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"Text summary   → {summary_path}")


def _print_summary_to_buf(summary_df, metrics, labels, best):
    """Re-print the summary table (called with stdout redirected to file)."""
    col_w, num_w = 28, 9
    sep = ('+' + '-' * (col_w + 2)
           + ('+' + '-' * (num_w + 2)) * len(metrics)
           + '+' + '-' * 6 + '+')
    print("=" * len(sep))
    print(" MODEL COMPARISON SUMMARY — 65/35 BALANCED TEST ".center(len(sep), "="))
    print("=" * len(sep))
    print(sep)
    print(f"| {'Model':<{col_w}} "
          + ''.join(f"| {m:^{num_w}} " for m in metrics)
          + f"| {'Rank':^4} |")
    print(sep)
    for _, row in summary_df.iterrows():
        cv_val = f"{row['CV_F1']:.4f}" if not pd.isna(row['CV_F1']) else "  —   "
        print(f"| {row['Model']:<{col_w}} "
              f"| {row['Accuracy']:^{num_w}.4f} "
              f"| {row['Precision']:^{num_w}.4f} "
              f"| {row['Recall']:^{num_w}.4f} "
              f"| {row['F1']:^{num_w}.4f} "
              f"| {cv_val:^{num_w}} "
              f"| {int(row['Rank']):^4} |")
    print(sep)
    for m, lbl in labels.items():
        score = summary_df.loc[summary_df['Model'] == best[m], m].values[0]
        print(f"  Best {lbl}: {best[m]} ({score:.4f})")
    champion = summary_df.iloc[0]
    print(f"\n★  OVERALL BEST: {champion['Model']}")
    print(f"   Accuracy={champion['Accuracy']:.4f}  F1={champion['F1']:.4f}")


def _save_outputs(results_summary):
    summary_df = (pd.DataFrame(results_summary)
                  .sort_values('F1', ascending=False)
                  .reset_index(drop=True))

    metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    plot_df = summary_df.sort_values('F1', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))
    y = np.arange(len(plot_df))
    bar_h = 0.18
    for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
        offset = (i - 1.5) * bar_h
        bars = ax.barh(y + offset, plot_df[metric], bar_h,
                       label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, plot_df[metric]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Model'], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Score')
    ax.set_title('Model Comparison — 65/35 Split, Balanced Test\n(sorted by F1)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"Comparison chart → {EVAL_DIR}/model_comparison.png")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    import sys

    # Allow running just the data split step independently:
    #   python scripts/train_6535.py --split-only
    if '--split-only' in sys.argv:
        prepare_split()
    else:
        # Full pipeline: split (if needed) + train + evaluate
        if not os.path.exists(os.path.join(SPLIT_DIR, 'train')):
            prepare_split()
        train_and_evaluate()
