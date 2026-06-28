import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODELS_DIR, CLASSES, BASE_DIR
from src.features import FeatureExtractor

# Evaluation outputs (charts, CSV, confusion matrices) go here.
# pkl and npy files stay in MODELS_DIR so other scripts can load them.
EVAL_DIR = os.path.join(MODELS_DIR, 'evaluation_test')
os.makedirs(EVAL_DIR, exist_ok=True)
from src.preprocessing import FaceProcessor

# Optional boosting libraries — used if installed
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


def get_df_from_dir(directory):
    data = []
    for label, cls_name in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls_name)
        if not os.path.exists(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.endswith('.jpg'):
                data.append({'image_path': os.path.join(cls_path, f), 'label': label})
    return pd.DataFrame(data)


def extract_features_for_df(df, processor, extractor):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['label']

        img_resized, skin_mask = processor.preprocess_image(img_path)
        if img_resized is not None and skin_mask is not None:
            features = extractor.extract(img_resized, skin_mask)
        else:
            print(f"Failed to process {img_path}")
            features = np.zeros(extractor.FEATURE_DIM)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_and_evaluate():
    train_dir = os.path.join(BASE_DIR, 'data_split', 'train')
    test_dir = os.path.join(BASE_DIR, 'data_split', 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Error: data_split directories not found. Run data_prep.py first.")
        return

    train_df = get_df_from_dir(train_dir)
    test_df = get_df_from_dir(test_dir)

    processor = FaceProcessor()
    extractor = FeatureExtractor()

    print(f"Extracting features for Training Set ({len(train_df)} images)...")
    X_train, y_train = extract_features_for_df(train_df, processor, extractor)

    print(f"Extracting features for Testing Set ({len(test_df)} images)...")
    X_test, y_test = extract_features_for_df(test_df, processor, extractor)

    print(f"Feature vector size: {X_train.shape[1]}")

    # 5-fold stratified CV — fitted only on training data, never touches X_test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Each entry: (pipeline, param_grid)
    # Pipeline bundles StandardScaler + classifier so the scaler is fit on each
    # training fold inside GridSearchCV (no leakage) and on the full X_train
    # when we call gs.best_estimator_.predict(X_test).
    # ── Model zoo ────────────────────────────────────────────────────────────
    # Models are chosen for their strengths on 50-dim texture/structural
    # feature vectors extracted from facial skin images.
    #
    # Core: SVM kernels excel at high-dim margin separation (skin texture).
    #       Tree ensembles capture non-linear lesion interactions.
    #       MLP learns composite pattern representations.
    # Extras: KNN (local pattern matching), Logistic Regression (strong linear
    #         baseline), HistGBM (fast native gradient boosting like LightGBM).
    # Conditionals: XGBoost / LightGBM added when installed — both consistently
    #               rank top-2 on skin-condition tabular benchmarks.
    models = {
        # ── SVM family ───────────────────────────────────────────────────────
        'SVM (RBF)': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', probability=True, random_state=42)),
            ]),
            {
                'clf__C':     [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 0.001, 0.01, 0.1],
            },
        ),
        'SVM (Polynomial)': (
            # Polynomial kernel captures multiplicative feature interactions
            # (e.g., lesion count × redness) — useful for skin texture grades.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='poly', probability=True, random_state=42)),
            ]),
            {
                'clf__C':      [0.1, 1, 10],
                'clf__degree': [2, 3],
                'clf__gamma':  ['scale', 0.01],
            },
        ),
        'SVM (Linear)': (
            # Fast linear SVM — strong when features are already well-separated.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='linear', probability=True, random_state=42)),
            ]),
            {
                'clf__C': [0.01, 0.1, 1, 10],
            },
        ),

        # ── Logistic Regression ───────────────────────────────────────────────
        'Logistic Regression': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    max_iter=2000, solver='lbfgs',
                    multi_class='multinomial', random_state=42)),
            ]),
            {
                'clf__C': [0.01, 0.1, 1, 10, 100],
            },
        ),

        # ── K-Nearest Neighbours ─────────────────────────────────────────────
        'KNN': (
            # Local structure in LBP/GLCM space directly reflects texture
            # similarity between skin patches → KNN is a natural fit.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(metric='minkowski')),
            ]),
            {
                'clf__n_neighbors': [3, 5, 7, 11],
                'clf__weights':     ['uniform', 'distance'],
                'clf__p':           [1, 2],   # Manhattan vs Euclidean
            },
        ),

        # ── Tree ensembles ───────────────────────────────────────────────────
        'Random Forest': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]),
            {
                'clf__n_estimators':      [100, 200, 300],
                'clf__max_depth':         [None, 15, 30],
                'clf__min_samples_split': [2, 5],
                'clf__max_features':      ['sqrt', 'log2'],
            },
        ),
        'Extra Trees': (
            # More randomised splits than RF → lower variance on small datasets;
            # fast and robust for texture features.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', ExtraTreesClassifier(random_state=42, n_jobs=-1)),
            ]),
            {
                'clf__n_estimators':      [100, 200, 300],
                'clf__max_depth':         [None, 15, 30],
                'clf__min_samples_split': [2, 5],
            },
        ),
        'AdaBoost': (
            # Iteratively focuses on hard-to-classify lesion boundaries;
            # performs well when class distributions differ in severity.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', AdaBoostClassifier(
                    algorithm='SAMME', random_state=42)),
            ]),
            {
                'clf__n_estimators':  [50, 100, 200],
                'clf__learning_rate': [0.5, 1.0, 1.5],
            },
        ),
        'Gradient Boosting': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(random_state=42)),
            ]),
            {
                'clf__n_estimators':  [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth':     [3, 5],
                'clf__subsample':     [0.8, 1.0],
            },
        ),
        'HistGradientBoosting': (
            # Histogram-based GBM (sklearn's LightGBM equivalent).
            # Natively handles varying feature scales; very fast.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', HistGradientBoostingClassifier(random_state=42)),
            ]),
            {
                'clf__max_iter':      [100, 200],
                'clf__learning_rate': [0.05, 0.1, 0.2],
                'clf__max_depth':     [None, 5, 10],
                'clf__min_samples_leaf': [10, 20],
            },
        ),

        # ── Neural network ───────────────────────────────────────────────────
        'MLP': (
            # Multi-layer perceptron learns composite skin-pattern
            # representations from LBP + GLCM + redness features.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(
                    max_iter=500, early_stopping=True,
                    validation_fraction=0.1, random_state=42)),
            ]),
            {
                'clf__hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128), (256, 128, 64)],
                'clf__activation':         ['relu', 'tanh'],
                'clf__alpha':              [1e-4, 1e-3, 1e-2],
                'clf__learning_rate_init': [0.001, 0.01],
            },
        ),

        # ── Discriminant Analysis ────────────────────────────────────────────
        'LDA': (
            # Linear Discriminant Analysis — maximises between-class variance
            # of projected features; very strong on handcrafted texture vectors.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LinearDiscriminantAnalysis()),
            ]),
            {
                'clf__solver':    ['svd', 'lsqr'],
                'clf__shrinkage': [None, 'auto', 0.1, 0.5],
            },
        ),
        'QDA': (
            # Quadratic Discriminant Analysis — models each class with its own
            # covariance; captures non-linear skin-grade boundaries.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', QuadraticDiscriminantAnalysis()),
            ]),
            {
                'clf__reg_param': [0.0, 0.1, 0.3, 0.5],
            },
        ),

        # ── Naive Bayes ──────────────────────────────────────────────────────
        'Gaussian NB': (
            # Probabilistic baseline; fast and interpretable.
            # Surprisingly competitive when LBP/GLCM features are near-Gaussian.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB()),
            ]),
            {
                'clf__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3],
            },
        ),

        # ── Calibrated Linear SVM ────────────────────────────────────────────
        'Calibrated Linear SVM': (
            # LinearSVC wrapped with Platt calibration — much faster than
            # SVC(kernel='linear') on large feature sets while giving
            # calibrated probabilities required for soft voting.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', CalibratedClassifierCV(
                    LinearSVC(max_iter=5000, random_state=42), cv=3)),
            ]),
            {
                'clf__estimator__C': [0.01, 0.1, 1, 10],
            },
        ),

        # ── Bagging SVM ──────────────────────────────────────────────────────
        'Bagging SVM': (
            # Bootstrap-aggregated SVMs — reduces SVM variance on small
            # acne datasets by training on random subsets with replacement.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', BaggingClassifier(
                    estimator=SVC(kernel='rbf', probability=True, random_state=42),
                    random_state=42, n_jobs=-1)),
            ]),
            {
                'clf__n_estimators':   [10, 20, 30],
                'clf__max_samples':    [0.7, 0.85, 1.0],
                'clf__max_features':   [0.7, 1.0],
            },
        ),

        # ── SGD Classifier ───────────────────────────────────────────────────
        'SGD Classifier': (
            # Stochastic Gradient Descent with SVM/log loss — efficient on
            # high-dimensional feature vectors; strong regularisation options.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1)),
            ]),
            {
                'clf__loss':        ['hinge', 'log_loss', 'modified_huber'],
                'clf__alpha':       [1e-5, 1e-4, 1e-3],
                'clf__penalty':     ['l2', 'elasticnet'],
                'clf__l1_ratio':    [0.1, 0.5],
            },
        ),

        # ── Ridge Classifier ─────────────────────────────────────────────────
        'Ridge Classifier': (
            # Ridge regression as classifier — closed-form solution, very fast,
            # strong L2 regularisation prevents overfitting on small datasets.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RidgeClassifier()),
            ]),
            {
                'clf__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
    }

    # ── Optional: XGBoost ────────────────────────────────────────────────────
    if _HAS_XGB:
        models['XGBoost'] = (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(
                    eval_metric='mlogloss', use_label_encoder=False,
                    random_state=42, n_jobs=-1)),
            ]),
            {
                'clf__n_estimators':  [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth':     [3, 5, 7],
                'clf__subsample':     [0.8, 1.0],
            },
        )
    else:
        print("XGBoost not installed — skipping. (pip install xgboost)")

    # ── Optional: LightGBM ───────────────────────────────────────────────────
    if _HAS_LGB:
        models['LightGBM'] = (
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)),
            ]),
            {
                'clf__n_estimators':  [100, 200, 300],
                'clf__learning_rate': [0.03, 0.05, 0.1],
                'clf__num_leaves':    [31, 63, 127],
                'clf__subsample':     [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
                'clf__reg_alpha':     [0, 0.1],
                'clf__reg_lambda':    [1, 5],
            },
        )
    else:
        print("LightGBM not installed — skipping. (pip install lightgbm)")

    # ── Optional: CatBoost ───────────────────────────────────────────────────
    if _HAS_CAT:
        models['CatBoost'] = (
            # CatBoost uses ordered boosting — robust to overfitting on small
            # medical image datasets; often top-1 on tabular benchmarks.
            Pipeline([
                ('scaler', StandardScaler()),
                ('clf', CatBoostClassifier(
                    random_seed=42, verbose=0, thread_count=-1)),
            ]),
            {
                'clf__iterations':    [100, 200, 300],
                'clf__learning_rate': [0.03, 0.05, 0.1],
                'clf__depth':         [4, 6, 8],
                'clf__l2_leaf_reg':   [1, 3, 5],
            },
        )
    else:
        print("CatBoost not installed — skipping. (pip install catboost)")

    # ── Train, evaluate, save ────────────────────────────────────────────────
    results_summary = []   # collect per-model metrics for final comparison

    for name, (pipeline, param_grid) in models.items():
        print(f"\nTraining {name} with GridSearchCV (5-fold stratified CV)...")
        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)   # scaler fit inside each fold; never sees X_test

        print(f"  Best params : {gs.best_params_}")
        print(f"  Best CV F1  : {gs.best_score_:.4f}")

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        joblib.dump(gs.best_estimator_, os.path.join(MODELS_DIR, f'acne_{safe_name}_model.pkl'))

        y_pred = gs.predict(X_test)

        acc       = accuracy_score(y_test, y_pred)
        prec      = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec       = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_cls    = f1_score(y_test, y_pred, average=None, zero_division=0)
        prec_cls  = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_cls   = recall_score(y_test, y_pred, average=None, zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)

        results_summary.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'CV_F1': gs.best_score_,
            **{f'P_{c}': prec_cls[i] for i, c in enumerate(CLASSES)},
            **{f'R_{c}': rec_cls[i]  for i, c in enumerate(CLASSES)},
            **{f'F1_{c}': f1_cls[i]  for i, c in enumerate(CLASSES)},
        })

        print(f"\n--- {name} Results ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nPer-class Report:")
        print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'cm_{safe_name}.png'))
        plt.close()
        print(f"Confusion Matrix saved to models/evaluation_test/cm_{safe_name}.png")

    # ── Voting Ensemble (top-3 by CV F1) ─────────────────────────────────────
    # Combine the three strongest individual models for a final ensemble that
    # is typically more stable than any single classifier alone.
    print("\n\nBuilding Soft-Voting Ensemble from top-3 CV models...")
    sorted_results = sorted(results_summary, key=lambda r: r['CV_F1'], reverse=True)
    top3_names = [r['Model'] for r in sorted_results[:3]]
    print(f"  Top-3 selected: {top3_names}")

    top3_estimators = []
    for model_name in top3_names:
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        pipeline_path = os.path.join(MODELS_DIR, f'acne_{safe}_model.pkl')
        top3_estimators.append((safe, joblib.load(pipeline_path)))

    voting_clf = VotingClassifier(estimators=top3_estimators, voting='soft', n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    joblib.dump(voting_clf, os.path.join(MODELS_DIR, 'acne_voting_ensemble_model.pkl'))

    y_pred_v      = voting_clf.predict(X_test)
    acc_v         = accuracy_score(y_test, y_pred_v)
    prec_v        = precision_score(y_test, y_pred_v, average='weighted', zero_division=0)
    rec_v         = recall_score(y_test, y_pred_v, average='weighted', zero_division=0)
    f1_v          = f1_score(y_test, y_pred_v, average='weighted', zero_division=0)
    f1_cls_v      = f1_score(y_test, y_pred_v, average=None, zero_division=0)
    prec_cls_v    = precision_score(y_test, y_pred_v, average=None, zero_division=0)
    rec_cls_v     = recall_score(y_test, y_pred_v, average=None, zero_division=0)
    cm_v          = confusion_matrix(y_test, y_pred_v)

    results_summary.append({
        'Model': 'Voting Ensemble (top-3)', 'Accuracy': acc_v, 'Precision': prec_v,
        'Recall': rec_v, 'F1': f1_v, 'CV_F1': float('nan'),
        **{f'P_{c}': prec_cls_v[i] for i, c in enumerate(CLASSES)},
        **{f'R_{c}': rec_cls_v[i]  for i, c in enumerate(CLASSES)},
        **{f'F1_{c}': f1_cls_v[i]  for i, c in enumerate(CLASSES)},
    })

    print(f"\n--- Voting Ensemble Results ---")
    print(f"Accuracy : {acc_v:.4f}")
    print(f"Precision: {prec_v:.4f}")
    print(f"Recall   : {rec_v:.4f}")
    print(f"F1-Score : {f1_v:.4f}")
    print(classification_report(y_test, y_pred_v, target_names=CLASSES, zero_division=0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_v, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Voting Ensemble (top-3)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'cm_voting_ensemble.png'))
    plt.close()

    # ── Stacking Ensemble (top-5 base + LR meta-learner) ─────────────────────
    # StackingClassifier generates out-of-fold predictions from each base model
    # and trains a Logistic Regression meta-learner on those predictions.
    # This typically outperforms both VotingClassifier and any single model.
    print("\n\nBuilding Stacking Ensemble from top-5 CV models...")
    top5_names = [r['Model'] for r in sorted_results[:5]]
    print(f"  Top-5 base estimators: {top5_names}")

    # Load saved best-estimator pipelines as base learners.
    # StackingClassifier will refit them internally via cross-val.
    stacking_estimators = []
    for model_name in top5_names:
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        pipeline_path = os.path.join(MODELS_DIR, f'acne_{safe}_model.pkl')
        stacking_estimators.append((safe, joblib.load(pipeline_path)))

    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(
            max_iter=2000, solver='lbfgs',
            multi_class='multinomial', random_state=42, C=1.0),
        stack_method='auto',   # predict_proba → decision_function → predict
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    stacking_clf.fit(X_train, y_train)
    joblib.dump(stacking_clf, os.path.join(MODELS_DIR, 'acne_stacking_ensemble_model.pkl'))

    y_pred_s      = stacking_clf.predict(X_test)
    acc_s         = accuracy_score(y_test, y_pred_s)
    prec_s        = precision_score(y_test, y_pred_s, average='weighted', zero_division=0)
    rec_s         = recall_score(y_test, y_pred_s, average='weighted', zero_division=0)
    f1_s          = f1_score(y_test, y_pred_s, average='weighted', zero_division=0)
    f1_cls_s      = f1_score(y_test, y_pred_s, average=None, zero_division=0)
    prec_cls_s    = precision_score(y_test, y_pred_s, average=None, zero_division=0)
    rec_cls_s     = recall_score(y_test, y_pred_s, average=None, zero_division=0)
    cm_s          = confusion_matrix(y_test, y_pred_s)

    results_summary.append({
        'Model': 'Stacking Ensemble (top-5)', 'Accuracy': acc_s, 'Precision': prec_s,
        'Recall': rec_s, 'F1': f1_s, 'CV_F1': float('nan'),
        **{f'P_{c}': prec_cls_s[i] for i, c in enumerate(CLASSES)},
        **{f'R_{c}': rec_cls_s[i]  for i, c in enumerate(CLASSES)},
        **{f'F1_{c}': f1_cls_s[i]  for i, c in enumerate(CLASSES)},
    })

    print(f"\n--- Stacking Ensemble Results ---")
    print(f"Accuracy : {acc_s:.4f}")
    print(f"Precision: {prec_s:.4f}")
    print(f"Recall   : {rec_s:.4f}")
    print(f"F1-Score : {f1_s:.4f}")
    print(classification_report(y_test, y_pred_s, target_names=CLASSES, zero_division=0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Stacking Ensemble (top-5)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'cm_stacking_ensemble.png'))
    plt.close()

    # ── Summary table ────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results_summary).sort_values('F1', ascending=False).reset_index(drop=True)
    summary_df['Rank'] = summary_df.index + 1

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'CV_F1']
    best_per_metric = {
        m: summary_df.loc[summary_df[m].idxmax(), 'Model']
        for m in metrics
    }

    # ── Console summary table ─────────────────────────────────────────────────
    col_w = 28
    num_w = 9
    sep = '+' + '-' * (col_w + 2) + ('+' + '-' * (num_w + 2)) * len(metrics) + '+' + '-' * 6 + '+'
    header = (
        f"| {'Model':<{col_w}} "
        + ''.join(f"| {m:^{num_w}} " for m in metrics)
        + f"| {'Rank':^4} |"
    )

    print("\n\n" + "=" * len(sep))
    print(" MODEL COMPARISON SUMMARY ".center(len(sep), "="))
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)
    for _, row in summary_df.iterrows():
        cv_val = f"{row['CV_F1']:.4f}" if not pd.isna(row['CV_F1']) else "  —   "
        line = (
            f"| {row['Model']:<{col_w}} "
            f"| {row['Accuracy']:^{num_w}.4f} "
            f"| {row['Precision']:^{num_w}.4f} "
            f"| {row['Recall']:^{num_w}.4f} "
            f"| {row['F1']:^{num_w}.4f} "
            f"| {cv_val:^{num_w}} "
            f"| {int(row['Rank']):^4} |"
        )
        print(line)
    print(sep)

    # ── Best model per metric (overall) ──────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│           BEST MODEL PER METRIC (OVERALL)           │")
    print("├──────────────────┬──────────────────────────────────┤")
    print(f"│ {'Metric':<16} │ {'Best Model':<32} │")
    print("├──────────────────┼──────────────────────────────────┤")
    metric_labels = {
        'Accuracy':  'Accuracy ',
        'Precision': 'Precision',
        'Recall':    'Recall   ',
        'F1':        'F1-Score ',
        'CV_F1':     'CV F1    ',
    }
    for m, label in metric_labels.items():
        best_model = best_per_metric[m]
        best_score = summary_df.loc[summary_df['Model'] == best_model, m].values[0]
        print(f"│ {label:<16} │ {best_model:<24} ({best_score:.4f}) │")
    print("└──────────────────┴──────────────────────────────────┘")

    # ── Per-class ranking tables (acne2 & acne3) ─────────────────────────────
    cls_col_w = 28
    cls_num_w = 9

    def _print_class_ranking(cls_name):
        f1_col  = f'F1_{cls_name}'
        p_col   = f'P_{cls_name}'
        r_col   = f'R_{cls_name}'
        if f1_col not in summary_df.columns:
            return
        cls_df = summary_df[['Model', p_col, r_col, f1_col]].copy()
        cls_df = cls_df.sort_values(f1_col, ascending=False).reset_index(drop=True)
        cls_df['ClsRank'] = cls_df.index + 1

        best_f1_model = cls_df.iloc[0]['Model']
        best_p_model  = cls_df.loc[cls_df[p_col].idxmax(), 'Model']
        best_r_model  = cls_df.loc[cls_df[r_col].idxmax(), 'Model']

        cols = [p_col, r_col, f1_col]
        hdrs = ['Precision', 'Recall  ', 'F1-Score']
        sep2 = '+' + '-' * (cls_col_w + 2) + ('+' + '-' * (cls_num_w + 2)) * 3 + '+' + '-' * 8 + '+'
        print(f"\n{'=' * len(sep2)}")
        print(f" PER-CLASS RANKING  ›  {cls_name} ".center(len(sep2), '='))
        print('=' * len(sep2))
        print(sep2)
        hdr2 = (f"| {'Model':<{cls_col_w}} "
                + ''.join(f"| {h:^{cls_num_w}} " for h in hdrs)
                + f"| {'ClsRank':^6} |")
        print(hdr2)
        print(sep2)
        for _, row in cls_df.iterrows():
            tag = ' ★' if row['Model'] == best_f1_model else '  '
            print(
                f"| {row['Model']:<{cls_col_w}} "
                f"| {row[p_col]:^{cls_num_w}.4f} "
                f"| {row[r_col]:^{cls_num_w}.4f} "
                f"| {row[f1_col]:^{cls_num_w}.4f} "
                f"| {int(row['ClsRank']):^4}{tag} |"
            )
        print(sep2)
        print(f"  Best F1        → {best_f1_model}")
        print(f"  Best Precision → {best_p_model}")
        print(f"  Best Recall    → {best_r_model}")

    for cls in CLASSES[1:]:   # acne2 and acne3 (skip acne1)
        _print_class_ranking(cls)

    # ── Overall champion (highest F1) ─────────────────────────────────────────
    champion = summary_df.iloc[0]
    print(f"\n★  OVERALL BEST (weighted F1): {champion['Model']}")
    print(f"   Accuracy={champion['Accuracy']:.4f}  Precision={champion['Precision']:.4f}"
          f"  Recall={champion['Recall']:.4f}  F1={champion['F1']:.4f}")

    summary_df.to_csv(os.path.join(EVAL_DIR, 'model_comparison.csv'), index=False)
    print(f"\nFull summary saved to models/evaluation_test/model_comparison.csv")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    plot_df = summary_df.sort_values('F1', ascending=True)   # ascending for readability
    metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))
    y = np.arange(len(plot_df))
    bar_h = 0.18
    for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
        offset = (i - 1.5) * bar_h
        bars = ax.barh(y + offset, plot_df[metric], bar_h, label=metric, color=color, alpha=0.85)
        # annotate value on best bar
        for bar, val in zip(bars, plot_df[metric]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Model'], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Score')
    ax.set_title('Model Comparison — Acne Severity Classification\n(sorted by F1, ascending)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)

    # Mark overall best
    best_idx = plot_df.index[plot_df['Model'] == champion['Model']].tolist()
    if best_idx:
        ax.get_yticklabels()[list(plot_df.index).index(best_idx[0])].set_color('#C44E52')
        ax.get_yticklabels()[list(plot_df.index).index(best_idx[0])].set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print("Comparison chart saved to models/evaluation_test/model_comparison.png")

    # ── Best-per-metric highlight chart ──────────────────────────────────────
    fig2, axes = plt.subplots(1, len(metrics_plot), figsize=(18, 5), sharey=False)
    fig2.suptitle('Best Model per Metric — Acne Severity Classification',
                  fontsize=12, fontweight='bold')
    for ax2, metric, color in zip(axes, metrics_plot, colors):
        sorted_m = summary_df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        bar_colors = [color if m == best_per_metric[metric] else '#cccccc'
                      for m in sorted_m['Model']]
        ax2.barh(sorted_m['Model'], sorted_m[metric], color=bar_colors, edgecolor='white')
        ax2.set_title(metric, fontsize=10, fontweight='bold')
        ax2.set_xlim(0, 1.05)
        ax2.tick_params(axis='y', labelsize=7)
        ax2.invert_yaxis()
        # annotate best
        best_val = sorted_m[metric].iloc[0]
        ax2.axvline(x=best_val, color=color, linestyle='--', linewidth=1, alpha=0.6)
        ax2.text(best_val - 0.01, -0.6, f'{best_val:.4f}',
                 color=color, fontsize=8, fontweight='bold', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'best_per_metric.png'), dpi=150)
    plt.close()
    print("Best-per-metric chart saved to models/evaluation_test/best_per_metric.png")

    # ── Per-class F1 ranking chart (acne2 & acne3) ───────────────────────────
    target_classes = CLASSES[1:]   # acne2, acne3
    cls_colors = ['#2ca02c', '#d62728']   # green=acne2, red=acne3

    fig3, axes3 = plt.subplots(1, len(target_classes),
                               figsize=(16, max(6, len(summary_df) * 0.42)),
                               sharey=False)
    if len(target_classes) == 1:
        axes3 = [axes3]

    fig3.suptitle('Per-Class F1 Ranking — Acne2 vs Acne3',
                  fontsize=13, fontweight='bold')

    for ax3, cls_name, col in zip(axes3, target_classes, cls_colors):
        f1_col = f'F1_{cls_name}'
        p_col  = f'P_{cls_name}'
        r_col  = f'R_{cls_name}'
        if f1_col not in summary_df.columns:
            continue

        cls_sorted = summary_df[['Model', p_col, r_col, f1_col]] \
            .sort_values(f1_col, ascending=True).reset_index(drop=True)

        best_f1_val   = cls_sorted[f1_col].iloc[-1]
        best_f1_model = cls_sorted['Model'].iloc[-1]

        bar_colors3 = [col if m == best_f1_model else '#cccccc'
                       for m in cls_sorted['Model']]

        y3 = np.arange(len(cls_sorted))
        bar_h3 = 0.25
        b_f1  = ax3.barh(y3 + bar_h3,  cls_sorted[f1_col], bar_h3,
                          label='F1',        color=bar_colors3, alpha=0.9)
        b_p   = ax3.barh(y3,            cls_sorted[p_col],  bar_h3,
                          label='Precision', color=col, alpha=0.45)
        b_r   = ax3.barh(y3 - bar_h3,  cls_sorted[r_col],  bar_h3,
                          label='Recall',    color=col, alpha=0.25)

        for bar, val in zip(b_f1, cls_sorted[f1_col]):
            ax3.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=6.5)

        ax3.set_yticks(y3)
        ax3.set_yticklabels(cls_sorted['Model'], fontsize=7.5)
        ax3.set_xlim(0, 1.12)
        ax3.set_xlabel('Score')
        ax3.set_title(f'{cls_name}\n(★ best F1: {best_f1_model})',
                      fontsize=9, fontweight='bold', color=col)
        ax3.axvline(x=best_f1_val, color=col, linestyle='--',
                    linewidth=1.2, alpha=0.7)
        ax3.legend(loc='lower right', fontsize=7)

        # Rank numbers on the right
        for i, (_, row) in enumerate(cls_sorted.iterrows()):
            rank = len(cls_sorted) - i
            ax3.text(1.10, y3[i], f'#{rank}', va='center',
                     fontsize=6.5, color='#444444')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'per_class_ranking.png'), dpi=150)
    plt.close()
    print("Per-class ranking chart saved to models/evaluation_test/per_class_ranking.png")

    np.save(
        os.path.join(MODELS_DIR, 'test_data_features.npy'),
        {'X_test': X_test, 'y_test': y_test},
    )
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_and_evaluate()
