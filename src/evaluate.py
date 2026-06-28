import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.config import MODELS_DIR, CLASSES

def evaluate():
    model_path = os.path.join(MODELS_DIR, 'acne_rf_model.pkl')
    test_data_path = os.path.join(MODELS_DIR, 'test_data.npy')
    
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return
        
    model = joblib.load(model_path)
    test_data = np.load(test_data_path, allow_pickle=True).item()
    X_test, y_test = test_data['X_test'], test_data['y_test']
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    # Feature Importance
    importances = model.feature_importances_
    feature_names = ['lesion_count', 'total_area', 'mean_a', 'std_a'] + [f'lbp_{i}' for i in range(len(importances)-4)]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as feature_importance.png")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Acne Severity')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_real.png')
    print("Confusion matrix saved as confusion_matrix_real.png")

if __name__ == "__main__":
    evaluate()
