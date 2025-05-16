import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import RidgeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# Ścieżki
BASE_DIR = Path(r'D:\Apnea_Detection_Project\processed_data')
PROCESSED_DIR = BASE_DIR / '3'
RANDOM_STATE = 42

# Logowanie
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_optimal_threshold(model, X, y):
    """Znajduje optymalny próg klasyfikacji maksymalizujący F1-score na danych walidacyjnych."""
    y_scores = model.decision_function(X)
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Wizualizacja krzywej precision-recall
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precyzja")
    plt.plot(thresholds, recalls[:-1], "g-", label="Czułość")
    plt.plot(thresholds, f1_scores[:-1], "k-", label="F1-score")
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label='Optymalny próg')
    plt.xlabel("Próg")
    plt.title("Krzywe Precision-Recall i F1-score")
    plt.legend()
    plt.grid()
    plt.show()
    
    logger.info(f"Optymalny próg: {optimal_threshold:.3f}, F1-score: {f1_scores[optimal_idx]:.3f}")
    return optimal_threshold

def evaluate(model, X, y, name, threshold=None):
    if threshold is not None:
        y_scores = model.decision_function(X)
        y_pred = (y_scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
    
    logger.info(f"\n=== {name} ===")
    print(classification_report(y, y_pred, zero_division=0))
    
    # Oblicz i wyświetl metryki
    f1 = f1_score(y, y_pred)
    logger.info(f"F1-score: {f1:.3f}")
    
    # Wyświetl macierz pomyłek
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Macierz pomyłek - {name}')
    plt.show()
    
    try:
        logger.info(f"AUC-ROC: {roc_auc_score(y, y_pred):.3f}")
        logger.info(f"AUC-PR: {average_precision_score(y, y_pred):.3f}")
    except Exception as e:
        logger.warning(f"Nie można policzyć AUC: {str(e)}")

def plot_feature_importance(pipeline, feature_names):
    """Wizualizacja ważności cech dla modelu Ridge"""
    if hasattr(pipeline.named_steps['clf'], 'coef_'):
        coef = pipeline.named_steps['clf'].coef_[0]
        importance = pd.Series(coef, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        importance.plot.barh()
        plt.title('Ważność cech w modelu Ridge')
        plt.xlabel('Wartość współczynnika')
        plt.tight_layout()
        plt.show()

def train_ridge():
    logger.info("Ładowanie danych...")
    data = joblib.load(PROCESSED_DIR / 'processed_data.pkl')
    
    logger.info("Trenowanie modelu...")
    # imblearn Pipeline aby SMOTE działał tylko na treningu
    pipeline = ImbPipeline([
        ('var', VarianceThreshold(0.01)),
        ('power', PowerTransformer()),
        ('select', SelectKBest(score_func=f_classif, k=30)),  # Wybierz top 20 cech
        ('scale', RobustScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),  # Balansowanie klas
        ('clf', RidgeClassifier(random_state=RANDOM_STATE))
    ])

    pipeline.fit(data['X_train'], data['y_train'])
    
    # Pobranie nazw wybranych cech
    selected_features = data['X_train'].columns[
        pipeline.named_steps['var'].get_support()
    ][pipeline.named_steps['select'].get_support()]
    
    # Wizualizacja ważności cech
    plot_feature_importance(pipeline, selected_features)

    # Znajdź optymalny próg
    optimal_threshold = find_optimal_threshold(pipeline, data['X_val'], data['y_val'])

    # Ocena na wszystkich zbiorach
    for set_name, key in [('Train', 'train'), ('Validation', 'val'), ('Test', 'test')]:
        evaluate(pipeline, data[f'X_{key}'], data[f'y_{key}'], set_name, optimal_threshold)

    # Zapisz model
    joblib.dump({
        'model': pipeline,
        'optimal_threshold': optimal_threshold,
        'selected_features': selected_features
    }, BASE_DIR / 'ridge_model_with_threshold.pkl')
    logger.info("Model zapisany.")

if __name__ == "__main__":
    train_ridge()