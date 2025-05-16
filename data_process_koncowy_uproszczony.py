from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from imblearn.over_sampling import SMOTE
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt
from scipy.stats import entropy
import pywt
import joblib

RANDOM_STATE = 42  # Ziarno losowości

# ------------------- KONFIGURACJA -------------------
@dataclass
class WindowConfig:
    duration: int = 120
    overlap: float = 0.05
    min_valid_ratio: float = 0.8

class PathConfig:
    def __init__(self, base_path: str, processing_version: str = "v1"):
        self.BASE_DIR = Path(base_path)
        self.DATA_DIR = self.BASE_DIR / "data"
        self.PROCESSED_DIR = self.BASE_DIR / f"processed_{processing_version}"
        self.LOG_DIR = self.PROCESSED_DIR / "logs"
        self.RAW_SUBDIRS = ['train', 'val', 'test']
        
        self._validate_structure()
        self._create_dirs()

    def _validate_structure(self):
        for subdir in self.RAW_SUBDIRS:
            if not (self.DATA_DIR / subdir).exists():
                raise FileNotFoundError(f"Brak katalogu: {self.DATA_DIR/subdir}")

    def _create_dirs(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- LOGOWANIE -------------------
def setup_logging(log_dir: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_dir / 'processing.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ------------------- PRZETWARZANIE SYGNAŁÓW -------------------
def preprocess_signals(hr: np.ndarray, spo2: np.ndarray) -> tuple:
    return (
        medfilt(np.clip(hr, 40, 140), 301),
        medfilt(np.clip(spo2, 50, 100), 151)
    )

def extract_features(hr: np.ndarray, spo2: np.ndarray, target: int) -> Dict:
    hr_features = {
        'hr_mean': np.nanmean(hr),
        'hr_std': np.nanstd(hr),
        'hr_rmssd': np.sqrt(np.nanmean(np.square(np.diff(hr)))),
        'target': target
    }

    try:
        peaks, _ = find_peaks(hr, distance=50, prominence=1)
        hr_features['hr_peaks_count'] = len(peaks)
    except:
        hr_features['hr_peaks_count'] = 0

    try:
        coeffs = pywt.wavedec(hr, 'db4', level=3)
        hr_features.update({f'hr_wavelet_{i}': np.std(c) for i, c in enumerate(coeffs)})
    except:
        hr_features.update({f'hr_wavelet_{i}': 0 for i in range(4)})

    spo2_features = {
        'spo2_min': np.nanmin(spo2),
        'spo2_entropy': entropy(np.histogram(spo2, bins=20)[0]),
        'spo2_desat_count': np.sum(np.diff(spo2) < -4)
    }

    return {**hr_features, **spo2_features}

# ------------------- PRZETWARZANIE PLIKU -------------------
def process_file(file_path: Path, window_cfg: WindowConfig, path_cfg: PathConfig) -> List[Dict]:
    try:
        df = pd.read_csv(file_path, usecols=["HR", "SAO2", "Obstructive_Apnea", "Central_Apnea", "Multiple_Events", "Hypopnea"])
        fs = 100
        window_size = int(window_cfg.duration * fs)
        step = int(window_size * (1 - window_cfg.overlap))

        all_features = []
        for i in range(0, len(df) - window_size + 1, step):
            window = df.iloc[i:i+window_size].copy()

            event_cols = ['Obstructive_Apnea', 'Central_Apnea', 'Multiple_Events', 'Hypopnea']
            existing_cols = [col for col in event_cols if col in window.columns]
            window.loc[:, existing_cols] = window[existing_cols].fillna(0)

            if window.isna().mean().mean() > (1 - window_cfg.min_valid_ratio):
                continue

            target = int(window[existing_cols].sum().sum() > 0)

            hr, spo2 = preprocess_signals(window['HR'].values, window['SAO2'].values)
            features = extract_features(hr, spo2, target)
            all_features.append(features)


        return all_features

    except Exception as e:
        logging.error(f"Błąd przetwarzania {file_path}: {str(e)}", exc_info=True)
        return []

# ------------------- ZAPIS DO PKL -------------------
def save_processed_data_from_memory(results_by_subset: Dict[str, List[Dict]], path_cfg: PathConfig):
    splits = {}
    for subset, feature_dicts in results_by_subset.items():
        df = pd.DataFrame(feature_dicts)
        if df.empty:
            continue
        X = df.drop('target', axis=1)
        y = df['target']
                # Usuwanie kolumn z NaN
        X_clean = X.dropna(axis=1)
        y_clean = y[X_clean.index]
        splits[f'X_{subset}'] = X_clean
        splits[f'y_{subset}'] = y_clean

    # Balansowanie SMOTE tylko dla danych treningowych
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        splits['X_train'], splits['y_train'] = smote.fit_resample(
            splits['X_train'], 
            splits['y_train']
        )
    except Exception as e:
        logging.error(f"Błąd balansowania SMOTE: {str(e)}")

    output_path = path_cfg.PROCESSED_DIR / "processed_data.pkl"
    joblib.dump(splits, output_path)
    logging.info(f"Zapisano dane do {output_path}")

# ------------------- PRZETWARZANIE RÓWNOLEGŁE -------------------
def parallel_processing(path_cfg: PathConfig, window_cfg: WindowConfig) -> Dict[str, List[Dict]]:
    file_list = []
    for subset in path_cfg.RAW_SUBDIRS:
        file_list.extend((path_cfg.DATA_DIR / subset).glob("*.csv"))

    results_by_subset = {'train': [], 'val': [], 'test': []}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_map = {executor.submit(process_file, file, window_cfg, path_cfg): file for file in file_list}

        with tqdm(total=len(futures_map), desc="Przetwarzanie plików") as pbar:
            for future in as_completed(futures_map):
                file = futures_map[future]
                subset = file.parent.name
                try:
                    result = future.result()
                    if result:
                        results_by_subset[subset].extend(result)
                except Exception as e:
                    logging.error(f"Błąd w wątku dla pliku {file}: {e}", exc_info=True)
                pbar.update(1)

    return results_by_subset

# ------------------- MAIN -------------------
if __name__ == "__main__":
    paths = PathConfig(r"D:\Apnea_Detection_Project", "v1")
    window_cfg = WindowConfig()

    setup_logging(paths.LOG_DIR)
    logging.info("Rozpoczęcie przetwarzania")

    try:
        results_by_subset = parallel_processing(paths, window_cfg)
        save_processed_data_from_memory(results_by_subset, paths)
    except Exception as e:
        logging.critical(f"Krytyczny błąd: {str(e)}", exc_info=True)
    finally:
        logging.info("Przetwarzanie zakończone")
