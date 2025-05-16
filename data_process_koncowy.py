"""
Skrypt do przetwarzania sygnałów HR i SpO2 do wykrywania bezdechów sennych

Główne funkcjonalności:
- Czyszczenie i filtracja sygnałów
- Ekstrakcja cech czasowych, częstotliwościowych i falkowych
- Wykrywanie desaturacji tlenowych
- Przetwarzanie równoległe plików
- Balansowanie danych SMOTE
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pywt
import joblib
import logging
from scipy import stats
from scipy.signal import find_peaks, welch, medfilt
from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional

# ------------------- KONFIGURACJA -------------------
# Ścieżki i parametry przetwarzania
BASE_DIR = Path(r'D:\Apnea_Detection_Project')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = BASE_DIR / 'processed_data'
SUBDIRS = ['train', 'val', 'test']

FS = 100  # Częstotliwość próbkowania [Hz]
EPOCH_DURATION = 30  # Długość epoki w sekundach
SAMPLES_PER_EPOCH = FS * EPOCH_DURATION
OVERLAP_RATIO = 0.20  # % nakładania okien
STEP = int(SAMPLES_PER_EPOCH * (1 - OVERLAP_RATIO))
RANDOM_STATE = 42  # Ziarno losowości

# Zakresy i parametry sygnałów
HR_MIN, HR_MAX = 40, 140
SAO2_MIN, SAO2_MAX = 50, 100
DESATURATION_THRESHOLD = 3  # Minimalny spadek SpO2 [%]
MIN_DESATURATION_DURATION = 10  # Minimalny czas desaturacji [s]

# Parametry analizy falkowej
WAVELET_PARAMS = {
    'hr': {'wavelet': 'db2', 'level': 2},
    'sao2': {'wavelet': 'db2', 'level': 2}
}

# ENTROPY_PARAMS = {
#     'shannon_bins': 20,
#     'sample_entropy': {
#         'm': 2,
#         'r': 0.2
#     }
# }

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------- FUNKCJE PRZETWARZANIA HR -------------------
def clean_hr_signal(hr_series: np.ndarray, fs: int = FS, log_prefix: str = "") -> np.ndarray:
    """
    Czyści i filtruje sygnał HR
    
    Args:
        hr_series: Surowy sygnał HR
        fs: Częstotliwość próbkowania
        log_prefix: Prefiks do logowania
        
    Returns:
        Oczyszczony sygnał HR
    """
    # Przycięcie wartości do zakresu fizjologicznego
    hr_clipped = np.clip(hr_series, HR_MIN, HR_MAX)
    
    # Wykrywanie gwałtownych zmian (artefakty)
    diff = np.abs(np.diff(hr_clipped, prepend=hr_clipped[0]))
    mask_spikes = diff > 5  # Próg wykrywania artefaktów
    
    # Filtracja medianowa z adaptacyjnym oknem
    window_size = 10 * fs  # x-sekundowe okno
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    hr_smoothed = medfilt(hr_clipped, kernel_size=window_size)
    
    # Zastępowanie artefaktów wartościami wygładzonymi
    cleaned_hr = np.where(mask_spikes, hr_smoothed, hr_clipped)
    
    # Logowanie statystyk
    n_corrected = np.sum(mask_spikes)
    if n_corrected > 0:
        logger.info(f"{log_prefix} | Poprawiono {n_corrected} artefaktów HR ({n_corrected/len(hr_series)*100:.2f}%)")
    
    return cleaned_hr

def calculate_hr_features(hr_signal: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """    Oblicza wszystkie cechy HR w jednej funkcji (łącznie z HRV i fluktuacjami)
    
    Args:
        hr_signal: Sygnał tętna
        fs: Częstotliwość próbkowania
        
    Returns:
        Słownik z cechami HR
    """
    features = {}
    valid_hr = hr_signal[~np.isnan(hr_signal)]
    
    if len(valid_hr) < 10:  # Minimalna liczba próbek
        return {f'hr_{k}': 0.0 for k in [
            'mean', 'std', 'median', 'range', 'skew', 'kurtosis',
            'change_rate', 'max_fluct', 'fluct_30s', 'fluct_60s', 'fluct_180s'
        ]}
    
    # Podstawowe statystyki
    features.update({
        'hr_mean': np.mean(valid_hr),
        'hr_std': np.std(valid_hr, ddof=1),
        'hr_median': np.median(valid_hr),
        'hr_range': np.ptp(valid_hr)
    })

    # Statystyki wyższego rzędu
    signal_std = np.std(valid_hr, ddof=1)
    if signal_std > 1e-8:
        features.update({
            'hr_skew': skew(valid_hr),
            'hr_kurtosis': kurtosis(valid_hr)
        })
    else:
        features.update({'hr_skew': 0.0, 'hr_kurtosis': 0.0})

    # Tempo zmian HR
    derivative = np.diff(valid_hr) * fs
    features['hr_change_rate'] = np.sqrt(np.mean(np.square(derivative)))

    # Fluktuacje HR w różnych oknach czasowych
    def get_fluctuation(signal: np.ndarray, window_sec: int) -> float:
        window_size = window_sec * fs
        max_fluct = 0
        for i in range(0, len(signal), window_size):
            window = signal[i:i+window_size]
            if len(window) > 5:
                max_fluct = max(max_fluct, np.ptp(window))
        return max_fluct
    
    features.update({
        'hr_max_fluct': get_fluctuation(valid_hr, 3),  # 3-sekundowe okno
        'hr_fluct_30s': get_fluctuation(valid_hr, 30),
        'hr_fluct_60s': get_fluctuation(valid_hr, 60),
        'hr_fluct_180s': get_fluctuation(valid_hr, 180)
    })

    # Analiza HRV w dziedzinie częstotliwości
    try:
        freqs, psd = welch(valid_hr, fs=fs, nperseg=min(256, len(valid_hr)))
        bands = {
            'ulf': (0, 0.003), 
            'vlf': (0.003, 0.04), 
            'lf': (0.04, 0.15), 
            'hf': (0.15, 0.4)
        }
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            features[f'hr_psd_{band}'] = np.trapezoid(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        
        features['hr_lf_hf_ratio'] = (
            features['hr_psd_lf'] / features['hr_psd_hf'] 
            if features['hr_psd_hf'] > 1e-8 else 0.0
        )
    except Exception as e:
        logger.warning(f"Błąd analizy PSD HR: {str(e)}")
        for band in bands:
            features[f'hr_psd_{band}'] = 0.0
        features['hr_lf_hf_ratio'] = 0.0

    return features


# ------------------- FUNKCJE PRZETWARZANIA SpO2 -------------------

def compute_shannon_entropy(signal: np.ndarray, bins: int = 20) -> float:
    """
    Oblicza entropię Shannona sygnału SpO2
    
    Args:
        signal: Sygnał SpO2
        bins: Liczba przedziałów do histogramu
        
    Returns:
        Wartość entropii Shannona
    """
    try:
        # Normalizacja sygnału i tworzenie histogramu
        hist, _ = np.histogram(signal, bins=bins, density=True)
        hist = hist[hist > 0]  # Usuń zera aby uniknąć log(0)
        
        # Obliczanie entropii
        entropy = -np.sum(hist * np.log2(hist))
        return entropy / np.log2(bins)  # Normalizacja do przedziału [0,1]
        
    except Exception as e:
        logger.warning(f"Błąd obliczania entropii Shannona: {str(e)}")
        return 0.0

def compute_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Oblicza Sample Entropy dla sygnału SpO2
    
    Args:
        signal: Sygnał SpO2
        m: Długość wzorca do porównania
        r: Próg podobieństwa (ułamek odchylenia standardowego)
        
    Returns:
        Wartość Sample Entropy
    """
    try:
        N = len(signal)
        if N <= m + 1:
            return 0.0
            
        # Normalizacja sygnału
        std = np.nanstd(signal)
        if std < 1e-8:
            return 0.0
            
        r *= std
        
        # Tworzenie wektorów
        vectors = np.array([signal[i:i+m+1] for i in range(N - m)])
        
        # Liczenie podobnych wzorców
        B = 0.0
        A = 0.0
        
        for i in range(len(vectors)-1):
            # Odległość Chebysheva między wektorami
            dist = np.max(np.abs(vectors[i+1:] - vectors[i]), axis=1)
            
            # Liczenie dopasowań
            B += np.sum(dist[:m] < r)
            A += np.sum(dist < r)
            
        # Zabezpieczenie przed dzieleniem przez zero
        if B == 0:
            return 0.0
            
        return -np.log(A / B)
        
    except Exception as e:
        logger.warning(f"Błąd obliczania Sample Entropy: {str(e)}")
        return 0.0

def clean_sao2_series(sao2_series: np.ndarray, fs: int = FS, log_prefix: str = "") -> np.ndarray:
    """
    Czyści sygnał SpO2, zastępując artefakty medianą pacjenta
    
    Args:
        sao2_series: Sygnał SpO2
        fs: Częstotliwość próbkowania
        log_prefix: Prefiks do logowania
        
    Returns:
        Oczyszczony sygnał SpO2
    """
    sao2_series = np.array(sao2_series)
    
    # Obliczanie mediany z prawidłowych wartości
    valid_mask = (sao2_series >= SAO2_MIN) & (sao2_series <= SAO2_MAX)
    valid_values = sao2_series[valid_mask]
    patient_median = np.median(valid_values) if len(valid_values) > 0 else (SAO2_MIN + SAO2_MAX) / 2
    
    # Zamień wartości spoza zakresu na medianę
    mask_invalid = ~valid_mask
    sao2_replaced = np.where(mask_invalid, patient_median, sao2_series)
    
    # Wykrywanie nagłych skoków (>3% zmiany)
    diff = np.abs(np.diff(sao2_replaced, prepend=sao2_replaced[0]))
    mask_spikes = diff > 3  # x% próg
    
    # Filtracja medianowa lokalna
    window_size = 3 * fs  # 3-sekundowe okno
    sao2_smoothed = (
        pd.Series(sao2_replaced)
        .rolling(window_size, center=True, min_periods=1)
        .median()
        .ffill()
        .bfill()
        .values
    )
    
    # Finalne czyszczenie
    cleaned_sao2 = np.where(mask_spikes, sao2_smoothed, sao2_replaced)
    
    # Logowanie
    n_corrected = np.sum(mask_invalid) + np.sum(mask_spikes)
    if n_corrected > 0:
        logger.info(
            f"{log_prefix} | Zastąpiono {n_corrected} artefaktów SAO2 "
            f"({n_corrected/len(sao2_series)*100:.1f}% danych) | "
            f"Mediana SAO2 pacjenta: {patient_median:.1f}%"
        )
    
    return cleaned_sao2

def analyze_sao2_features(sao2_segment: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """
    Analiza sygnału SpO2 i wykrywanie desaturacji
    
    Args:
        sao2_segment: Sygnał SpO2
        fs: Częstotliwość próbkowania
        
    Returns:
        Słownik z cechami SpO2
    """
    features = {
        'sao2_num_desats': 0,
        'sao2_ct90': 0.0,
        'sao2_avg_depth': 0.0,
        'sao2_max_depth': 0.0,
        'sao2_avg_duration': 0.0,
        'sao2_mean': np.nanmean(sao2_segment),
        'sao2_std': np.nanstd(sao2_segment),
        'sao2_shannon_entropy': 0.0,
        'sao2_sample_entropy': 0.0
    }
    
    try:
        # Obliczanie baseline
        baseline_window = min(30*fs, len(sao2_segment))
        baseline = np.median(sao2_segment[:baseline_window])
        
        # Wykrywanie desaturacji
        desat_mask = sao2_segment < (baseline - DESATURATION_THRESHOLD)
        below_90_mask = sao2_segment < 90
        
        # Znajdowanie epizodów desaturacji
        changes = np.diff(desat_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Analiza epizodów
        durations = []
        depths = []
        for s, e in zip(starts, ends):
            duration = (e - s) / fs
            if duration >= MIN_DESATURATION_DURATION:
                depth = baseline - np.min(sao2_segment[s:e])
                durations.append(duration)
                depths.append(depth)
        
        # Aktualizacja cech
        if depths:
            features.update({
                'sao2_num_desats': len(durations),
                'sao2_avg_depth': np.mean(depths),
                'sao2_max_depth': np.max(depths),
                'sao2_avg_duration': np.mean(durations)
            })
            
        features['sao2_ct90'] = np.sum(below_90_mask) / fs
        
        # Nowe cechy entropijne
        features.update({
            'sao2_shannon_entropy': compute_shannon_entropy(sao2_segment),
            'sao2_sample_entropy': compute_sample_entropy(sao2_segment)
        })

    except Exception as e:
        logger.error(f"Błąd analizy SpO2: {str(e)}")
    
    return features

# ------------------- ANALIZA FALKOWA -------------------
def extract_wavelet_features(signal: np.ndarray, signal_type: str) -> Dict[str, float]:
    """
    Ekstrakcja cech falkowych z obsługą błędów
    
    Args:
        signal: Sygnał wejściowy
        signal_type: Typ sygnału ('hr' lub 'sao2')
        
    Returns:
        Słownik z cechami falkowymi
    """
    params = WAVELET_PARAMS.get(signal_type, {'wavelet': 'db4', 'level': 5})
    features = {}
    
    try:
        coeffs = pywt.wavedec(signal, params['wavelet'], level=params['level'])
        for i, c in enumerate(coeffs):
            c_valid = c[~np.isnan(c)]
            if len(c_valid) == 0:
                stats = [0.0] * 8
            else:
                stats = [
                    np.mean(c_valid),
                    np.std(c_valid),
                    np.median(c_valid),
                    np.percentile(c_valid, 25),
                    np.percentile(c_valid, 75),
                    np.sum(np.square(c_valid)),
                    skew(c_valid) if np.std(c_valid) > 1e-8 else 0.0,
                    kurtosis(c_valid) if np.std(c_valid) > 1e-8 else 0.0
                ]
            
            for j, stat_name in enumerate(['mean', 'std', 'median', 'p25', 'p75', 'energy', 'skew', 'kurtosis']):
                features[f'{signal_type}_wf_l{i}_{stat_name}'] = stats[j]
                
    except Exception as e:
        logger.warning(f"Błąd analizy falkowej {signal_type}: {str(e)}")
        for i in range(params['level'] + 1):
            for stat_name in ['mean', 'std', 'median', 'p25', 'p75', 'energy', 'skew', 'kurtosis']:
                features[f'{signal_type}_wf_l{i}_{stat_name}'] = 0.0
    
    return features

# ------------------- GŁÓWNE PRZETWARZANIE -------------------
def process_segment(segment: pd.DataFrame) -> Dict[str, float]:
    """
    Przetwarza pojedynczy segment danych
    
    Args:
        segment: DataFrame z danymi segmentu
        
    Returns:
        Słownik z cechami segmentu
    """
    features = {}
    
    # Cechy HR
    hr_features = calculate_hr_features(segment['HR'].values)
    features.update(hr_features)
    
    # Cechy falkowe HR
    hr_wavelet = extract_wavelet_features(segment['HR'].values, 'hr')
    features.update(hr_wavelet)
    
    # Cechy SpO2
    sao2_features = analyze_sao2_features(segment['SAO2'].values)
    features.update(sao2_features)
    
    # Cechy falkowe SpO2
    sao2_wavelet = extract_wavelet_features(segment['SAO2'].values, 'sao2')
    features.update(sao2_wavelet)
    
    # Etykieta
    features['target'] = int(segment[['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events']].sum().sum() > 0)
    
    return features

def process_file(file_path: Path) -> pd.DataFrame:
    """
    Przetwarza pojedynczy plik z danymi
    
    Args:
        file_path: Ścieżka do pliku CSV
        
    Returns:
        DataFrame z wyekstrahowanymi cechami
    """
    try:
        logger.info(f"Przetwarzanie: {file_path.name}")
        
        # Wczytywanie danych
        df = pd.read_csv(file_path, usecols=[
            'TIMESTAMP', 'HR', 'SAO2', 
            'Obstructive_Apnea', 'Central_Apnea', 
            'Hypopnea', 'Sleep_Stage', 'Multiple_Events'
        ])
        
        # Filtracja faz snu
        df = df[df['Sleep_Stage'].isin(['N1', 'N2', 'N3', 'R'])].copy()
        
        # Czyszczenie sygnałów
        df['HR'] = clean_hr_signal(df['HR'].values, FS, file_path.name)
        df['SAO2'] = clean_sao2_series(df['SAO2'].values * 100, FS, file_path.name)
        
        # Przetwarzanie segmentów
        features = []
        for i in range(0, len(df) - SAMPLES_PER_EPOCH + 1, STEP):
            segment = df.iloc[i:i+SAMPLES_PER_EPOCH]
            features.append(process_segment(segment))
            
        return pd.DataFrame(features)
    
    except Exception as e:
        logger.error(f"Błąd przetwarzania {file_path.name}: {str(e)}")
        return pd.DataFrame()

# ------------------- ZAPIS DANYCH -------------------
def save_processed_data(data: pd.DataFrame):
    """
    Zapisuje przetworzone dane z podziałem na zbiory i balansowaniem
    
    Args:
        data: Pełny zbiór danych
    """
    logger.info("Finalne przetwarzanie danych...")
    
    # Podział na zbiory
    splits = {}
    for subset in SUBDIRS:
        split_data = data[data['source'] == subset].drop('source', axis=1)
        X = split_data.drop('target', axis=1)
        y = split_data['target']
        
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
        logger.error(f"Błąd balansowania SMOTE: {str(e)}")
    
    # Zapis do pliku
    PROCESSED_DIR.mkdir(exist_ok=True)
    joblib.dump(splits, PROCESSED_DIR / 'processed_data.pkl')
    logger.info(f"Zapisano dane w: {PROCESSED_DIR / 'processed_data.pkl'}")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    logger.info("Rozpoczęcie przetwarzania...")
    
    # Przetwarzanie równoległe
    file_paths = [p for subdir in SUBDIRS for p in (DATA_DIR/subdir).glob('*.csv')]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, file_paths))
    
    # Konsolidacja wyników
    full_data = pd.concat([df for df in results if not df.empty], ignore_index=True)
    full_data['source'] = np.concatenate([
        [subdir] * len(df) 
        for subdir, df in zip(
            np.repeat(SUBDIRS, [len(list((DATA_DIR/subdir).glob('*.csv'))) for subdir in SUBDIRS]),
            results
        )
    ])
    
    save_processed_data(full_data)
    logger.info("Przetwarzanie zakończone pomyślnie!")