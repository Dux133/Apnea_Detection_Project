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
from antropy import sample_entropy
from scipy.signal import welch, medfilt
from scipy.ndimage import median_filter

from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
from typing import Dict

# ------------------- KONFIGURACJA -------------------
# Ścieżki i parametry przetwarzania
BASE_DIR = Path(r'D:\Apnea_Detection_Project')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = BASE_DIR / 'processed_data'
SUBDIRS = ['train', 'val', 'test']
N_CORE = 3 # ile rdzeni użyć

FS = 100  # Częstotliwość próbkowania [Hz]
EPOCH_DURATION = 120 # Długość epoki w sekundach
SAMPLES_PER_EPOCH = FS * EPOCH_DURATION
OVERLAP_RATIO = 0.33  # % nakładania okien
STEP = int(SAMPLES_PER_EPOCH * (1 - OVERLAP_RATIO))
RANDOM_STATE = 42  # Ziarno losowości

# Zakresy i parametry sygnałów
HR_MIN, HR_MAX = 40, 140
SAO2_MIN, SAO2_MAX = 50, 100
DESATURATION_THRESHOLD = 3  # Minimalny spadek SpO2 [%]
MIN_DESATURATION_DURATION = 10  # Minimalny czas desaturacji [s]

# Parametry analizy falkowej
USE_WAVELET_FEATURES = True  # Można dynamicznie zmieniać
WAVELET_PARAMS = {
    'hr': {'wavelet': 'db4', 'level': 5},
    'sao2': {'wavelet': 'sym5', 'level': 4},
    'ibi': {'wavelet': 'db3', 'level': 3}  # Mniejsza głębokość dla krótszych sygnałów
}

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
    Czyści i filtruje sygnał HR, usuwając artefakty i nierealistyczne wartości
    
    Args:
        hr_series: Surowy sygnał HR [bpm]
        fs: Częstotliwość próbkowania [Hz] (domyślnie: FS)
        log_prefix: Prefiks do logowania (domyślnie: "")
        
    Returns:
        Oczyszczony sygnał HR [bpm]
        
    Przykład:
        >>> clean_hr_signal(np.array([80, 82, 800, 0, 85]), fs=1)
        array([80, 82, 82, 80, 85])  # przy HR_MIN=40, HR_MAX=200
    """
    # Sprawdzenie danych wejściowych
    if len(hr_series) == 0:
        return hr_series.copy()
    
    if fs <= 0:
        raise ValueError("Częstotliwość próbkowania musi być dodatnia")
    
    # Przycięcie wartości do zakresu fizjologicznego
    hr_clipped = np.clip(hr_series, HR_MIN, HR_MAX)
    
    # Wykrywanie artefaktów (gwałtowne zmiany i izolowane wartości)
    diff = np.abs(np.diff(hr_clipped, prepend=hr_clipped[0]))
    
    # Podwójny próg - większy dla pojedynczych skoków, mniejszy dla serii zmian
    mask_spikes = diff > 5  # Nagłe skoki >5 bpm
    mask_sustained = (diff > 2) & (~mask_spikes)  # Utrzymujące się zmiany >2 bpm
    
    # Filtracja medianowa z adaptacyjnym oknem
    window_sec = 10  # Domyślne okno 10-sekundowe
    window_size = min(window_sec * fs, len(hr_clipped))
    
    # Upewnij się, że okno jest nieparzyste i nie mniejsze niż 3
    window_size = max(3, window_size if window_size % 2 == 1 else window_size - 1)
    
    # Filtracja tylko jeśli okno ma sens
    if window_size >= 3:
        hr_smoothed = medfilt(hr_clipped, kernel_size=int(window_size))
    else:
        hr_smoothed = hr_clipped
    
    # Zastępowanie artefaktów - różna logika dla różnych typów
    cleaned_hr = np.where(
        mask_spikes,  # Dla nagłych skoków - użyj wygładzonych wartości
        hr_smoothed,
        np.where(
            mask_sustained,  # Dla utrzymujących się zmian - średnia z sąsiadów
            (np.roll(hr_clipped, 1) + np.roll(hr_clipped, -1)) / 2,
            hr_clipped  # Wartości prawidłowe pozostaw bez zmian
        )
    )
    
    # Dodatkowa filtracja dla bardzo krótkich serii artefaktów
    if len(cleaned_hr) > 10:
        # Znajdź pojedyncze wartości różniące się od sąsiadów
        neighbor_diff = np.abs(2*cleaned_hr - np.roll(cleaned_hr,1) - np.roll(cleaned_hr,-1))
        mask_single = (neighbor_diff > 10) & (~mask_spikes) & (~mask_sustained)
        cleaned_hr[mask_single] = (np.roll(cleaned_hr,1)[mask_single] + np.roll(cleaned_hr,-1)[mask_single]) / 2
    
    # Logowanie statystyk (tylko jeśli logowanie jest włączone)
    if logger.isEnabledFor(logging.INFO):
        n_total = len(hr_series)
        n_corrected = np.sum(mask_spikes | mask_sustained)
        n_clipped = np.sum((hr_series < HR_MIN) | (hr_series > HR_MAX))
        
        if n_corrected > 0 or n_clipped > 0:
            logger.info(
                f"{log_prefix} | Statystyki czyszczenia HR:\n"
                f"- Przycięto {n_clipped} wartości ({n_clipped/n_total*100:.1f}%)\n"
                f"- Poprawiono {n_corrected} artefaktów ({n_corrected/n_total*100:.1f}%)\n"
                f"- Ostateczny zakres: {np.min(cleaned_hr):.0f}-{np.max(cleaned_hr):.0f} bpm"
            )
    
    return cleaned_hr

def calculate_hr_features(hr_signal: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """
    Oblicza cechy HR (włącznie z HRV i fluktuacjami)
    
    Args:
        hr_signal: Sygnał tętna [bpm]
        fs: Częstotliwość próbkowania [Hz]
        
    Returns:
        Słownik z cechami HR:
            - hr_mean: Średnie tętno [bpm]
            - hr_std: Odchylenie standardowe [bpm]
            - hr_median: Mediana tętna [bpm]
            - hr_range: Rozstęp (max-min) [bpm]
            - hr_skew: Skosność rozkładu
            - hr_kurtosis: Kurtoza rozkładu
            - hr_change_rate: Średnie tempo zmian [bpm/s]
            - hr_max_fluct: Maksymalna fluktuacja w 3-sekundowych oknach [bpm]
            - hr_fluct_30s/60s/180s: Maks. fluktuacja w dłuższych oknach [bpm]
            - hr_psd_[ulf/vlf/lf/hf]: Moc PSD w pasmach częstotliwości [ms²/Hz]
            - hr_lf_hf_ratio: Stosunek mocy LF/HF
    """
    MIN_SAMPLES = 10  # Minimalna liczba próbek do analizy
    features = {}
    valid_hr = hr_signal[~np.isnan(hr_signal)]
    
    # Inicjalizacja domyślnych wartości
    default_features = {
        'hr_mean': 0.0, 'hr_std': 0.0, 'hr_median': 0.0, 'hr_range': 0.0,
        'hr_skew': 0.0, 'hr_kurtosis': 0.0, 'hr_change_rate': 0.0,
        'hr_max_fluct': 0.0, 'hr_fluct_30s': 0.0, 'hr_fluct_60s': 0.0, 'hr_fluct_180s': 0.0,
        'hr_psd_ulf': 0.0, 'hr_psd_vlf': 0.0, 'hr_psd_lf': 0.0, 'hr_psd_hf': 0.0,
        'hr_lf_hf_ratio': 0.0
    }
    
    if len(valid_hr) < MIN_SAMPLES:
        return default_features
    
    # Podstawowe statystyki
    hr_mean = np.mean(valid_hr)
    hr_std = np.std(valid_hr, ddof=1)
    
    features.update({
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_median': np.median(valid_hr),
        'hr_range': np.ptp(valid_hr)
    })

    # Statystyki wyższego rzędu
    if hr_std > 1e-8:
        features.update({
            'hr_skew': skew(valid_hr),
            'hr_kurtosis': kurtosis(valid_hr)
        })
    else:
        features.update({'hr_skew': 0.0, 'hr_kurtosis': 0.0})

    # Tempo zmian HR
    derivative = np.diff(valid_hr) * fs
    features['hr_change_rate'] = np.sqrt(np.mean(np.square(derivative)))

    # Fluktuacje HR - zoptymalizowana wersja
    def get_max_fluctuation(signal: np.ndarray, window_size: int) -> float:
        if len(signal) <= window_size:
            return np.ptp(signal)
        
        # Użyj rolling window dla lepszej wydajności
        strides = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        return np.max(np.ptp(strides, axis=1)) if strides.size > 0 else 0.0
    
    window_sizes = {
        'hr_max_fluct': 3 * fs,    # 3 sekundy
        'hr_fluct_30s': 30 * fs,    # 30 sekund
        'hr_fluct_60s': 60 * fs,    # 1 minuta
        'hr_fluct_180s': 180 * fs   # 3 minuty
    }
    
    for name, size in window_sizes.items():
        features[name] = get_max_fluctuation(valid_hr, int(size)) if size > 0 else 0.0

    # Analiza HRV w dziedzinie częstotliwości
    try:
        nperseg = min(256, len(valid_hr))
        if nperseg < 8:  # Zbyt mało próbek dla PSD
            raise ValueError("Za mało próbek do analizy PSD")
            
        freqs, psd = welch(valid_hr, fs=fs, nperseg=nperseg)
        
        bands = {
            'ulf': (0, 0.003), 
            'vlf': (0.003, 0.04), 
            'lf': (0.04, 0.15), 
            'hf': (0.15, 0.4)
        }
        
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features[f'hr_psd_{band}'] = np.trapz(psd[mask], freqs[mask])
            else:
                features[f'hr_psd_{band}'] = 0.0
        
        # Zabezpieczenie przed dzieleniem przez zero
        hf_power = features['hr_psd_hf']
        features['hr_lf_hf_ratio'] = features['hr_psd_lf'] / hf_power if hf_power > 1e-8 else 0.0
        
    except Exception as e:
        logger.warning(f"Błąd analizy PSD HR: {str(e)}", exc_info=True)
        # Nie trzeba przypisywać 0.0, bo domyślne wartości już są ustawione
    
    return features

# ------------------- FUNKCJE PRZETWARZANIA SpO2 -------------------

def clean_sao2_series(sao2_series: np.ndarray, fs: int = FS, log_prefix: str = "") -> np.ndarray:
    """
    Czyści sygnał SpO2, zastępując artefakty medianą pacjenta
    
    Args:
        sao2_series: Sygnał SpO2 (wartości w %)
        fs: Częstotliwość próbkowania [Hz] (domyślnie: FS)
        log_prefix: Prefiks do logowania (domyślnie: "")
        
    Returns:
        Oczyszczony sygnał SpO2
        
    Przykład:
        >>> clean_sao2_series(np.array([95, 96, 120, 94, 0, 97]), fs=1)
        array([95, 96, 96, 94, 96, 97])  # przy założeniu SAO2_MIN=50, SAO2_MAX=100
    """
    # Konwersja na numpy array i sprawdzenie danych wejściowych
    sao2_series = np.array(sao2_series)
    if len(sao2_series) == 0:
        return sao2_series
    if fs <= 0:
        raise ValueError("Częstotliwość próbkowania musi być dodatnia")

    # Obliczanie mediany z prawidłowych wartości
    valid_mask = (sao2_series >= SAO2_MIN) & (sao2_series <= SAO2_MAX)
    valid_values = sao2_series[valid_mask]
    
    if len(valid_values) == 0:
        patient_median = (SAO2_MIN + SAO2_MAX) / 2
        logger.warning(f"{log_prefix} | Brak prawidłowych wartości SpO2 - użyto mediany domyślnej")
    else:
        patient_median = np.median(valid_values)

    # Zamień wartości spoza zakresu na medianę
    mask_invalid = ~valid_mask
    sao2_replaced = np.where(mask_invalid, patient_median, sao2_series)

    # Filtracja medianowa lokalna (najpierw wygładzanie)
    window_size = min(3 * fs, len(sao2_replaced))  # 3-sekundowe okno, ale nie dłuższe niż sygnał
    if window_size > 1:  # Tylko jeśli okno ma sens
        sao2_smoothed = median_filter(sao2_replaced, size=window_size, mode='nearest')
    else:
        sao2_smoothed = sao2_replaced

    # Wykrywanie nagłych skoków (>3% zmiany) po wygładzeniu
    diff = np.abs(np.diff(sao2_smoothed, prepend=sao2_smoothed[0]))
    mask_spikes = diff > 3  # 3% próg

    # Finalne czyszczenie
    cleaned_sao2 = np.where(mask_spikes | mask_invalid, sao2_smoothed, sao2_series)

    # Logowanie
    n_corrected = np.sum(mask_invalid) + np.sum(mask_spikes)
    if n_corrected > 0 and logger.isEnabledFor(logging.INFO):
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
        fs: Częstotliwość próbkowania [Hz]
        
    Returns:
        Słownik z cechami SpO2:
            - sao2_num_desats: Liczba desaturacji
            - sao2_ct90: Czas spędzony poniżej 90% [s]
            - sao2_avg_depth: Średnia głębokość desaturacji [%]
            - sao2_max_depth: Maksymalna głębokość desaturacji [%]
            - sao2_avg_duration: Średni czas trwania desaturacji [s]
            - sao2_mean: Średnia wartość SpO2 [%]
            - sao2_std: Odchylenie standardowe SpO2 [%]
    """
    # Inicjalizacja wyników z domyślnymi wartościami
    features = {
        'sao2_num_desats': 0,
        'sao2_ct90': 0.0,
        'sao2_avg_depth': 0.0,
        'sao2_max_depth': 0.0,
        'sao2_avg_duration': 0.0,
        'sao2_mean': np.nanmean(sao2_segment),
        'sao2_std': np.nanstd(sao2_segment),
    }
    
    # Sprawdzenie danych wejściowych
    if len(sao2_segment) == 0 or fs <= 0:
        return features
    
    try:
        # Obliczanie baseline (używamy mediany z pierwszych 30 sekund)
        baseline_window = min(30*fs, len(sao2_segment))
        baseline = np.median(sao2_segment[:baseline_window]) if baseline_window > 0 else np.nanmedian(sao2_segment)
        
        if np.isnan(baseline):
            return features
            
        # Wykrywanie desaturacji
        desat_mask = sao2_segment < (baseline - DESATURATION_THRESHOLD)
        below_90_mask = sao2_segment < 90
        
        # Znajdowanie epizodów desaturacji
        changes = np.diff(desat_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Dopasowanie długości jeśli różne
        if len(starts) > len(ends):
            ends = np.append(ends, len(sao2_segment))
        elif len(ends) > len(starts):
            starts = np.insert(starts, 0, 0)

        # Analiza epizodów
        durations = []
        depths = []
        for s, e in zip(starts, ends):
            duration = (e - s) / fs
            if duration >= MIN_DESATURATION_DURATION:
                depth = baseline - np.min(sao2_segment[s:e])
                durations.append(duration)
                depths.append(depth)
        
        # Aktualizacja cech jeśli znaleziono desaturacje
        if depths:
            features.update({
                'sao2_num_desats': len(durations),
                'sao2_avg_depth': np.mean(depths),
                'sao2_max_depth': np.max(depths),
                'sao2_avg_duration': np.mean(durations)
            })
            
        # CT90 jest obliczany niezależnie od desaturacji
        features['sao2_ct90'] = np.sum(below_90_mask) / fs
        
    except Exception as e:
        logger.error(f"Błąd analizy SpO2: {str(e)}", exc_info=True)
    
    return features

# ------------------- ANALIZA FALKOWA -------------------
def extract_wavelet_features(signal: np.ndarray, signal_type: str) -> Dict[str, float]:
    """
    Ekstrakcja cech falkowych z sygnału z obsługą błędów i walidacją danych
    
    Args:
        signal: Sygnał wejściowy (1D numpy array)
        signal_type: Typ sygnału ('hr' dla tętna lub 'sao2' dla saturacji)
        
    Returns:
        Słownik z cechami falkowymi w formacie:
        {
            '<typ>_wf_l<poziom>_<cecha>': wartość,
            ...
        }
        gdzie <cecha> to jedna z: mean, std, median, p25, p75, energy, skew, kurtosis
    """
    # Domyślne parametry i walidacja wejścia
    DEFAULT_PARAMS = {'wavelet': 'db4', 'level': 5}
    params = WAVELET_PARAMS.get(signal_type, DEFAULT_PARAMS)
    features = {}
    
    # Sprawdzenie czy sygnał jest wystarczająco długi
    min_length = 2 ** params['level']  # Minimalna długość dla dekompozycji
    if len(signal) < min_length:
        logger.warning(f"Sygnał {signal_type} zbyt krótki ({len(signal)} próbek) dla poziomu {params['level']}")
        return _generate_empty_wavelet_features(signal_type, params['level'])
    
    try:
        # Usuń wartości NaN i sprawdź czy zostały jakieś dane
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) < min_length:
            logger.warning(f"Za mało nie-NaN wartości w sygnale {signal_type} ({len(clean_signal)}/{len(signal)})")
            return _generate_empty_wavelet_features(signal_type, params['level'])
        
        # Dekompozycja falkowa
        coeffs = pywt.wavedec(clean_signal, params['wavelet'], level=params['level'], mode='per')
        
        # Obliczanie statystyk dla każdego poziomu dekompozycji
        for level, coeff in enumerate(coeffs):
            if len(coeff) == 0:
                stats = [0.0] * 8
            else:
                stats = _calculate_wavelet_stats(coeff)
            
            # Zapisz cechy
            for stat_name, value in zip(['mean', 'std', 'median', 'p25', 'p75', 'energy', 'skew', 'kurtosis'], stats):
                features[f'{signal_type}_wf_l{level}_{stat_name}'] = value
        
        # Dodatkowe cechy globalne
        features.update(_calculate_global_wavelet_features(coeffs, signal_type))
        
    except pywt.WaveletError as e:
        logger.error(f"Błąd falki {params['wavelet']} dla {signal_type}: {str(e)}")
        return _generate_empty_wavelet_features(signal_type, params['level'])
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd analizy falkowej {signal_type}: {str(e)}", exc_info=True)
        return _generate_empty_wavelet_features(signal_type, params['level'])
    
    return features

def _calculate_wavelet_stats(coeff: np.ndarray) -> list:
    """Oblicza statystyki dla pojedynczego współczynnika falkowego"""
    coeff = coeff[~np.isnan(coeff)]
    if len(coeff) == 0:
        return [0.0] * 8
    
    std = np.std(coeff, ddof=1)
    stats = [
        np.mean(coeff),
        std,
        np.median(coeff),
        np.percentile(coeff, 25),
        np.percentile(coeff, 75),
        np.sum(np.square(coeff)),  # Energia
        skew(coeff) if std > 1e-8 else 0.0,
        kurtosis(coeff) if std > 1e-8 else 0.0
    ]
    return stats

def _generate_empty_wavelet_features(signal_type: str, level: int) -> Dict[str, float]:
    """Generuje domyślne wartości cech falkowych"""
    features = {}
    for l in range(level + 1):
        for stat_name in ['mean', 'std', 'median', 'p25', 'p75', 'energy', 'skew', 'kurtosis']:
            features[f'{signal_type}_wf_l{l}_{stat_name}'] = 0.0
    return features

def _calculate_global_wavelet_features(coeffs: list, signal_type: str) -> Dict[str, float]:
    """Oblicza globalne cechy falkowe"""
    global_features = {}
    
    # Stosunek energii poziomów
    energies = [np.sum(np.square(c[~np.isnan(c)])) for c in coeffs]
    total_energy = sum(energies)
    
    if total_energy > 1e-8:
        for i, energy in enumerate(energies):
            global_features[f'{signal_type}_wf_energy_ratio_l{i}'] = energy / total_energy
    
    # Entropia współczynników
    for i, c in enumerate(coeffs):
        c_clean = c[~np.isnan(c)]
        if len(c_clean) > 1:
            energy = np.sum(np.square(c_clean))
            if energy > 1e-8:
                p = np.square(c_clean) / energy
                global_features[f'{signal_type}_wf_entropy_l{i}'] = -np.sum(p * np.log2(p + 1e-12))
            else:
                global_features[f'{signal_type}_wf_entropy_l{i}'] = 0.0

    
    return global_features

# ------------------- FUNKCJE PRZETWARZANIA IBI -------------------
def clean_ibi_signal(ibi_series: np.ndarray, 
                    fs: float = 1.0, 
                    log_prefix: str = "") -> np.ndarray:
    """
    Czyści sygnał IBI (Inter-Beat Intervals) w milisekundach [ms]
    
    Args:
        ibi_series: Sygnał IBI w [ms]
        fs: Częstotliwość próbkowania (opcjonalna)
        log_prefix: Prefiks do logowania
        
    Returns:
        Oczyszczony sygnał IBI [ms]
        
    Przykład:
        >>> clean_ibi_signal(np.array([800, 2000, 300, 850, np.nan]))
        array([800., 800., 800., 850., 850.])  # przy IBI_MIN=300, IBI_MAX=2000
    """
    # Stałe fizjologiczne
    IBI_MIN = 0.4  # 40 bpm
    IBI_MAX = 1.5  # 150 bpm
    
    ibi = ibi_series.copy().astype(float)
    
    # 1. Usuwanie NaN i wartości spoza zakresu
    mask_nan = np.isnan(ibi)
    ibi[mask_nan] = np.nanmedian(ibi)  # Tymczasowe wypełnienie dla obliczeń
    
    # 2. Przycinanie do zakresu fizjologicznego
    ibi_clipped = np.clip(ibi, IBI_MIN, IBI_MAX)
    
    # 3. Wykrywanie artefaktów (nagłe zmiany >20% mediany)
    median_ibi = np.median(ibi_clipped)
    threshold = 0.2 * median_ibi  # 20% mediany
    diff = np.abs(np.diff(ibi_clipped, prepend=median_ibi))
    mask_spikes = diff > threshold
    
    # 4. Filtracja medianowa adaptacyjna
    window_size = max(3, min(5, len(ibi_clipped)))  # Okno 3-5 próbek
    ibi_smoothed = medfilt(ibi_clipped, kernel_size=window_size)
    
    # 5. Rekonstrukcja sygnału
    cleaned_ibi = np.where(mask_spikes | mask_nan, ibi_smoothed, ibi_clipped)
    
    # Logowanie
    n_corrected = np.sum(mask_spikes | mask_nan)
    if n_corrected > 0 and logger.isEnabledFor(logging.INFO):
        logger.info(
            f"{log_prefix} | IBI: Poprawiono {n_corrected} artefaktów "
            f"({n_corrected/len(ibi)*100:.1f}%) | Mediana: {median_ibi:.1f} s"
        )
    return (cleaned_ibi * 1000).astype(int)  # Z sekundy → ms zwracamy wartości całkowitoliczbowe [s]

def _generate_default_ibi_features() -> Dict[str, float]:
    """Generuje domyślne wartości dla błędów"""
    
    return {k: 0.0 for k in [
        'ibi_mean', 'ibi_std', 'ibi_median', 'ibi_range', 'ibi_cv',
        'ibi_rmssd', 'ibi_nn50', 'ibi_pnn50', 'ibi_sdsd', 'ibi_si', 'ibi_ti',
        'ibi_entropy', 'ibi_skewness', 'ibi_kurtosis',
        'ibi_psd_ulf', 'ibi_psd_vlf', 'ibi_psd_lf', 'ibi_psd_hf',
        'ibi_lf_hf_ratio', 'ibi_total_power'
    ]}

def _calculate_frequency_features(ibi: np.ndarray, fs: float) -> Dict[str, float]:
    """Analiza częstotliwościowa IBI (HF, LF, VLF)"""
    from scipy.signal import welch
    freqs, psd = welch(ibi, fs=fs, nperseg=min(256, len(ibi)))
    
    bands = {
        'ulf': (0, 0.003),
        'vlf': (0.003, 0.04),
        'lf': (0.04, 0.15),
        'hf': (0.15, 0.4)
    }
    
    features = {}
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        features[f'ibi_psd_{band}'] = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
    
    if features.get('ibi_psd_hf', 0) > 1e-8:
        features.update({
            'ibi_lf_hf_ratio': features['ibi_psd_lf'] / features['ibi_psd_hf'],
            'ibi_total_power': np.trapezoid(psd, freqs)
        })
    
    return features  

def analyze_ibi_features(ibi_series: np.ndarray, fs: float = 1.0, log_prefix: str = "") -> Dict[str, float]:
    """
    Analiza sygnału IBI (Inter-Beat Intervals) w milisekundach [ms].
    Oblicza statystyki czasowe, nieregularności oraz cechy nieliniowe.
    """
    features = {}
    valid_ibi = ibi_series[~np.isnan(ibi_series)]
    
    if len(valid_ibi) < 5:
        logger.warning(f"{log_prefix} | Za krótki sygnał IBI ({len(valid_ibi)} próbek)")
        return _generate_default_ibi_features()
    
    try:
        # Podstawowe statystyki
        features.update({
            'ibi_mean': np.mean(valid_ibi),
            'ibi_std': np.std(valid_ibi, ddof=1),
            'ibi_median': np.median(valid_ibi),
            'ibi_range': np.ptp(valid_ibi),
            'ibi_cv': np.std(valid_ibi, ddof=1) / np.mean(valid_ibi) if np.mean(valid_ibi) > 0 else 0.0
        })
        
        # Analiza różnicowa
        diff = np.diff(valid_ibi)
        features.update({
            'ibi_rmssd': np.sqrt(np.mean(np.square(diff))) if len(diff) > 0 else 0.0,
            'ibi_nn50': np.sum(np.abs(diff) > 50),
            'ibi_pnn50': np.sum(np.abs(diff) > 50) / len(diff) * 100 if len(diff) > 0 else 0.0,
            'ibi_sdsd': np.std(diff, ddof=1) if len(diff) > 1 else 0.0
        })

        # Wskaźniki stresu
        features.update({
            'ibi_si': (features['ibi_std'] * 1000) / (2 * features['ibi_mean'] * features['ibi_rmssd']) 
                     if features.get('ibi_rmssd', 0) > 0 else 0.0,
            'ibi_ti': features['ibi_mean'] / features['ibi_std'] if features.get('ibi_std', 0) > 0 else 0.0
        })

        # Cechy nieliniowe
        if len(valid_ibi) >= 10:
            try:
                features['ibi_entropy'] = sample_entropy(valid_ibi, order=2)
            except ImportError:
                logger.warning(f"{log_prefix} | antropy nie zainstalowane. Pominięto entropy.")
                features['ibi_entropy'] = 0.0
            
            if np.std(valid_ibi, ddof=1) > 1e-8:
                features.update({
                    'ibi_skewness': skew(valid_ibi),
                    'ibi_kurtosis': kurtosis(valid_ibi)
                })
            else:
                features.update({
                    'ibi_skewness': 0.0,
                    'ibi_kurtosis': 0.0
                })


        # Analiza częstotliwościowa
        if fs > 0 and len(valid_ibi) >= 30:
            try:
                features.update(_calculate_frequency_features(valid_ibi, fs))
            except Exception as e:
                logger.warning(f"{log_prefix} | Błąd analizy częstotliwościowej: {str(e)}")
    
    except Exception as e:
        logger.info(f"{log_prefix} | Średnia IBI [ms]: {np.mean(valid_ibi):.2f}")
        logger.error(f"{log_prefix} | Krytyczny błąd: {str(e)}", exc_info=True)
        return _generate_default_ibi_features()
    
    return features

# ------------------- GŁÓWNE PRZETWARZANIE -------------------
def process_segment(segment: pd.DataFrame) -> Dict[str, float]:
    """
    Przetwarza pojedynczy segment danych z uwzględnieniem HR, SpO2 i IBI.
    
    Args:
        segment: DataFrame z kolumnami:
                - 'HR' (tętno w bpm)
                - 'SAO2' (saturacja w %)
                - 'IBI' (interwały między uderzeniami w ms)
                - Znaczniki zdarzeń ('Obstructive_Apnea', itp.)
                
    Returns:
        Słownik z cechami i etykietą
    """
    features = {}
    
    # 1. Cechy czasowe
    features.update(calculate_hr_features(segment['HR'].values))
    features.update(analyze_sao2_features(segment['SAO2'].values))
    
    # 2. Analiza IBI (jeśli dostępne)
    if 'IBI' in segment.columns:
        ibi_clean = segment['IBI'].dropna().values
        if len(ibi_clean) >= 5:  # Minimalna liczba interwałów
            features.update(analyze_ibi_features(ibi_clean))
        else:
            logger.warning(f"Segment ma za mało danych IBI: {len(ibi_clean)} próbek")
            features.update(_generate_default_ibi_features())
    
    # 3. Analiza falkowa (opcjonalna)
    if USE_WAVELET_FEATURES:
        # Dla HR i SpO2
        hr_coeffs = pywt.wavedec(segment['HR'].values, 
                               WAVELET_PARAMS['hr']['wavelet'], 
                               level=WAVELET_PARAMS['hr']['level'])
        features.update(_calculate_global_wavelet_features(hr_coeffs, 'hr'))
        
        # Dla IBI (jeśli dostępne)
        if 'IBI' in segment.columns and len(ibi_clean) >= 10:
            ibi_coeffs = pywt.wavedec(ibi_clean, 
                                    WAVELET_PARAMS['ibi']['wavelet'], 
                                    level=min(3, WAVELET_PARAMS['ibi']['level']))  # Mniejszy poziom dla krótkich IBI
            features.update(_calculate_global_wavelet_features(ibi_coeffs, 'ibi'))
    
    # 4. Etykieta
    features['target'] = int(segment[['Obstructive_Apnea', 'Central_Apnea', 
                                   'Hypopnea', 'Multiple_Events']].any().any())
    
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
            'TIMESTAMP', 'HR', 'SAO2', 'IBI',
            'Obstructive_Apnea', 'Central_Apnea', 
            'Hypopnea', 'Sleep_Stage', 'Multiple_Events'
        ])
        
        # Filtracja faz snu
        df = df[df['Sleep_Stage'].isin(['N1', 'N2', 'N3', 'R'])].copy()
        
        # Czyszczenie sygnałów
        df['HR'] = clean_hr_signal(df['HR'].values, FS, file_path.name)
        df['SAO2'] = clean_sao2_series(df['SAO2'].values * 100, FS, file_path.name)

        # Dodaj czyszczenie IBI jeśli kolumna istnieje
        if 'IBI' in df.columns:
            df['IBI'] = clean_ibi_signal(df['IBI'].values, FS, file_path.name)
        else:
            logger.warning(f"{file_path.name} | Brak kolumny IBI w danych")

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
    
    with ProcessPoolExecutor(max_workers=N_CORE) as executor:
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