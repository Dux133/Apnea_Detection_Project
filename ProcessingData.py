import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pywt
import antropy as ant
from pathlib import Path
import logging
from time import time
import warnings
warnings.filterwarnings("error")


# ---------- KONFIGURACJA ----------
fs = 100  # Hz
N_CORE = 6
DATA_DIR = Path("D:/Apnea_Detection_Project/data")
OUTPUT_FILE = "processed_data_Vr.pkl"
RELEVANT_COLUMNS = ["TIMESTAMP", "SAO2", "HR", "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea"]

# ---------- LOGOWANIE ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- FUNKCJE ----------
def clean_and_interpolate(df):
    """
    Czyści i interpoluje dane w DataFrame.
    - Wypełnia brakujące wartości w kolumnach związanych z bezdechem zerami
    - Konwertuje SAO2 na procenty i przycina wartości do zakresu 50-100
    - Przycina wartości HR do zakresu 30-220
    - Interpoluje liniowo brakujące wartości w SAO2 i HR
    - Mapuje etapy snu na wartości liczbowe
    
    Cleans and interpolates data in DataFrame.
    - Fills missing values in apnea-related columns with zeros
    - Converts SAO2 to percentages and clips values to 50-100 range
    - Clips HR values to 30-220 range
    - Linearly interpolates missing values in SAO2 and HR
    - Maps sleep stages to numeric values
    """
    apnea_cols = [col for col in df.columns if "Apnea" in col]
    df[apnea_cols] = df[apnea_cols].fillna(0).astype(int)
    df["SAO2"] *= 100  # konwersja do procentów / convert to percentages
    df["SAO2"] = df["SAO2"].clip(lower=50, upper=100)
    df["HR"] = df["HR"].clip(lower=30, upper=220)

    for col in ["SAO2", "HR"]:
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].bfill().ffill()

    sleep_stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
    df["Sleep_Stage_Coded"] = df["Sleep_Stage"].map(sleep_stage_map).fillna(-1).astype(int)

    return df

def wavelet_features(signal, wavelet='db4', level=4):
    """
    Oblicza cechy falkowe (energia i odchylenie standardowe) dla sygnału.
    - Dla sygnału stałego zwraca wartości NaN
    - Używa falki db4 i 4 poziomów dekompozycji
    
    Computes wavelet features (energy and standard deviation) for a signal.
    - Returns NaN values for constant signals
    - Uses db4 wavelet and 4 decomposition levels
    """
    if np.all(signal == signal[0]):  # sygnał stały
        return {f'w_energy_d{i}': np.nan for i in range(1, level+1)} | \
               {f'w_std_d{i}': np.nan for i in range(1, level+1)}
    
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = {}
        for i, c in enumerate(coeffs[1:], 1):
            std = np.std(c)
            features[f'w_energy_d{i}'] = np.sum(np.square(c)) if np.isfinite(std) and std > 0 else np.nan
            features[f'w_std_d{i}'] = std if np.isfinite(std) and std > 0 else np.nan
        return features
    except Exception:
        return {f'w_energy_d{i}': np.nan for i in range(1, level+1)} | \
               {f'w_std_d{i}': np.nan for i in range(1, level+1)}

# def wavelet_features(signal, wavelet='db4', level=4):
#     """
#     Oblicza wybrane cechy falkowe:
#     - Energia: tylko z poziomu D3
#     - Odchylenia standardowe: wszystkie poziomy D1–D4
#     """
#     if np.all(signal == signal[0]):  # sygnał stały
#         return {'w_energy_d3': np.nan} | {f'w_std_d{i}': np.nan for i in range(1, level + 1)}
    
#     try:
#         coeffs = pywt.wavedec(signal, wavelet, level=level)
#         features = {}

#         for i, c in enumerate(coeffs[1:], 1):
#             std = np.std(c)
#             if np.isfinite(std) and std > 0:
#                 features[f'w_std_d{i}'] = std
#                 if i == 3:
#                     features[f'w_energy_d3'] = np.sum(np.square(c))
#             else:
#                 features[f'w_std_d{i}'] = np.nan
#                 if i == 3:
#                     features[f'w_energy_d3'] = np.nan

#         return features
#     except Exception:
#         return {'w_energy_d3': np.nan} | {f'w_std_d{i}': np.nan for i in range(1, level + 1)}

def wavelet_features(signal, wavelet='db4', level=4):
    """
    Oblicza wybrane cechy falkowe:
    - Energia: tylko z poziomu D3
    - Odchylenie standardowe: tylko z D1 (lub można pominąć całkiem)
    """
    if np.all(signal == signal[0]):  # sygnał stały
        return {
            'w_energy_d3': np.nan,
            'w_std_d1': np.nan 
        }

    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = {}
        for i, c in enumerate(coeffs[1:], 1):
            std = np.std(c)
            if i == 3:
                features['w_energy_d3'] = np.sum(np.square(c)) if np.isfinite(std) and std > 0 else np.nan
            if i == 1:
                features['w_std_d1'] = std if np.isfinite(std) and std > 0 else np.nan
        return features
    except Exception:
        return {
            'w_energy_d3': np.nan,
            'w_std_d1': np.nan 
        }


def entropy_features(signal):
    """
    Oblicza entropię próbkowania i entropię permutacji dla sygnału.
    - Zwraca NaN jeśli sygnał jest zbyt krótki, zawiera NaN lub jest stały
    
    Computes sample entropy and permutation entropy for a signal.
    - Returns NaN if signal is too short, contains NaN or is constant
    """
    try:
        signal = np.asarray(signal)
        if len(signal) < 20:
            raise ValueError("Zbyt krótki sygnał / Signal too short")
        if not np.all(np.isfinite(signal)):
            raise ValueError("NaN lub inf w sygnale / NaN or inf in signal")
        if np.std(signal) == 0:
            raise ValueError("Sygnał stały / Constant signal")

        return {
            'sample_entropy': ant.sample_entropy(signal),
            'perm_entropy': ant.perm_entropy(signal, normalize=True),
        }
    except Exception as e:
        return {
            'sample_entropy': np.nan,
            'perm_entropy': np.nan,
        }

def signal_relationships(df_window):
    """
    Oblicza relacje między sygnałami HR i SAO2 w oknie czasowym.
    - Korelację między HR i SAO2
    - Opóźnienie między HR i SAO2 na podstawie korelacji krzyżowej
    
    Computes relationships between HR and SAO2 signals in a time window.
    - Correlation between HR and SAO2
    - Lag between HR and SAO2 based on cross-correlation
    """
    rel = {'hr_sao2_corr': np.nan, 'hr_sao2_lag': np.nan}
    if len(df_window) < 2:
        return rel

    try:
        hr = df_window['HR'].to_numpy()
        sao2 = df_window['SAO2'].to_numpy()

        if np.std(hr) == 0 or not np.all(np.isfinite(hr)):
            return rel

        if np.std(sao2) == 0 or not np.all(np.isfinite(sao2)):
            return rel

        rel['hr_sao2_corr'] = np.corrcoef(hr, sao2)[0, 1]

        hr_norm = hr - np.mean(hr)
        sao2_norm = sao2 - np.mean(sao2)
        cross_corr = np.correlate(hr_norm, sao2_norm, mode='full')
        rel['hr_sao2_lag'] = np.argmax(cross_corr) - (len(hr) - 1)

    except Exception:
        pass

    return rel

def extract_features_in_sliding_windows(df, window_sec=30, step_sec=10):
    """
    Ekstrakcja cech w przesuwających się oknach czasowych.
    - Dla każdego okna oblicza statystyki, cechy falkowe, entropię i relacje między sygnałami
    - Domyślne okno: 30 sekund, krok: 10 sekund
    
    Extracts features in sliding time windows.
    - For each window computes statistics, wavelet features, entropy and signal relationships
    - Default window: 30 seconds, step: 10 seconds
    """
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)
    features_list = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        if window.empty:
            continue

        feat = {
            'start_time': window['TIMESTAMP'].iloc[0],
            'end_time': window['TIMESTAMP'].iloc[-1],
            'sao2_mean': window['SAO2'].mean(),
            'sao2_min': window['SAO2'].min(),
            'sao2_max': window['SAO2'].max(),
            'sao2_std': window['SAO2'].std(),
            'sao2_drop': window['SAO2'].max() - window['SAO2'].min(),
            'hr_mean': window['HR'].mean(),
            'hr_min': window['HR'].min(),
            'hr_max': window['HR'].max(),
            'hr_std': window['HR'].std(),
            'hr_delta': window['HR'].iloc[-1] - window['HR'].iloc[0],
            'sleep_stage_mode': window['Sleep_Stage_Coded'].mode().iloc[0] if not window['Sleep_Stage_Coded'].mode().empty else -1,
            'obstructive_apnea': int(window['Obstructive_Apnea'].max() == 1),
            'central_apnea': int(window['Central_Apnea'].max() == 1),
            'hr_slope_mean': np.mean(np.gradient(window['HR'])),
            'sao2_slope_mean': np.mean(np.gradient(window['SAO2'])),
            'hr_range': window['HR'].max() - window['HR'].min(),
            'hr_jump': np.abs(window['HR'].iloc[-1] - window['HR'].iloc[0])
        }

        for signal_name in ['SAO2', 'HR']:
            signal = window[signal_name].values
            feat.update({f'{signal_name.lower()}_' + k: v for k, v in wavelet_features(signal).items()})
            feat.update({f'{signal_name.lower()}_' + k: v for k, v in entropy_features(signal).items()})

        feat.update(signal_relationships(window))
        features_list.append(feat)

    return pd.DataFrame(features_list)

def balance_classes(df, target_col='Apnea', pos_ratio=0.3):
    """
    Balansuje klasy przez undersampling klasy 0.
    - pos_ratio - pożądany udział klasy 1 w zbiorze (np. 0.3 = 30%)
    - Zwraca zbalansowany DataFrame
    
    Balances classes by undersampling class 0.
    - pos_ratio - desired ratio of class 1 in dataset (e.g. 0.3 = 30%)
    - Returns balanced DataFrame
    """
    pos_df = df[df[target_col] == 1]
    neg_df = df[df[target_col] == 0]

    desired_neg_count = int(len(pos_df) * (1 - pos_ratio) / pos_ratio)
    neg_df_sampled = neg_df.sample(n=min(desired_neg_count, len(neg_df)), random_state=42)

    balanced_df = pd.concat([pos_df, neg_df_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def process_file(file_path):
    """
    Przetwarza pojedynczy plik CSV:
    - Wczytuje plik
    - Czyści i interpoluje dane
    - Ekstrahuje cechy
    - Dodaje identyfikator pacjenta
    
    Processes single CSV file:
    - Loads file
    - Cleans and interpolates data
    - Extracts features
    - Adds patient identifier
    """
    try:
        df = pd.read_csv(file_path, usecols=lambda x: x in RELEVANT_COLUMNS)
        df = clean_and_interpolate(df)
        feats = extract_features_in_sliding_windows(df)
        feats['patient_id'] = Path(file_path).stem.split("_")[0]
        return feats
    except Exception as e:
        logger.error(f"Błąd w pliku {file_path}: {e} / Error in file {file_path}: {e}")
        return pd.DataFrame()

def split_data(df):
    """
    Dzieli dane na zbiory treningowy, walidacyjny i testowy.
    - Używa kolumny 'source' do określenia przynależności
    - Zwraca słownik z podzielonymi danymi
    
    Splits data into training, validation and test sets.
    - Uses 'source' column to determine set membership
    - Returns dictionary with split data
    """
    features = df.drop(columns=['Apnea', 'patient_id', 'source', 'start_time', 'end_time'])
    labels = df['Apnea']

    X_train = features[df['source'] == 'train']
    y_train = labels[df['source'] == 'train']

    X_val = features[df['source'] == 'val']
    y_val = labels[df['source'] == 'val']

    X_test = features[df['source'] == 'test']
    y_test = labels[df['source'] == 'test']

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }

# ------------------- MAIN -------------------
if __name__ == "__main__":
    start_time = time()
    logger.info("🔁 Rozpoczęcie przetwarzania... / Starting processing...")

    SUBDIRS = ['train', 'val', 'test']
    file_paths = [p for subdir in SUBDIRS for p in (DATA_DIR / subdir).glob("*.csv")]
    total_files = len(file_paths)

    results = []
    with ProcessPoolExecutor(max_workers=N_CORE) as executor:
        for i, result in enumerate(executor.map(process_file, file_paths), 1):
            results.append(result)
            logger.info(f"Postęp: {i}/{total_files} ({i/total_files:.1%}) zakończono / Progress: {i}/{total_files} ({i/total_files:.1%}) done")

    full_data = pd.concat([df for df in results if not df.empty], ignore_index=True)
    full_data['source'] = np.concatenate([
        [subdir] * len(df)
        for subdir, df in zip(
            np.repeat(SUBDIRS, [len(list((DATA_DIR/subdir).glob('*.csv'))) for subdir in SUBDIRS]),
            results
        )
    ])
    logger.info("Rozmiar danych przed balansowaniem: %s / Data size before balancing: %s", full_data.shape[0], full_data.shape[1])


    full_data['Apnea'] = (full_data['obstructive_apnea'] | full_data['central_apnea']).astype(int)
    full_data = full_data.drop(columns=['obstructive_apnea', 'central_apnea'], errors='ignore')

    balanced_data = balance_classes(full_data, target_col='Apnea', pos_ratio=0.3)

    logger.info("Rozkład klas po balansowaniu: / Class distribution after balancing:")
    logger.info(balanced_data['Apnea'].value_counts(normalize=True))

    split_dict = split_data(balanced_data)
    pd.to_pickle(split_dict, OUTPUT_FILE)
    logger.info(f"✅ Dane zapisane do: {OUTPUT_FILE} / Data saved to: {OUTPUT_FILE}")

    logger.info(f"🏁 Przetwarzanie zakończone! Czas: {(time() - start_time) / 60:.1f} min / Processing completed! Time: {(time() - start_time) / 60:.1f} min")
