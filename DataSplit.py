import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from scipy.stats import ks_2samp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# === KONFIGURACJA / CONFIGURATION ===
INPUT_FOLDER = r'D:\Apnea_Detection_Project\data'
OUTPUT_SUBFOLDERS = {
    'train': os.path.join(INPUT_FOLDER, 'train'),
    'val': os.path.join(INPUT_FOLDER, 'val'),
    'test': os.path.join(INPUT_FOLDER, 'test')
}
RANDOM_STATE = 42  # Ziarno losowości / Random seed
TEST_SIZE = 0.15   # Rozmiar zbioru testowego / Test set size
VAL_SIZE = 0.15    # Rozmiar zbioru walidacyjnego / Validation set size

# === FUNKCJE POMOCNICZE / HELPER FUNCTIONS ===
def extract_patient_id(filename):
    """Wyodrębnij ID pacjenta z nazwy pliku / Extract patient ID from filename"""
    return filename.split('_')[0]

def process_file(filename, input_folder):
    """Przetwarzaj pojedynczy plik CSV i wylicz metryki / Process single CSV file and calculate metrics"""
    try:
        filepath = os.path.join(input_folder, filename)
        # Wczytaj tylko kolumny bezdechu  / Load only apnea-related columns
        df = pd.read_csv(filepath, usecols=['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events'])
        
        # Oblicz występowanie bezdechu w każdej próbce / Calculate apnea presence in each sample
        apnea_present = (df['Obstructive_Apnea'] > 0) | (df['Central_Apnea'] > 0) | (df['Hypopnea'] > 0) | (df['Multiple_Events'] > 0)
        labels = apnea_present.astype(int)
        
        return {
            'filename': filename,
            'patient_id': extract_patient_id(filename),
            'total_samples': len(labels),
            'apnea_samples': labels.sum(),
            'apnea_ratio': labels.mean()
        }
    except Exception as e:
        print(f"Błąd przetwarzania / Error processing {filename}: {str(e)}")
        return None

def check_distributions(train_df, test_df, features):
    """Porównaj rozkłady cech między zbiorami i wyświetl pełne wyniki testu KS
    Compare feature distributions between sets and display full KS test results"""
    
    print("\nTest Kołmogorowa-Smirnowa między zbiorami / Kolmogorov-Smirnov test between sets:")
    print("---------------------------------------------------------------")
    
    for feature in features:
        # Oblicza statystykę KS / Calculate KS statistic
        _, p_value = ks_2samp(train_df[feature], test_df[feature])
        
        # Określ status w dwóch językach / Determine status in both languages
        status_pl = "RÓŻNIĄ SIĘ istotnie" if p_value < 0.05 else "są podobne"
        status_en = "SIGNIFICANTLY DIFFERENT" if p_value < 0.05 else "are similar"
        
        # Sformatuj wyniki / Format results
        print(
            f"[{feature}] "
            f"p-value: {p_value:.4f} | "
            f"Status PL: {status_pl} | "
            f"Status EN: {status_en}"
        )
        
        # Dodatkowe ostrzeżenie dla istotnych różnic / Additional warning for significant differences
        if p_value < 0.05:
            print(f"   Uwaga! Potencjalny problem z rozkładem '{feature}' / Warning! Potential distribution issue for '{feature}'")
    
    print("---------------------------------------------------------------")
    print("Interpretacja (PL): p-value < 0.05 - istotne różnice w rozkładach")
    print("Interpretation (EN): p-value < 0.05 - significant distribution differences\n")
# === GŁÓWNY ALGORYTM / MAIN ALGORITHM ===
def main():
    print("Wczytywanie i przetwarzanie plików... / Loading and processing files...")
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    
    # Przetwarzanie plików wielowątkowo / Multithreaded file processing
    with ThreadPoolExecutor() as executor:
        process_fn = partial(process_file, input_folder=INPUT_FOLDER)
        results = list(executor.map(process_fn, csv_files))
    
    # Tworzenie DataFrame z metadanymi / Create metadata DataFrame
    meta_df = pd.DataFrame([r for r in results if r is not None])
    patients_df = meta_df.groupby('patient_id').agg({
        'total_samples': 'sum',
        'apnea_samples': 'sum',
        'apnea_ratio': 'mean',
        'filename': list
    }).reset_index()

    # Tworzenie binów dla stratyfikacji / Create bins for stratification
    patients_df['stratify_bin'] = pd.qcut(patients_df['apnea_ratio'], q=5, duplicates='drop')

    print("\nPodział danych z zachowaniem grup pacjentów... / Splitting data with patient grouping...")
    
    # Pierwszy podział: train + temp vs test / First split: train + temp vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(patients_df, groups=patients_df['patient_id']))
    
    train_val_df = patients_df.iloc[train_val_idx]
    test_df = patients_df.iloc[test_idx]

    # Drugi podział: train vs val / Second split: train vs val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(train_val_df, train_val_df['stratify_bin']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # Sprawdzenie rozkładów / Distribution checks
    print("\nWeryfikacja rozkładów: / Distribution verification:")
    check_distributions(train_df, test_df, ['apnea_ratio', 'total_samples'])
    check_distributions(train_df, val_df, ['apnea_ratio', 'total_samples'])

    # Sprawdzenie unikalności pacjentów / Patient uniqueness check
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])
    
    assert not (train_patients & val_patients), "Konflikt pacjentów train-val! / Train-val patient conflict!"
    assert not (train_patients & test_patients), "Konflikt pacjentów train-test! / Train-test patient conflict!"
    assert not (val_patients & test_patients), "Konflikt pacjentów val-test! / Val-test patient conflict!"
    print("Brak konfliktów między zbiorami / No conflicts between sets")

    # Przenoszenie plików / File organization
    print("\nOrganizacja plików... / Organizing files...")
    for split, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        split_folder = OUTPUT_SUBFOLDERS[split]
        os.makedirs(split_folder, exist_ok=True)
        
        for filename in np.concatenate(df['filename'].values):
            src = os.path.join(INPUT_FOLDER, filename)
            dst = os.path.join(split_folder, filename)
            shutil.move(src, dst)

    # Raport końcowy / Final report
    print("\nPodsumowanie podziału: / Split summary:")
    for split, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        print(f"\n=== {split.upper()} ===")
        print(f"Pacjenci / Patients: {len(df)}")
        print(f"Pliki / Files: {sum(len(files) for files in df['filename'])}")
        print(f"Próbki / Samples: {df['total_samples'].sum():,}")
        print(f"Średni współczynnik bezdechu / Mean apnea ratio: {df['apnea_ratio'].mean():.2%}")

if __name__ == "__main__":
    main()
