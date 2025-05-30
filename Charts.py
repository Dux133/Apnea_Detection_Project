import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
from concurrent.futures import ProcessPoolExecutor
import os
import re

# Konfiguracja ścieżek / Path configuration
BASE_DIR = r'D:/Apnea_Detection_Project/data'
INPUT_SUBDIRS = ['train', 'test', 'val']
OUTPUT_DIR = os.path.join(BASE_DIR, 'wykresy', 'HR_SAO2')
PARTICIPANT_INFO_PATH = os.path.join(BASE_DIR, 'participant_info.csv')
EVENT_TYPES = ['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea','Multiple_Events']
PARAMETERS = ['HR', 'SAO2']
FS = 100  # Częstość próbkowania / Sampling frequency
WINDOW_SIZE_SECONDS = 30  # Rozmiar okna w sekundach / Window size in seconds 
START = 200  # Początek zakresu czasu w minutach / Start time in minutes
END = 320    # Koniec zakresu czasu w minutach / End time in minutes

# Kolory dla różnych typów zdarzeń / Colors for different event types
EVENT_COLORS = {
    'Obstructive_Apnea': 'grey',
    'Central_Apnea': 'orange',
    'Hypopnea': '#98FB98',
    'Multiple_Events': '#B19CD9'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_percentage_value(value):
    """
    Konwertuje wartości procentowe z formatu '95%' na 95.0.
    Obsługuje wartości brakujące (NaN) i tekstowe.
    
    Converts percentage values from '95%' format to 95.0.
    Handles missing values (NaN) and text values.
    """
    if pd.isna(value) or value == 'None':
        return np.nan
    if isinstance(value, str):
        return float(value.replace('%', '').strip())
    return float(value)

def load_participant_info():
    """
    Wczytuje i czyści dane pacjentów z pliku CSV.
    - Konwertuje kolumny numeryczne
    - Standaryzuje nazwy kolumn
    
    Loads and cleans patient data from CSV file.
    - Converts numeric columns
    - Standardizes column names
    """
    try:
        df = pd.read_csv(PARTICIPANT_INFO_PATH)
        
        # Konwersja kolumn numerycznych / Numeric columns conversion
        num_cols = ['AGE', 'BMI', 'OAHI', 'AHI', 'Mean_HR', 'Arousal Index']
        for col in num_cols:
            if col in df.columns:
                if col == 'Mean_HR':
                    df[col] = df[col].apply(clean_percentage_value)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standaryzacja nazw kolumn / Column names standardization
        df.columns = df.columns.str.replace(' ', '_')
        
        return df
    except Exception as e:
        print(f"❌ Błąd przy wczytywaniu pliku z danymi pacjentów: {e} / Error loading patient data file: {e}")
        return pd.DataFrame()

def get_patient_info(sid, participant_df):
    """
    Pobiera i formatuje informacje o pacjencie na podstawie ID.
    - Zwraca słownik z danymi pacjenta
    - Skraca zbyt długie opisy zaburzeń snu
    
    Retrieves and formats patient information based on ID.
    - Returns dictionary with patient data
    - Shortens too long sleep disorder descriptions
    """
    if participant_df.empty:
        return {}
    
    try:
        patient_data = participant_df[participant_df['SID'] == sid].iloc[0].to_dict()
    except IndexError:
        print(f"⚠️ Brak danych dla pacjenta {sid} w pliku informacyjnym / No data for patient {sid} in info file")
        return {}
    except Exception as e:
        print(f"❌ Błąd przy wczytywaniu danych dla {sid}: {e} / Error loading data for {sid}: {e}")
        return {}
    
    def format_value(val, is_percent=False):
        """Pomocnicza funkcja formatująca wartości / Helper function for value formatting"""
        if pd.isna(val) or val == 'None':
            return 'NA'
        try:
            if is_percent:
                return f"{float(val):.1f}%"
            return f"{float(val):.1f}"
        except (ValueError, TypeError):
            return str(val)
    
    # Formatowanie informacji / Information formatting
    info = {
        'BMI': format_value(patient_data.get('BMI')),
        'AGE': format_value(patient_data.get('AGE')),
        'SEX': patient_data.get('GENDER', patient_data.get('SEX', 'NA')),
        'OAHI': format_value(patient_data.get('OAHI')),
        'AHI': format_value(patient_data.get('AHI')),
        'Mean_SaO2': format_value(patient_data.get('Mean_SaO2'), is_percent=True),
        'Arousal_Index': format_value(patient_data.get('Arousal_Index')),
        'Sleep_Disorders': patient_data.get('Sleep_Disorders', 'NA')
    }
    
    # Skrócenie opisów / Description shortening
    if len(str(info['Sleep_Disorders'])) > 50:
        info['Sleep_Disorders'] = str(info['Sleep_Disorders'])[:47] + "..."
    
    return info

def generate_and_save_plot(df, events_by_type, sid, patient_info):
    """
    Generuje i zapisuje wykres z danymi pacjenta.
    - Tworzy wykresy SAO2 i HR
    - Zaznacza zdarzenia bezdechu
    - Dodaje informacje o pacjencie
    
    Generates and saves patient data plot.
    - Creates SAO2 and HR plots
    - Marks apnea events
    - Adds patient information
    """
    df = df[(df['Time_min'] >= START) & (df['Time_min'] <= END)].copy()
    
    fig, ax1 = plt.subplots(figsize=(20, 10))
    
    # Wykres SAO2 / SAO2 plot
    if 'SAO2' in df.columns:
        ax1.plot(df['Time_min'], df['SAO2'], 'b-', alpha=0.8, linewidth=0.5, zorder=1, label='SpO₂')
        ax1.set_ylabel('SpO₂ (%)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0.1, 1.3)  # Skalowanie osi / Axis scaling
        ax1.grid(True, alpha=0.3, linestyle='--')
    else:
        print(f"⚠️ Brak danych SAO2 dla pacjenta {sid} / Missing SAO2 data for patient {sid}")
    
    # Wykres HR / HR plot
    if 'HR' in df.columns:
        ax2 = ax1.twinx()
        df['is_apnea'] = ((df['Obstructive_Apnea'] == 1) | 
                          (df['Central_Apnea'] == 1) | 
                          (df['Hypopnea'] == 1)).astype(int)

        # Oblicz zakres zmienności HR / Calculate HR variability range
        window_size = WINDOW_SIZE_SECONDS * FS
        df['HR_rolling_max'] = df['HR'].rolling(window=window_size, center=True, min_periods=1).max()
        df['HR_rolling_min'] = df['HR'].rolling(window=window_size, center=True, min_periods=1).min()
        df['HR_range'] = df['HR_rolling_max'] - df['HR_rolling_min']
        df['HR_range_apnea'] = df['HR_range'].where(df['is_apnea'] == 1)
        df['HR_range_no_apnea'] = df['HR_range'].where(df['is_apnea'] == 0)

        # Wykres surowego HR / Raw HR plot
        ax2.plot(df['Time_min'], df['HR'], 'r-', alpha=0.8, linewidth=0.5, zorder=1, label='HR')
        
        # Wypełnienie zakresu zmienności / Variability range fill
        ax2.fill_between(df['Time_min'], 0, df['HR_range_apnea'], color='red', alpha=0.2)
        ax2.fill_between(df['Time_min'], 0, df['HR_range_no_apnea'], color='green', alpha=0.2)

        ax2.set_ylabel('HR (bpm)', color='red', fontsize=12)
        ax2.set_ylim(0, 150)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(True, alpha=0.3, linestyle='--')
    else:
        print(f"⚠️ Brak danych HR dla pacjenta {sid} / Missing HR data for patient {sid}")

    # Zdarzenia bezdechu / Apnea events
    duration = 0.01 / 60
    for event_type, events in events_by_type.items():
        if not events.empty:
            rects = [Rectangle((t, 0), duration, 1) for t in events]
            pc = PatchCollection(
                rects,
                facecolor=EVENT_COLORS[event_type],
                alpha=0.5,
                edgecolor='none',
                transform=ax1.get_xaxis_transform()
            )
            ax1.add_collection(pc)

    # Legenda zdarzeń / Event legend
    legend_labels = [f"{et} (n={len(events)})" for et, events in events_by_type.items()]
    legend_patches = [Patch(color=EVENT_COLORS[et], label=label, alpha=1) 
                     for et, label in zip(EVENT_TYPES, legend_labels)]
    
    leg1 = ax1.legend(
        handles=legend_patches,
        loc='upper left',
        title="Apneas",
        title_fontsize=10,
        fontsize=10,
        framealpha=1.0,
        facecolor='white'
    )
    ax1.add_artist(leg1)

    # Panel informacji o pacjencie / Patient info panel
    patient_text = (
        f"Dane pacjenta {sid}:\n"
        f"• Wiek: {patient_info.get('AGE', 'NA')} lat\n"
        f"• Płeć: {patient_info.get('SEX', 'NA')}\n"
        f"• BMI: {patient_info.get('BMI', 'NA')}\n"
        f"• OAHI: {patient_info.get('OAHI', 'NA')}\n"
        f"• AHI: {patient_info.get('AHI', 'NA')}\n"
        f"• Śr. SaO₂: {patient_info.get('Mean_SaO2', 'NA')}\n"
        f"• Wskaźnik Arousal: {patient_info.get('Arousal_Index', 'NA')}\n"
    )
    
    fig.text(0.02, 0.88, patient_text, fontsize=10, va='top', ha='left', 
             bbox=dict(facecolor='white', edgecolor='none'))
    plt.title(f'Monitorowanie bezdechów sennych - pacjent {sid} / Sleep apnea monitoring - patient {sid}', 
              fontsize=16, pad=20)
    ax1.set_xlabel('Czas (minuty) / Time (minutes)', fontsize=12)
    ax1.set_xlim(df['Time_min'].min(), df['Time_min'].max())
    output_path = os.path.join(OUTPUT_DIR, f'{sid}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def process_data(file_path, sid, participant_df):
    """
    Przetwarza pojedynczy plik danych pacjenta.
    - Wczytuje dane
    - Przygotowuje informacje o zdarzeniach
    - Generuje wykres
    
    Processes single patient data file.
    - Loads data
    - Prepares event information
    - Generates plot
    """
    try:
        df = pd.read_csv(file_path, usecols=EVENT_TYPES + PARAMETERS)
        df['Time_min'] = np.arange(len(df)) / (FS * 60)
        
        events_by_type = {
            event: df.loc[df[event] == 1, 'Time_min'] 
            for event in EVENT_TYPES
        }
        
        patient_info = get_patient_info(sid, participant_df)
        generate_and_save_plot(df, events_by_type, sid, patient_info)
        print(f"✔️ Zapisano wykres: {sid}.png / Saved plot: {sid}.png")
    except Exception as e:
        print(f"❌ Błąd przy pliku {file_path}: {e} / Error processing file {file_path}: {e}")

def process_all_files():
    """
    Główna funkcja przetwarzająca wszystkie pliki z podfolderów.
    - Wczytuje dane pacjentów
    - Znajduje pliki do przetworzenia
    - Używa ProcessPoolExecutor do równoległego przetwarzania
    
    Main function processing all files from subdirectories.
    - Loads patient data
    - Finds files to process
    - Uses ProcessPoolExecutor for parallel processing
    """
    participant_df = load_participant_info()
    files = []

    for subdir in INPUT_SUBDIRS:
        full_path = os.path.join(BASE_DIR, subdir)
        subdir_files = [os.path.join(full_path, f) for f in os.listdir(full_path)
                        if re.match(r'^S\d{3}_PSG_df\.csv$', f)]
        files.extend(subdir_files)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for file_path in files:
            filename = os.path.basename(file_path)
            sid = filename.split('_')[0]
            print(f"Przetwarzanie pliku: {filename}... / Processing file: {filename}...")
            futures.append(executor.submit(process_data, file_path, sid, participant_df))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"❌ Błąd podczas przetwarzania: {e} / Processing error: {e}")

    print("Przetwarzanie zakończone. / Processing completed.")

if __name__ == '__main__':
    process_all_files()