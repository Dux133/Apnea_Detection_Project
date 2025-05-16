


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
    
         'sao2_shannon_entropy': 0.0,
        'sao2_sample_entropy': 0.0

        
        # Nowe cechy entropijne
        features.update({
            'sao2_shannon_entropy': compute_shannon_entropy(sao2_segment),
            'sao2_sample_entropy': compute_sample_entropy(sao2_segment)
        })