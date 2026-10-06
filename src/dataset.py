import numpy as np
import pandas as pd

# Carica il CSV e restituisce una lista di dataframe
def load_dataset(csv_path = 'data/dataset.csv'):
    df = pd.read_csv(csv_path)

    simulations = []
    for sim_id, group in df.groupby('simulation_id'):
        simulations.append(group.reset_index(drop = True))

    return simulations

# Divisione delle simulazioni in train, validation e test set
def split_simulations(simulations, train_ratio = 0.7, val_ratio = 0.15):
    n = len(simulations)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    indices = np.random.permutation(n) # Mescola le simulazioni

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_sims = [simulations[i] for i in train_idx]
    val_sims = [simulations[i] for i in val_idx]
    test_sims = [simulations[i] for i in test_idx]

    return train_sims, val_sims, test_sims

# Calcola la media e la deviazione standard delle temperature sul training set
def compute_stats(simulations):
    all_T_A = []
    all_T_B = []

    for sim in simulations:
        all_T_A.append(sim['T_A'].values)
        all_T_B.append(sim['T_B'].values)
    
    all_T_A = np.concatenate(all_T_A)
    all_T_B = np.concatenate(all_T_B)
    all_temps = np.concatenate([all_T_A, all_T_B])

    mean = all_temps.mean()
    std = all_temps.std()

    return mean, std

def normalize_simulations(simulations, mean, std):
    normalized = []

    # Applica z-score alle temperature
    for sim in simulations:
        sim = sim.copy() # Copia per non modificare il dataframe originale
        sim['T_A'] = (sim['T_A'] - mean) / std
        sim['T_B'] = (sim['T_B'] - mean) / std
        normalized.append(sim)

    return normalized

# Estrae da ogni simulazione tutte le finestre consecutive di window_size righe
def make_sliding_windows(simulations, window_size):

    windows = []

    for sim in simulations:
        values = sim[['T_A', 'T_B', 'heater_state', 'cooler_level']].values
        
        n_rows = len(values)

        n_windows = n_rows - window_size + 1 # Numero di finestre che entrano nella simulazione

        # La finestra scorre di una riga alla volta senza mai attraversare due simulazioni
        for i in range(n_windows):
            window = values[i : i + window_size]
            windows.append(window)
    
    return windows

# Costruisce input (X) e target (Y) a partire dalle finestre
def map_XY(windows, prediction_steps):
    X_list = []
    Y_list = []

    for window in windows:

        # Stato all'istante iniziale della finestra
        first = window[0]
        T_A_0 = first[0]
        T_B_0 = first[1]
        heater_0 = first[2]
        cooler_0 = first[3]

        # [tipo (0 = sensore, 1 = heater, 2 = cooler), 0, valore]
        sensor_A = [0, 0, T_A_0]
        sensor_B = [0, 0, T_B_0]
        heater = [1, 0, heater_0]
        cooler = [2, 0, cooler_0]

        x = np.array([sensor_A, sensor_B, heater, cooler]) # Shape (4, 3)
        X_list.append(x)

        # Temperature (T_A, T_B) degli istanti successivi al primo
        future_temps = window[1:, :2]

        y = future_temps.T # Shape (2, window_size - 1)
        Y_list.append(y)

    X = np.array(X_list)
    Y = np.array(Y_list)

    return X,Y



if __name__ == '__main__':
    # Caricamento
    simulations = load_dataset()
    print(f"Caricate {len(simulations)} simulazioni")

    # Split
    train_sims, val_sims, test_sims = split_simulations(simulations)
    print(f"Train: {len(train_sims)}, Val: {len(val_sims)}, Test: {len(test_sims)}")
    
    # Statistiche sul training set
    mean, std = compute_stats(train_sims)
    print(f"Media temperature: {mean:.4f}, Std: {std:.4f}")

    # Normalizzazione di tutti i set
    train_norm = normalize_simulations(train_sims, mean, std)
    val_norm = normalize_simulations(val_sims, mean, std)
    test_norm = normalize_simulations(test_sims, mean, std)

    # Sliding windows
    prediction_steps = 10
    window_size = prediction_steps + 1

    train_windows = make_sliding_windows(train_norm, window_size)
    val_windows = make_sliding_windows(val_norm, window_size)
    test_windows = make_sliding_windows(test_norm, window_size)
    print(f"Finestre Train: {len(train_windows)}, Finestre Val: {len(val_windows)}, Finestre Test: {len(test_windows)}")

    # Creazione X e Y
    X_train, Y_train = map_XY(train_windows, prediction_steps)
    X_val, Y_val = map_XY(val_windows, prediction_steps)
    X_test, Y_test = map_XY(test_windows, prediction_steps)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")



