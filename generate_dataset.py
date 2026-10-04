import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from room_simulator import deriv, get_heater_ref, get_cooler_power

def generate_random_actions(duration, min_interval=300, max_interval=1500, seed=None):
    rng = np.random.default_rng(seed) # Generatore di numeri casuali

    # Situazione iniziale
    actions = []
    t = 0

    heater = rng.choice([0, 1]) # Generatore casuale per l'heater (acceso o spento)
    cooler = rng.integers(0, 10) # Generatore casuale per il cooler (tra 0 e 9)
    actions.append((t, heater, int(cooler)))

    while t < duration:
        dt = rng.integers(min_interval, max_interval + 1) # Genera l'intervallo di tempo fino alla prossima azione
        t += dt

        if t >= duration:
            break
        
        heater = rng.choice([0, 1])
        cooler = rng.integers(0, 10)
        actions.append((t, heater, int(cooler)))
    
    return actions

def run_simulation(actions, duration, T_A_initial, T_B_initial):
    y0 = [T_A_initial, T_B_initial, 0.0, 0.0] # Stato inziale

    sol = solve_ivp(
        fun = deriv,
        t_span = (0, duration),
        y0 = y0,
        method = 'RK45',
        args = (actions,),
        dense_output = True,
        max_step = 10.0
    )

    return sol

def sample_on_grid(sol, actions, duration, dt = 10, sigma = 0.1):
    t_grid = np.arange(0, duration + 1, dt)

    states = sol.sol(t_grid)
    T_A_real = states[0]
    T_B_real = states[1]

    # Aggiunta di rumore gaussiano
    T_A_noisy = T_A_real + np.random.normal(0, sigma, size=len(t_grid))
    T_B_noisy = T_B_real + np.random.normal(0, sigma, size=len(t_grid))

    heater_states = np.zeros(len(t_grid), dtype = int)
    cooler_levels = np.zeros(len(t_grid), dtype = int)

    # Determina lo stato degli attuatori per ogni istante della griglia
    for i, t in enumerate(t_grid):
        active_action = actions[0]
        for action in actions:
            if action[0] <= t:
                active_action = action
            else:
                break
        heater_states[i] = active_action[1]
        cooler_levels[i] = active_action[2]

    # Risultati inseriti in un dataframe
    df = pd.DataFrame({
        'timestamp': t_grid,
        'T_A': T_A_noisy,
        'T_B': T_B_noisy,
        'heater_state': heater_states,
        'cooler_level': cooler_levels,
    })

    return df

def generate_dataset(n_simulations = 50, duration = 15000, dt = 10, sigma = 0.1, output_path = 'data/dataset.csv',):
    all_dfs = []

    # Loop su 50 simulazioni
    for sim_id in range(n_simulations):

        # Le temperature sono sempre casuali e comprese fra 15 e 20 gradi
        T_A_initial = np.random.uniform(15, 25)
        T_B_initial = np.random.uniform(15, 25)

        # Genera azioni casuali
        actions = generate_random_actions(duration)

        # Esegue le ODE
        sol = run_simulation(actions, duration, T_A_initial, T_B_initial)

        # Campiona sulla griglia e aggiunge rumore
        df = sample_on_grid(sol, actions, duration, dt, sigma)
        
        # Aggiunge l'id della simulazione
        df.insert(0, 'simulation_id', sim_id)

        all_dfs.append(df)
    
    dataset = pd.concat(all_dfs, ignore_index = True)

    dataset.to_csv(output_path, index = False)
    print(f"\nDataset salvato in {output_path}")
    print(f"\nSimulazioni: {n_simulations}")

    return dataset

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok = True)
    dataset = generate_dataset()