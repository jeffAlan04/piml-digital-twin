import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from room_simulator import deriv, get_heater_ref, get_cooler_power


def generate_random_actions(duration, dt = 60, min_steps = 30, max_steps = 180, rng = None):
    rng = rng if rng is not None else np.random.default_rng()

    # Situazione iniziale
    actions = [(0, int(rng.choice([0, 1])), int(rng.integers(0, 10)))]
    t = 0

    while True:
        t += int(rng.integers(min_steps, max_steps + 1)) * dt

        if t >= duration:
            break
        actions.append((t, int(rng.choice([0, 1])), int(rng.integers(0, 10))))
    return actions

def run_simulation(actions, duration, T_A_initial, T_B_initial, dt = 60):
    t_grid = np.arange(0, duration + 1, dt)
    y0 = [T_A_initial, T_B_initial, 0.0, 0.0] # Stato inziale

    sol = solve_ivp(
        fun = deriv,
        t_span = (0, duration),
        y0 = y0,
        method = 'RK45',
        args = (actions,),
        t_eval = t_grid,
        max_step = 30.0
    )

    return sol

def sample_on_grid(sol, actions, duration, dt = 60, sigma = 0.1, rng = None):
    rng = rng if rng is not None else np.random.default_rng()
    t_grid = sol.t

    # Aggiunta di rumore gaussiano
    T_A_noisy = sol.y[0] + rng.normal(0, sigma, size=len(t_grid))
    T_B_noisy = sol.y[1] + rng.normal(0, sigma, size=len(t_grid))

    action_times = np.array([a[0] for a in actions])
    heater_states = np.array([a[1] for a in actions])
    cooler_levels = np.array([a[2] for a in actions])

    idx = np.searchsorted(action_times, t_grid, side = 'right') - 1

    # Risultati inseriti in un dataframe
    df = pd.DataFrame({
        'timestamp': t_grid,
        'T_A': T_A_noisy,
        'T_B': T_B_noisy,
        'heater_state': heater_states[idx],
        'cooler_level': cooler_levels[idx],
    })

    return df

def generate_dataset(n_simulations = 30, duration = 288000, dt = 60, sigma = 0.1,  output_path = 'data/dataset.csv', seed = 42):
    rng = np.random.default_rng(seed)
    all_dfs = []
    sim_id = 0

    for sim_id in range(n_simulations):
        T_A0 = rng.uniform(0, 30)
        T_B0 = rng.uniform(0, 30)

        actions = generate_random_actions(duration, dt, rng = rng)
        sol = run_simulation(actions, duration, T_A0, T_B0, dt)
        df = sample_on_grid(sol, actions, duration, dt, sigma, rng)
        df.insert(0, 'simulation_id', sim_id)
        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index = True)
    dataset.to_csv(output_path, index = False)
    print(f"Simulazioni: {n_simulations}, campioni ciascuna: {len(all_dfs[0])}")
    return dataset
"""
    # Loop su 50 simulazioni
    for sim_id in range(n_simulations):

        # Le temperature sono sempre casuali e comprese fra 15 e 20 gradi
        T_A_initial = rng.uniform(15, 25)
        T_B_initial = rng.uniform(15, 25)

        # Genera azioni casuali
        actions = generate_random_actions(duration, dt, rng = rng)

        # Esegue le ODE
        sol = run_simulation(actions, duration, T_A_initial, T_B_initial, dt)

        # Campiona sulla griglia e aggiunge rumore
        df = sample_on_grid(sol, actions, duration, dt, sigma, rng)
        
        # Aggiunge l'id della simulazione
        df.insert(0, 'simulation_id', sim_id)

        all_dfs.append(df)
    
    dataset = pd.concat(all_dfs, ignore_index = True)

    dataset.to_csv(output_path, index = False)
    print(f"\nDataset salvato in {output_path}")
    print(f"\nSimulazioni: {n_simulations}")
    """

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok = True)
    dataset = generate_dataset()