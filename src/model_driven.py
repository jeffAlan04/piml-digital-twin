import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BETA_TRUE = np.array([
    0.001, # beta1
    0.1, # beta2
    0.1, # beta3
    0.002, # beta4
    0.1, # beta5
    0.001, # beta6
    1.0, # beta7
    0.5, # beta8
    0.1, # beta9
    15.0 # T_out
])

def make_commands(sim):
    timestamp = sim['timestamp'].values.astype(float) # Estrae il timestamp
    heater = sim['heater_state'].values.astype(float) # Estrae lo stato dell'heater
    cooler = sim['cooler_level'].values.astype(float) # Estrae il livello del cooler
    last = len(timestamp) - 1 # Indice massimo valido

    def commands(time):
        active_row = int(np.clip(np.searchsorted(timestamp, time, side = 'right') - 1, 0, last))
        return heater[active_row], cooler[active_row]

    return commands

def deriv_beta(t, y, beta, commands):
    T_A, T_B, H_A, H_B = y # Vettore di stato
    b1, b2, b3, b4, b5, b6, b7, b8, b9, T_out = beta

    heater_state, cooler_level = commands(t)
    cooling = b9 * cooler_level 

    dT_A = b1 * (b2 * (T_out - T_A) + H_A - cooling + b3 * (T_B - T_A)) # Differenza di temperatura al secondo in questo istante nella stanza A
    dT_B = b4 * (b5 * (T_out - T_B) + H_B + b3 * (T_A - T_B)) # Differenza di temperatura al secondo in questo istante nella stanza B
    dH_A = b6 * (b7 * heater_state - H_A) 
    dH_B = b6 * (b8 * heater_state - H_B) 
    
    return [dT_A, dT_B, dH_A, dH_B]

def simulate(sim, beta, y0 = None, max_step = 60.0):
    t_grid = sim['timestamp'].values.astype(float) # Griglia temporale campionata ogni 60 secondi

    # Stato di partenza
    if y0 is None:
        # Temperature misurate al primo campione e riscaldatori spenti
        y0 = [sim['T_A'].iloc[0], sim['T_B'].iloc[0], 0.0, 0.0]

    return solve_ivp(
        deriv_beta, 
        (t_grid[0], t_grid[-1]), 
        y0, 
        t_eval = t_grid, 
        args = (beta, make_commands(sim)), 
        method = 'RK45', 
        max_step = max_step
    )

if __name__ == '__main__':
    dataset = pd.read_csv('data/dataset.csv') # Carica le simulazioni
    sim = dataset[dataset['simulation_id'] == 0].reset_index(drop = True) # Prende una simulazione

    sol = simulate(sim, BETA_TRUE)
    res = np.concatenate(
        [sol.y[0] - sim['T_A'].values, 
        sol.y[1] - sim['T_B'].values]
    )
    print(f'MAE {np.abs(res).mean():.4f}', f'RMSE {np.sqrt((res ** 2).mean()):.4f}')
    
    hours = sim['timestamp'].values / 3600

    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True,gridspec_kw = {'height_ratios': [2, 2, 2, 2, 1, 1]})

    axes[0].plot(hours, sim['T_A'].values, 'b-', linewidth=1.5, label='Real')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Sensor A Measured')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, sol.y[0], 'r-', linewidth=1.5, label='Predicted')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Sensor A Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hours, sim['T_B'].values, 'b-', linewidth=1.5, label='Real')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_title('Sensor B Measured')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(hours, sol.y[1], 'r-', linewidth=1.5, label='Predicted')
    axes[3].set_ylabel('Temperature (°C)')
    axes[3].set_title('Sensor B Predicted')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].step(hours, sim['heater_state'], 'r-', linewidth=1.5, where='post')
    axes[4].set_ylabel('On/Off')
    axes[4].set_title('Heater')
    axes[4].set_yticks([0, 1])
    axes[4].grid(True, alpha=0.3)

    axes[5].step(hours, sim['cooler_level'], 'b-', linewidth=1.5, where='post')
    axes[5].set_ylim(-0.5, 9.5)
    axes[5].set_yticks(range(0, 10))
    axes[5].set_ylabel('Level')
    axes[5].set_title('Cooler')
    axes[5].set_xlabel('Hours')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
