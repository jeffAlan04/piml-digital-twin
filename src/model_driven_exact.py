import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from dataset import split_simulations

BETA_NAMES = [f'beta{i}' for i in range(1, 10)] + ['T_out']

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

BETA7 = 1.0
ADJUSTABLE = [0, 1, 2, 3, 4, 5, 7, 8, 9] # Tutti tranne beta7

BETA_LOWER = np.array([1e-5, 1e-3, 1e-3, 1e-5, 1e-3, 1e-5, 1e-2, 1e-2, 1e-3, -20.0])
BETA_UPPER = np.array([1e-1, 10.0, 10.0, 1e-1, 10.0, 1e-1, 100., 100., 10.0, 50.0])


def make_commands(sim):
    timestamp = sim['timestamp'].values.astype(float) # Estrae il timestamp
    heater = sim['heater_state'].values.astype(float) # Estrae lo stato dell'heater
    cooler = sim['cooler_level'].values.astype(float) # Estrae il livello del cooler
    last = len(timestamp) - 1 # Indice massimo valido

    def commands(time):
        active_row = int(np.clip(np.searchsorted(timestamp, time, side = 'right') - 1, 0, last)) # Cerca l'ultima riga con timestamp <= time
        return heater[active_row], cooler[active_row]

    return commands

def deriv_beta(t, y, beta, commands):
    T_A, T_B, H_A, H_B = y # Vettore di stato
    b1, b2, b3, b4, b5, b6, b7, b8, b9, T_out = beta

    heater_state, cooler_level = commands(t) # Azioni degli attutatori all'istante t
    cooling = b9 * cooler_level

    dT_A = b1 * (b2 * (T_out - T_A) + H_A - cooling + b3 * (T_B - T_A)) # Differenza di temperatura al secondo in questo istante nella stanza A
    dT_B = b4 * (b5 * (T_out - T_B) + H_B + b3 * (T_A - T_B)) # Differenza di temperatura al secondo in questo istante nella stanza B
    dH_A = b6 * (b7 * heater_state - H_A) # Heater A con inerzia
    dH_B = b6 * (b8 * heater_state - H_B) # Heater B con inerzia


    return [dT_A, dT_B, dH_A, dH_B]

def simulate(sim, beta, y0 = None, max_step = 60.0):
    t_grid = sim['timestamp'].values.astype(float) # Griglia temporale campionata ogni 60 secondi
    commands = make_commands(sim)

    # Stato di partenza
    if y0 is None:
        # Temperature misurate al primo campione e riscaldatori spenti
        y0 = [sim['T_A'].iloc[0], sim['T_B'].iloc[0], 0.0, 0.0]

    return solve_ivp(
        deriv_beta, 
        (t_grid[0], t_grid[-1]), 
        y0, 
        t_eval = t_grid, 
        args = (beta, commands), 
        method = 'RK45', 
        max_step = max_step
    )

def residuals(beta, sims, max_step = 60.0):
    errors = []

    for sim in sims:
        # Simula la porzione di simulazione con questi parametri
        sol = simulate(sim, beta, max_step = max_step)
        if sol.y.shape[1] != len(sim):
            return np.full(2 * sum(len(s) for s in sims), 1e3)
        
        errors.append(sol.y[0] - sim['T_A'].values)
        errors.append(sol.y[1] - sim['T_B'].values)
    
    return np.concatenate(errors)

def initial_guess(sims, heater_rate = 1e-3):
    rows, dTdt_A, dTdt_B = [], [], []

    for sim in sims:
        T_A = sim['T_A'].values
        T_B = sim['T_B'].values
        heater = sim['heater_state'].values.astype(float)
        cooler = sim['cooler_level'].values.astype(float)

        # Distanza di 60 secondi fra due campioni
        step = sim['timestamp'].values[1] - sim['timestamp'].values[0]

        dTdt_A.append((T_A[2:] - T_A[:-2]) / (2 * step))
        dTdt_B.append((T_B[2:] - T_B[:-2]) / (2 * step))

        # Restituisce 30 matrici
        rows.append(np.column_stack([
            np.ones(len(T_A) - 2), # 1
            T_A[1 : -1], # Temperatura di A 
            T_B[1 : -1], # Temperatura di B
            heater[1 : -1], # Heater
            cooler[1 : - 1] # Cooler
        ]))

    predictors = np.vstack(rows)
    coef_A, *_ = np.linalg.lstsq(predictors, np.concatenate(dTdt_A), rcond = None)
    coef_B, *_ = np.linalg.lstsq(predictors, np.concatenate(dTdt_B), rcond = None)

    outside_A, self_A, from_B, heater_A, cooler_A = coef_A
    outside_B, from_A, self_B, heater_B, cooler_B = coef_B

    """
    H_A = b7 * heater
    H_B = b8 * heater
    C = b9 * cooler

    dT_A/dt = b1[b2(T_out - T_A) + H_A - C + b3(T_B - T_A)]
    = b1[b2(T_out - T_A) + b7 * heater - b9 * cooler + b3(T_B - T_A)]
    = b1b2(T_out - T_A) + b1b7 * heater - b1b9 * cooler + b1b3(T_B - T_A)
    = b1b2 * T_out - b1b2 * T_A + b1b7 * heater - b1b9 * cooler + b1b3 * T_B - b1b3 * T_A
    = (b1b2 * T_out) + (-b1b2 - b1b3) * T_A + (b1b3) * T_B + (b1b7) * heater + (-b1b9) * cooler
    = (outside_A) + (self_A * T_A) + (from_B * T_B) + (heater_A * heater) + (cooler_A * cooler)

    dT_B/dt = b4[b5(T_out - T_B) + H_B + b3(T_A - T_B)]
    = (b4b5 * T_out) + (b4b3) * T_A + ((-b4b5 - b4b3) * T_B) + (b4b8 * heater) + 0 * cooler

    - self_A = quanto A si raffredda da sola
    - from_B = quanto B scalda A attraverso la parete
    - self_B = quanto B si raffredda da sola
    - from_A = quanto A scalda B attraverso la parete

    """

    b1 = heater_A / BETA7
    b2 = (-self_A - from_B) / b1
    b3 = from_B / b1
    b4 = from_A / b3
    b5 = (-self_B - from_A) / b4
    b8 = heater_B / b4
    b9 = -cooler_A / b1
    T_out = outside_A / (b1 * b2)

    return np.array([b1, b2, b3, b4, b5, heater_rate, BETA7, b8, b9, T_out])

# Inserisce b7 fra i parametri
def rebuild_beta(adjustable_values):
    beta = np.empty(10)
    beta[ADJUSTABLE] = adjustable_values
    beta[6] = BETA7
    return beta

# Passa 10 valori a residual
def residuals_adjustable(adjustable_values, sims, max_step = 60.0):
    return residuals(rebuild_beta(adjustable_values), sims, max_step)

def fit_beta(sims, beta_guess, max_step = 300.0, verbose = 1):
    result = least_squares(
        residuals_adjustable, # Funzione che misura l'errore
        beta_guess[ADJUSTABLE], # Nove valori del guess iniziale
        args = (sims, max_step), # Argomenti passati alla funzione: porzioni da usare e passo con cui integrare
        bounds = (BETA_LOWER[ADJUSTABLE], BETA_UPPER[ADJUSTABLE]), # Limiti
        x_scale = np.abs(beta_guess[ADJUSTABLE]), 
        verbose = verbose,
    )
    return rebuild_beta(result.x), result
"""
def plot_simulation(sim, beta, title = ''):
    sol = simulate(sim, beta)
    hours = sim['timestamp'].values / 3600

    fig, axes = plt.subplots(6, 1, figsize = (12, 12), sharex = True, gridspec_kw = {'height_ratios': [2, 2, 2, 2, 1, 1]})

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
    axes[5].set_yticks(range(0, 10, 3))
    axes[5].set_ylabel('Level')
    axes[5].set_title('Cooler')
    axes[5].set_xlabel('Hours')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return sol
"""

if __name__ == '__main__':
    dataset = pd.read_csv('data/dataset.csv') # Carica le simulazioni
    sims = [g.reset_index(drop = True) for _, g in dataset.groupby('simulation_id')] 

    train, val , test = split_simulations(sims)

    fit_set = [pd.concat([t, v], ignore_index = True)
               for t, v in zip(train, val)]

    beta_guess = initial_guess(fit_set)

    ratios = beta_guess / BETA_TRUE
    worst_ratio = max(ratios.max(), 1 / ratios.min())
    print(f'Worst ratio: {worst_ratio:.2f}')
    print(f'Initial MAE:' f'{np.abs(residuals(beta_guess, fit_set, 300.)).mean():.4f}\n')

    beta_estimated, result = fit_beta(fit_set, beta_guess)

    print(f'\nFinal MAE:'f'{np.abs(result.fun).mean():.4f}\n')

    for name, guessed, estimated, true in zip(BETA_NAMES, beta_guess, beta_estimated, BETA_TRUE):
        print(f'{name:8s}{guessed:12.5g}{estimated:12.5g}{true:11.5g}')

    np.save('model/beta.npy', beta_estimated)

    fit_portions = [s.iloc[:12 * 60 + 1] for s in fit_set]

    rng = np.random.default_rng(1)
    print(f'\n{"Perturbation":>16s}{"Failures":>12s}{"Worst MAE":>15s}')

    reference_estimated, reference = fit_beta(fit_portions, beta_guess, verbose = 0)
    reference_mae = np.abs(reference.fun).mean()

    print(f'{"perturbation":>14s}{"best MAE":>11s}{"worst MAE":>11s}'f'{"worst drift":>13s}{"parameter":>12s}')

    rng = np.random.default_rng(1)

    for k in [1.5, 2.0, 3.0, 5.0]:
        best_mae, worst_mae = np.inf, 0.0
        worst_drift, worst_name = 0.0, ''

        for attempt in range(3):
            perturbed = beta_guess * np.exp(rng.uniform(-np.log(k), np.log(k), 10))
            perturbed[6] = BETA7
            perturbed = np.clip(perturbed, BETA_LOWER, BETA_UPPER)

            estimated, res = fit_beta(fit_portions, perturbed, verbose = 0)
            mae = np.abs(res.fun).mean()
            best_mae = min(best_mae, mae)
            worst_mae = max(worst_mae, mae)

            drift = np.abs(estimated - reference_estimated) / np.abs(reference_estimated)
            if drift.max() > worst_drift:
                worst_drift = drift.max()
                worst_name = BETA_NAMES[int(np.argmax(drift))]

        print(f'{f"{k}x":>14s}{best_mae:11.4f}{worst_mae:11.4f}'f'{worst_drift:12.1%}{worst_name:>12s}')

    # plot_simulation(train[0], beta_estimated, 'Parametri stimati vs misure')



