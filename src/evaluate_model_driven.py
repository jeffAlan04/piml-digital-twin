import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import load_dataset, split_simulations
import model_driven_exact as exact
import model_driven_approx as approx

EXAMPLE = 28

def initial_state_exact(sim, beta):
    heater_state, _ = exact.make_commands(sim)(sim['timestamp'].iloc[0])
    return [sim['T_A'].iloc[0],
            sim['T_B'].iloc[0],
            beta[6] * heater_state, # H_A a regime
            beta[7] * heater_state] # H_B a regime

def initial_state_approx(sim, alpha):
    _, cooler_level = approx.make_commands(sim)(sim['timestamp'].iloc[0])
    return [sim['T_A'].iloc[0],
            sim['T_B'].iloc[0],
            alpha[9] * cooler_level] # C a regime

def prediction(test_sims, params, module, initial_state):
    predicted_A, predicted_B = [], []

    for test_sim in test_sims:
        
        sol = module.simulate(test_sim, params, y0 = initial_state(test_sim, params), max_step = 60.0)
        predicted_A.append(sol.y[0])
        predicted_B.append(sol.y[1])

    return np.array(predicted_A), np.array(predicted_B)

def metrics(true_A, true_B, predicted_A, predicted_B):
    true = np.concatenate([true_A.ravel(), true_B.ravel()])
    predicted = np.concatenate([predicted_A.ravel(), predicted_B.ravel()])
    error = predicted - true
    
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(error ** 2) / np.sum((true - true.mean()) ** 2)

    return {
        'MSE' : mse,
        'MAE' : mae,
        'RMSE' : rmse,
        'R2' : r2
    }

def plot_prediction(sim, exact_A, exact_B, approx_A, approx_B):
    hours = sim['timestamp'].values / 3600

    fig, axes = plt.subplots(4, 1, figsize = (12, 11), sharex = True, gridspec_kw = {'height_ratios' : [3, 3, 1, 1]})

    axes[0].plot(hours, sim['T_A'].values, 'gray', linewidth = 1.5, label = 'Measured')
    axes[0].plot(hours, exact_A, 'orange', linewidth = 1.5, label = 'Model-driven exact')
    axes[0].plot(hours, approx_A, 'g-', linewidth = 1.5, label = 'Model-driven approx')
    axes[0].set_ylabel('Temperature')
    axes[0].set_title('Sensor A')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    axes[1].plot(hours, sim['T_B'].values, 'gray', linewidth = 1.5, label = 'Measured')
    axes[1].plot(hours, exact_B, 'orange', linewidth = 1.5, label = 'Model-driven exact')
    axes[1].plot(hours, approx_B, 'g-', linewidth = 1.5, label = 'Model-driven approx')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Sensor B')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    axes[2].step(hours, sim['heater_state'].values, 'r-', linewidth = 1.5, where = 'post')
    axes[2].set_ylabel('On/Off')
    axes[2].set_title('Heater')
    axes[2].set_yticks([0, 1])
    axes[2].grid(True, alpha = 0.3)

    axes[3].step(hours, sim['cooler_level'].values, 'b-', linewidth = 1.5, where = 'post')
    axes[3].set_ylim(-0.5, 9.5)
    axes[3].set_ylabel('Level')
    axes[3].set_xlabel('Hours')
    axes[3].set_title('Cooler')
    axes[3].set_yticks(range(0, 10, 3))
    axes[3].grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    sims = load_dataset('data/dataset.csv')

    _, _, test = split_simulations(sims) # Splitta in 3 set

    alpha = np.load('model/alpha.npy') # Carica gli alpha migliori
    beta = np.load('model/beta.npy') # Carica i beta migliori

    true_A = np.array([s['T_A'].values for s in test])
    true_B = np.array([s['T_B'].values for s in test])

    approx_A, approx_B = prediction(test, alpha, approx, initial_state_approx) # Usa gli alpha caricati nel test set
    exact_A, exact_B = prediction(test, beta, exact, initial_state_exact) # Usa i beta caricati nel test set

    columns = ['MSE', 'MAE', 'RMSE', 'R2']
    print(f'{"":28s}' + ''.join(f'{c:>10s}' for c in columns))

    for label, pred_A, pred_B in [('model-driven exact', exact_A, exact_B),
                                  ('model-driven approximate', approx_A, approx_B)]:
        scores = metrics(true_A, true_B, pred_A, pred_B)    
        print(f'{label:28s}' + ''.join(f'{scores[c]:10.4f}' for c in columns))
    
    plot_prediction(test[EXAMPLE], exact_A[EXAMPLE], exact_B[EXAMPLE], approx_A[EXAMPLE], approx_B[EXAMPLE])