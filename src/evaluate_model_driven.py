import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import split_simulations
from model_driven import simulate, make_commands

def prediction(test_sims, alpha):
    predicted_A, predicted_B = [], []

    for test_sim in test_sims:
        # Comando del riscaldatore al primo istante
        _, cooler_level = make_commands(test_sim)(test_sim['timestamp'].iloc[0])

        # Stato di partenza
        y0 = [test_sim['T_A'].iloc[0], test_sim['T_B'].iloc[0], alpha[9] * cooler_level]

        sol = simulate(test_sim, alpha, y0 = y0, max_step = 60.0)
        predicted_A.append(sol.y[0])
        predicted_B.append(sol.y[1])

    return np.array(predicted_A), np.array(predicted_B)

def metrics(predicted_A, predicted_B, test_sims):
    real_A = np.array([s['T_A'].values for s in test_sims])
    real_B = np.array([s['T_B'].values for s in test_sims])

    real = np.concatenate([real_A.ravel(), real_B.ravel()])
    predicted = np.concatenate([predicted_A.ravel(), predicted_B.ravel()])
    error = predicted - real
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(error ** 2) / np.sum((real - real.mean()) ** 2)

    return {
        'MSE' : mse,
        'MAE' : mae,
        'RMSE' : rmse,
        'R2' : r2
    }

def plot_prediction(test_sim, predicted_A, predicted_B, index = 0):
    hours = (test_sim['timestamp'].values - test_sim['timestamp'].values[0]) / 3600

    fig, axes = plt.subplots(4, 1, figsize = (12, 12), sharex = True, gridspec_kw = {'height_ratios' : [3, 3, 1, 1]})

    axes[0].plot(hours, test_sim['T_A'].values, 'b-', linewidth = 1.5, label = 'Real')
    axes[0].plot(hours, predicted_A, 'r-', linewidth = 1.5, label = 'Predicted')
    axes[0].set_ylabel('Temperature')
    axes[0].set_title('Sensor A')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    axes[1].plot(hours, test_sim['T_B'].values, 'b-', linewidth = 1.5, label = 'Real')
    axes[1].plot(hours, predicted_B, 'r-', linewidth = 1.5, label = 'Predicted')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Sensor B')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    axes[2].step(hours, test_sim['heater_state'].values, 'r-', linewidth = 1.5, where = 'post')
    axes[2].set_ylabel('On/Off')
    axes[2].set_title('Heater')
    axes[2].set_yticks([0, 1])
    axes[2].grid(True, alpha = 0.3)

    axes[3].step(hours, test_sim['cooler_level'].values, 'b-', linewidth = 1.5, where = 'post')
    axes[3].set_ylim(-.5, 9.5)
    axes[3].set_ylabel('Level')
    axes[3].set_xlabel('Hours')
    axes[3].set_title('Cooler')
    axes[3].set_yticks(range(0, 10, 3))
    axes[3].grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset = pd.read_csv('data/dataset.csv') # Legge il dataset
    sims = [g.reset_index(drop = True) for _, g in dataset.groupby('simulation_id')]
    train, val, test = split_simulations(sims) # Splitta in 3 set

    alpha = np.load('model/alpha.npy') # Carica gli alpha migliori
    predicted_A, predicted_B = prediction(test, alpha) # Usa gli alpha caricati nel test set
    m = metrics(predicted_A, predicted_B, test)

    print(' '.join(f'{name}: {value:.4f}' for name, value in m.items()))
    
    index = 0
    plot_prediction(test[index], predicted_A[index], predicted_B[index], index)