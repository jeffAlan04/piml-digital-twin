import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_dataset, split_simulations, normalize_simulations, make_sliding_windows, map_XY
from model import SeqGen

# Carica il modello
model = tf.keras.models.load_model('model/best_model.keras', custom_objects = {'SeqGen' : SeqGen})

# Carica le statistiche
stats = np.load('model/norm_stats.npz')
mean = stats['mean'] 
std = stats['std']

prediction_steps = 10
simulations = load_dataset('data/dataset.csv')
_, _, test_sims = split_simulations(simulations) # Ottiene le simulazioni test
test_norm = normalize_simulations(test_sims, mean, std) # Normalizza

window_size = prediction_steps + 1 
test_window = make_sliding_windows(test_norm, window_size)
X_test, Y_test = map_XY(test_window, prediction_steps) # Separa ogni finestra in input X e target Y

# Predizione
Y_pred = model.predict(X_test)

# Denormalizzazione delle predizioni e dei valori reali
Y_pred_real = Y_pred * std + mean
Y_test_real = Y_test * std + mean

# Metriche
mse = np.mean(np.power(Y_pred_real - Y_test_real, 2))
mae = np.mean(np.abs(Y_pred_real - Y_test_real))
rmse = np.sqrt(np.mean((Y_pred_real - Y_test_real) ** 2))

y_true_mean = np.mean(Y_test_real)
ss_tot = np.sum(np.power(Y_test_real - y_true_mean, 2))
ss_res = np.sum(np.power(Y_test_real - Y_pred_real, 2))
r2 = 1 - ss_res / ss_tot


print(f"\nMetrics")
print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f} °C")
print(f"RMSE: {rmse:.4f} °C")
print(f"R^2: {r2:.4f}")


n_samples = 200
# Sensore A
plt.figure(figsize = (12, 4))
plt.plot(Y_test_real[:n_samples, 0, 0], label = 'Real', alpha = 0.7)
plt.plot(Y_pred_real[:n_samples, 0, 0], label = 'Predicted', alpha = 0.7)
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.title('Sensor A')
plt.legend()
plt.tight_layout()

# Sensore B
plt.figure(figsize=(12, 4))
plt.plot(Y_test_real[:n_samples, 1, 0], label='Real', alpha=0.7)
plt.plot(Y_pred_real[:n_samples, 1, 0], label='Predicted', alpha=0.7)
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.title('Sensor B')
plt.legend()
plt.tight_layout()

# Scatter plot
plt.figure(figsize = (12, 4))
plt.scatter(Y_test_real[:, 0, 0], Y_pred_real[:, 0, 0], alpha = 0.3, s = 10)
plt.plot([Y_test_real.min(), Y_test_real.max()],
         [Y_test_real.min(), Y_test_real.max()], 'r--', label = 'Perfect')
plt.xlabel('Real temperature')
plt.ylabel('Predicted temperature')
plt.legend()
plt.tight_layout()
plt.show()
