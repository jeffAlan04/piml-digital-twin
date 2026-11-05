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
dt = 60

"""
test_window = make_sliding_windows(test_norm, window_size)
X_test, Y_test = map_XY(test_window, prediction_steps) # Separa ogni finestra in input X e target Y

# Denormalizzazione delle predizioni e dei valori reali
delta_pred = model.predict(X_test, verbose = 0)
T_in = X_test[:, :2, 2]
Y_pred_real = (T_in[:, :, None] + delta_pred) * std + mean
Y_test_real = (T_in[:, :, None] + Y_test) * std + mean

T0 = X_test[:, :2, 2] * std + mean
persistence = np.repeat(T0[:, :, None], prediction_steps, axis = 2)
"""

def metrics(pred, real):
    # Metriche
    mse = np.mean((pred - real) ** 2)
    mae = np.mean(np.abs(pred - real))
    rmse = np.sqrt(mse)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    ss_res = np.sum((real - pred) ** 2)
    r2 = 1 - ss_res / ss_tot

    return mse, mae, rmse, r2

# Lunghezza della porzione di test set più corta
length = min(len(c) for c in test_sims)
# Numero di porzioni nel test set
count = len(test_sims)

real_A = np.stack([c['T_A'].values[:length] for c in test_sims])
real_B = np.stack([c['T_B'].values[:length] for c in test_sims])
heater = np.stack([c['heater_state'].values[:length] for c in test_sims])
cooler = np.stack([c['cooler_level'].values[:length] for c in test_sims]) / 9.0

# Prima temperatura di ogni simulazione normalizzata
TA = (real_A[:, 0] - mean) / std
TB = (real_B[:, 0] - mean) / std

# Seleziona fra il test set quelle che partono con l'heater spento e lo accendono successivamente
with_switch_on = [j for j in range(count) if heater[j, 0] == 0 and heater[j].max() == 1]

example = min(with_switch_on, default = 0, key = lambda j: abs(np.argmax(heater[j]) - length // 2))

preds_A = np.zeros((count, length)); preds_A[:, 0] = real_A[:, 0]
preds_B = np.zeros((count, length)); preds_B[:, 0] = real_B[:, 0]

zeros, ones = np.zeros(count), np.ones(count)

for t in range(1, length):
    # Mette insieme temperatura attuale di A, B e azioni degli attuatori
    X = np.stack([
        np.stack([zeros, zeros, TA], axis = 1), # Sensore A: tipo 0, stanza 0
        np.stack([zeros, ones, TB], axis = 1), # Sensore B: tipo 0, stanza 1
        np.stack([ones, zeros, heater[:, t-1]], axis = 1), # heater: tipo 1, stanza 0
        np.stack([2 * ones, zeros, cooler[:, t - 1]], axis = 1), # cooler: tipo 2, stanza 0
    ], axis = 1).astype(np.float32)

    # Variazione di temperatura
    D = model(X, training = False).numpy()

    # Somma il cambiamento alle temperature attuali
    TA = TA + D[:, 0, 0]
    TB = TB + D[:, 1, 0]

    preds_A[:, t] = TA * std + mean
    preds_B[:, t] = TB * std + mean

mse, mae, rmse, r2 = metrics(np.concatenate([preds_A, preds_B]),np.concatenate([real_A,  real_B]))
print(f"MSE {mse:.4f}  MAE {mae:.4f} °C  RMSE {rmse:.4f} °C  R2 {r2:.4f}")

hours = test_sims[example]['timestamp'].values[:length] / 3600

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axes[0].plot(hours, real_A[example], 'b-', linewidth=1.5, label='Real')
axes[0].plot(hours, preds_A[example], 'r-', linewidth=1.5, label='Predicted')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Sensor A')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hours, real_B[example], 'b-', linewidth=1.5, label='Real')
axes[1].plot(hours, preds_B[example], 'r-', linewidth=1.5, label='Predicted')
axes[1].set_ylabel('Temperature (°C)')
axes[1].set_title('Sensor B')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].step(hours, heater[example], 'r-', linewidth=1.5, where='post')
axes[2].set_ylabel('On/Off')
axes[2].set_title('Heater')
axes[2].set_yticks([0, 1])
axes[2].grid(True, alpha=0.3)

axes[3].step(hours, cooler[example] * 9, 'b-', linewidth=1.5, where='post')
axes[3].set_ylim(-0.5, 9.5)
axes[3].set_yticks(range(0, 10))
axes[3].set_ylabel('Level')
axes[3].set_title('Cooler')
axes[3].set_xlabel('Hours')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()