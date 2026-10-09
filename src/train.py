import os
import tensorflow as tf
import numpy as np
from dataset import load_dataset, split_simulations, compute_stats, normalize_simulations, make_sliding_windows, map_XY
from model import SeqGen

# Crea la cartella per salvare le statistiche
os.makedirs('model', exist_ok = True)

np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
num_layers = 4 
num_heads = 4
d_model = 64
dff = 128
prediction_steps = 10
dropout_rate = 0.1
batch_size = 128
epochs = 100 # Massimo di epoche
learning_rate = 0.001

# Carica il dataset
simulations = load_dataset('data/dataset.csv')

# Splitta il dataset
train_sims, val_sims, test_sims = split_simulations(simulations)

# Trova mean e std del training set
mean, std = compute_stats(train_sims)

# Normalization dei vari set
train_norm = normalize_simulations(train_sims, mean, std)
val_norm = normalize_simulations(val_sims, mean, std)
test_norm = normalize_simulations(test_sims, mean, std)

# Grandezza della window
window_size = prediction_steps + 1

# Ogni simulazione viene tagliata in finestre scorrevoli di lunghezza window_size
train_windows = make_sliding_windows(train_norm, window_size)
val_windows = make_sliding_windows(val_norm, window_size)
test_windows = make_sliding_windows(test_norm, window_size)

# Separa ogni finestra in input e target
X_train, Y_train = map_XY(train_windows, prediction_steps)
X_val, Y_val = map_XY(val_windows, prediction_steps)
X_test, Y_test = map_XY(test_windows, prediction_steps)

# Creazione del modello
model = SeqGen(
    num_layers = num_layers,
    num_heads = num_heads,
    d_model = d_model,
    dff = dff,
    prediction_steps = prediction_steps,
    dropout_rate = dropout_rate
)

# Regressione: Adam come ottimizzatore e MSE come loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss=tf.keras.losses.MeanSquaredError()
)

# Ferma il training se la val_loss non migliora per 10 epoche e ripristina i pesi migliori
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    restore_best_weights = True
)

# Salva il modello solo quando la val_loss migliora
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = 'model/best_model.keras',
    monitor = 'val_loss',
    save_best_only = True
)

# Training
model.fit(
    X_train, 
    Y_train,
    batch_size = batch_size, 
    epochs = epochs, 
    callbacks = [early_stopping, model_checkpoint], 
    validation_data = (X_val, Y_val)
    )

# Salva mean e std
np.savez('model/norm_stats.npz', mean = mean, std = std)
print(f"Statistiche salvate: mean = {mean:.4f}, std = {std:.4f}")

# Valutazione finale sul test set 
test_loss = model.evaluate(X_test, Y_test)
print(f"Test Loss (MSE): = {test_loss:.6f}")
