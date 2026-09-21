"""
Simulatore di una stanza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Parametri fisici
alpha1 = 0.001
alpha2 = 0.1 # Coefficiente di scambio termico tra la stanza e l'esterno
alpha9 = 0.1 # Efficienza del cooler
T_out = 15  # Temperatura esterna
H_max = 1.0  # Potenza massima del riscaldatore


# Definizione delle azioni (istante, heater (acceso/spento), potenza del cooler (0-9))
actions = [
    (0, False, 0),
    (3600, True, 0),
    (14400, False, 5),
    (21600, True, 0),
    (28800, False, 9),
    (32400, False, 0),
]

def get_cooler_power(t, actions):
    # Dato un tempo t, restituisce la potenza del raffreddatore 
    # guardando l'ultima azione eseguita prima di t
    current_power = 0 # Stato iniziale

    for action_time, heater, cooler in actions:
        if t >= action_time:
            current_power = cooler
        else:
            break

    return alpha9 * current_power

def get_heater_power(t, actions):
    # Dato un tempo t, restituisce la potenza del riscaldatore
    # guardando l'ultima azione eseguita prima di t
    current_state = False # Stato iniziale

    for action_time, heater, cooler in actions:
        if t >= action_time:
            current_state = heater
        else:
            break

    return H_max if current_state else 0.0

# Equazione differenziale
def deriv(t, y):
    T = y[0] # Temperatura corrente
    H = get_heater_power(t, actions) # Potenza del riscaldatore al tempo t
    C = get_cooler_power(t, actions) # Potenza del cooler al tempo t
    dTdt = alpha1 * (alpha2 * (T_out - T) + H - C) # Cambiamenti di gradi al secondo in questo istante

    return [dTdt]

#Condizioni iniziali
T_initial = 10.0 
# Intervallo di tempo
t_start = 0
t_end = 36000
t_eval = np.arange(t_start, t_end, 60) # Punti in cui si vuole conoscere la temperatura (ogni 60 secondi)

# Integrazione della derivata con solve_ivp
solution = solve_ivp(
    fun = deriv, # Funzione che calcola le derivate
    t_span = [t_start, t_end], # Intervallo di tempo
    y0 = [T_initial], 
    t_eval = t_eval,
    method = 'RK45'
)

times = solution.t  # Array dei tempi
temperature = solution.y[0]  # Array delle temperature

# Aggiunta del rumore
sigma = 0.1
measured_temperatures = temperature + np.random.normal(0, sigma, size = len(temperature))

# Restituisce la potenza del riscaldatore ad ogni momento
heater_state = np.array([get_heater_power(t, actions) for t in times])
cooler_state = np.array([get_cooler_power(t, actions) for t in times])

hour = times / 3600

# Stampa
time_step = np.arange(t_start, t_end + 1, 1800)
print(f"{'Hour':>6} | {'T real':>10} | {'T Measured':>10} | {'Heater':>12} | {'Cooler':>10}")

for t_step in time_step:
    idx = np.argmin(np.abs(times - t_step))
    hour_stamp = t_step / 3600
    T_real = temperature[idx]
    T_measured = measured_temperatures[idx]
    heater = "ON" if get_heater_power(t_step, actions) > 0 else "OFF"
    cooler_val = get_cooler_power(t_step, actions) / alpha9
    print(f"{hour_stamp:5.1f}h | {T_real:9.4f} C | {T_measured:9.4f} C | {heater:>12} | {cooler_val:>8.0f}/9")

print(f"Final real temperature: {temperature[-1]:.4f} C")

# Grafico
fig, ax1 = plt.subplots(figsize=(12,6))
 
# Temperature 
ax1.plot(hour, temperature, 'b-', linewidth=2, label='Real temperature')
ax1.scatter(hour[::5], measured_temperatures[::5], c='red', s=8, alpha=0.5, label='Sensor Measure')
ax1.set_ylabel('Temperature (C)')
ax1.set_xlabel('Time (hour)')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
 
# Attuatori 
ax2 = ax1.twinx()
ax2.fill_between(hour, 0, heater_state, alpha=0.15, color='orange', label='Heater')
ax2.fill_between(hour, 0, -cooler_state, alpha=0.15, color='cyan', label='Cooler')
ax2.set_ylabel('Power')
ax1.tick_params(axis='y')
ax2.set_ylim(-1.5, 3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

