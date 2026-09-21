"""
Simulatore di una stanza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Parametri fisici
alpha1 = 0.001
alpha2 = 0.1 # Coefficiente di scambio termico tra la stanza e l'esterno
T_out = 15  # Temperatura esterna
H_max = 1.0  # Potenza massima del riscaldatore

# Definizione delle azioni
actions = [
    (0, False),
    (3600, True),
    (14400, False),
    (21600, True),
    (28800, False)
]

def get_heater_power(t, actions):
    # Dato un tempo t, restituisce la potenza del riscaldatore 
    # guardando l'ultima azione eseguita prima di t
    current_state = False # Stato iniziale

    for action_time, state in actions:
        if t >= action_time:
            current_state = state
        else:
            break

    return H_max if current_state else 0.0

# Equazione differenziale
def deriv(t, y):
    T = y[0] # Temperatura corrente
    H = get_heater_power(t, actions) # Potenza del riscaldatore al tempo t

    dTdt = alpha1 * (alpha2 * (T_out - T) + H) # Cambiamenti di gradi al secondo in questo istante

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

hour = times / 3600

# Grafico
fig, ax1 = plt.subplots(figsize=(12,6))
 
# Temperature 
ax1.plot(hour, temperature, 'b-', linewidth=2, label='Real temperature')
ax1.scatter(hour[::5], measured_temperatures[::5], c='red', s=8, alpha=0.5, label='Sensor Measure')
ax1.set_ylabel('Temperature (C)')
ax1.set_xlabel('Time (hour)')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
 
# Riscaldatore 
ax2 = ax1.twinx()
ax2.fill_between(hour, 0, heater_state, alpha=0.3, color='orange', label='Heater')
ax2.set_ylabel('Power')
ax1.tick_params(axis='y')
ax2.set_ylim(-0.1, 1.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()
