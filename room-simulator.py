"""
Simulatore di una stanza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Parametri fisici
alpha1 = 0.001 # Quanto velocemente A reagisce ai cambiamenti
alpha2 = 0.1 # Coefficiente di scambio termico tra la stanza A e l'esterno
alpha3 = 0.1 # Scambio termico tra le due stanze
alpha4 = 0.002 # Quanto velocemente B reagisce ai cambiamenti
alpha5 = 0.1 # Coefficiente di scambio termico tra la stanza B e l'esterno
alpha6 = 0.001 # Velocita' di riscaldamento degli heater
alpha7 = 1.0 # Potenza massima heater A
alpha8 = 0.5 # Potenza massima heater B
alpha9 = 0.1 # Efficienza del cooler
T_out = 15  # Temperatura esterna
H_max = 1.0  # Potenza massima del riscaldatore
sigma = 0.1



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
    current_power = 0 

    for action_time, heater, cooler in actions:
        if t >= action_time:
            current_power = cooler
        else:
            break

    return alpha9 * current_power

def get_heater_ref(t, actions):
    # Dato un tempo t, verifica se heater e' accesso o spento
    current_state = False # Stato iniziale

    for action_time, heater, cooler in actions:
        if t >= action_time:
            current_state = heater
        else:
            break

    return 1.0 if current_state else 0.0

# Equazione differenziale
def deriv(t, y):
    T_A = y[0] # Temperatura corrente stanza A
    T_B = y[1] # Temperatura corrente stanza B    
    H_A = y[2] # Potenza del riscaldatore A al tempo t
    H_B = y[3] # Potenza del riscaldatore B al tempo t
    
    heater_ref = get_heater_ref(t, actions) # 1 o 0
    C = get_cooler_power(t, actions) # Potenza del cooler al tempo t
    
    dTdt_A = alpha1 * (alpha2 * (T_out - T_A) + H_A - C + alpha3 * (T_B - T_A)) # Differenza di temperatura al secondo in questo istante nella stanza A
    dTdt_B = alpha4 * (alpha5 * (T_out - T_B) + H_B + alpha3 * (T_A - T_B)) # Differenza di temperatura al secondo in questo istante nella stanza B
    dHdt_A = alpha6 *  (alpha7 * heater_ref - H_A) # Inerzia di heater A
    dHdt_B = alpha6 *  (alpha8 * heater_ref - H_B) # Inerzia di heater B

    return [dTdt_A, dTdt_B, dHdt_A, dHdt_B]

#Condizioni iniziali
T_A_initial = 10.0 
T_B_initial = 10.0 
t_start = 0
t_end = 36000
t_eval = np.arange(t_start, t_end, 60) # Punti in cui si vuole conoscere la temperatura (ogni 60 secondi)

# Integrazione della derivata con solve_ivp
solution = solve_ivp(
    fun = deriv, # Funzione che calcola le derivate
    t_span = [t_start, t_end], # Intervallo di tempo
    y0 = [T_A_initial, T_B_initial, 0, 0], 
    t_eval = t_eval,
    method = 'RK45'
)

times = solution.t  # Array dei tempi
temp_A = solution.y[0]  # Temperature stanza A
temp_B = solution.y[1]  # Temperature stanza B
heat_A = solution.y[2]  # Potenza effettiva heater A nel tempo
heat_B = solution.y[3]  # Potenza effettiva heater B nel tempo


# Aggiunta del rumore
measured_A_temp = temp_A + np.random.normal(0, sigma, size = len(temp_A))
measured_B_temp = temp_B + np.random.normal(0, sigma, size = len(temp_B))

# Restituisce la potenza del riscaldatore ad ogni momento
heater_state = np.array([get_heater_ref(t, actions) for t in times])
cooler_state = np.array([get_cooler_power(t, actions) for t in times])

hour = times / 3600

# Stampa
time_step = np.arange(t_start, t_end + 1, 1800)
print(f"{'Hour':>6} | {'T A real':>10} | {'T A Measured':>10} | {'T B real':>10} | {'T B Measured':>10} |  {'Heater':>12} | {'Cooler':>10}")

for step in time_step:
    idx = np.argmin(np.abs(times - step))
    hour_stamp = step / 3600
    heater = "ON" if get_heater_ref(step, actions) > 0 else "OFF"
    cooler_val = get_cooler_power(step, actions) / alpha9
    print(f"{hour_stamp:5.1f}h | {temp_A[idx]:9.4f} C | {measured_A_temp[idx]:9.4f} C | {temp_B[idx]:9.4f} C | {measured_B_temp[idx]:9.4f} C  | {heater:>12} | {cooler_val:>8.0f}/9")

print(f"Final real temperature: A = {temp_A[-1]:.4f} C, B = {temp_B[-1]:.4f} C")

# Grafico
fig, ax1 = plt.subplots(figsize=(12,6))
 
# Temperature 
ax1.plot(hour, temp_A, 'b-', linewidth=2, label='Real temperature A')
ax1.plot(hour, temp_B, 'r-', linewidth=2, label='Real temperature B')
ax1.scatter(hour[::5], measured_A_temp[::5], c='blue', s=8, alpha=0.5, label='Sensor Measure A')
ax1.scatter(hour[::5], measured_B_temp[::5], c='red', s=8, alpha=0.5, label='Sensor Measure B')
ax1.set_ylabel('Temperature (C)')
ax1.set_xlabel('Time (hour)')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
 
# Attuatori 
ax2 = ax1.twinx()
ax2.fill_between(hour, 0, heater_state, alpha=0.15, color='orange', label='Heater')
cooler_setpoint = cooler_state / alpha9
ax2.fill_between(hour, 0, -cooler_setpoint, alpha=0.15, color='cyan', label='Cooler')
ax2.set_ylabel('Power')
ax1.tick_params(axis='y')
ax2.set_ylim(-10, 5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

