# rocket_kalman.py
import numpy as np
import matplotlib.pyplot as plt

# ------------------ 1. Simulate rocket flight ------------------

t_final = 100  # time of first burn (s)
t = np.linspace(0, t_final, 100)
dt = t_final / len(t)

# Parameters
a_liftoff = 1.5   # accel at liftoff (m/s^2)
a_peak = 12.0      # peak accel (m/s^2)
t_peak = 90     # time of peak accel (s)

# Mock accel (true val)
a_true = a_liftoff + (a_peak - a_liftoff) * np.exp(-((t - t_peak)/40)**2)

# True vel and alt
v_true = np.cumsum(a_true*dt)
h_true = np.cumsum(v_true*dt)

# ------------------ 2. Add measurement noise ------------------

np.random.seed(42)  # for reproducibility
accel_noise_std = 1.5
alt_noise_std = 200

accel_meas = np.random.normal(a_true, accel_noise_std, size=len(t))

with open("measures_accel.txt", "w") as f:
    for v in accel_meas:
        f.write(str(v) + "\n")

alt_meas = np.random.normal(h_true, alt_noise_std, size=len(t))

with open("measures_alt.txt", "w") as f:
    for v in alt_meas:
        f.write(str(v) + "\n")

###   Plot accel
plt.plot(t, a_true, label='True Accel')
plt.plot(t, (accel_meas), 'r.', label='Noisy Measurements')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Mock Rocket Acceleration Curve')
plt.grid(True)
plt.show()

# ------------------ 3. Kalman Filter Setup ------------------

# Initial states: (altitude_0, velocity_0)
x = np.array([[0], [0]])  

# State transition matrix -------------------------------------------------------------
F = np.array([[1, dt], [0, 1]])

# Control input matrix
B = np.array([[0.5*dt**2], [dt]])

# Observation matrix (we measure altitude directly)
H = np.array([[1, 0]])

# Covariance matrices
P = np.eye(2) * 100         # initial estimate covariance
Q = np.array([[0.01, 0], [0, 0.01]])     # process noise
R = np.array([[alt_noise_std**2]])  # measurement noise

# Allocate storage for estimates
x_estimates = np.zeros((2, len(t)))

# ------------------ 4. Run Kalman Filter ------------------

for k in range(len(t)):
    # Prediction step
    u = accel_meas[k]  # acceleration measurement as control input
    x_pred = F @ x + B * u
    P_pred = F @ P @ F.T + Q
    
    # Update step
    z = np.array([[alt_meas[k]]])
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred
    
    x_estimates[:, k] = x.flatten()

# ------------------ 5. Plot results ------------------

###   Altitude Plot
plt.figure(figsize=(10,6))
plt.plot(t, h_true, 'k-', label='True Altitude')
plt.plot(t, alt_meas, 'r.', label='Noisy Measurements')
plt.plot(t, x_estimates[0,:], 'b-', label='Kalman Estimate')
plt.xlabel('Time [s]')
plt.ylabel('Altitude [m]')
plt.title('Rocket Altitude Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()

###   Velocity Plot 
plt.figure(figsize=(10,6))
plt.plot(t, v_true, 'k-', label='True Velocity')
plt.plot(t, x_estimates[1,:], 'b-', label='Estimated Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Rocket Velocity Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()

###   Altitude Error Plot
plt.figure(figsize=(10,6))
plt.plot(t, (x_estimates[0,:]-h_true), 'k-', label='Error')
plt.plot(t, (alt_meas-h_true), 'r.', label='Noisy Measurements')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.title('Kalman Filter Error vs Time')
plt.legend()
plt.grid(True)
plt.show()