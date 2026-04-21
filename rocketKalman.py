###  Riley Martin
# Final Project

# rocket_kalman.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------- Write out Function ----------------------------------------

def write_sensor_file(folder, filename, x_data, y_data, x_header, y_header):

    # Full file path
    filepath = os.path.join(folder, filename)

    with open(filepath, "w") as file:
        file.write(f"{x_header}, {y_header}\n")

        for x, y in zip(x_data, y_data):
            file.write(f"{x}, {y}\n")


# ---------------------------------------- Simulate rocket flight ----------------------------------------
# This section is for modeling the flight dynamics of the rocket

# Define theta as the launch angle of the rocket (constant)
theta = np.pi/4 

# Define the length of the simulation and create time step (dt)
t_final = 100  # time of first burn (s)
t = np.linspace(0, t_final, 100)
dt = t_final / len(t)

# Acceleration parameters of a typical rocket (average stats)
a_liftoff = 1.5   # accel at liftoff (m/s^2)
a_peak = 12.0      # peak accel (m/s^2)
t_peak = 90     # time of peak accel (s)

# Calculate the true acceleration (This is where the rocket really is in both x and y space)
a_true = a_liftoff + (a_peak - a_liftoff) * np.exp(-((t - t_peak)/40)**2)
a_true_x = a_true * np.cos(theta)
a_true_y = a_true * np.sin(theta)

# Calculate the true velocity (This is the rocket's actual velocity in both x and y)
v_true = np.cumsum(a_true*dt)
v_true_x = v_true * np.cos(theta)
v_true_y = v_true * np.sin(theta)

# Calculate the true position (This is the rocket's actual position in x and y space)
p_true = np.cumsum(v_true*dt)
p_true_x = p_true * np.cos(theta)
p_true_y = p_true * np.sin(theta)


# ---------------------------------------- Create the Calculated Model ----------------------------------------
# This model is meant to semi-accuratly model the roacket's acceleration, velocity, and position. While still having error for the filter to correct

# Compute the mock calculated acceleration (This is a physics based estimation of the rocket's acceleration) (Error is added intentionally)
mock_error = 0.35
a_est = a_liftoff + (a_peak - a_liftoff + mock_error) * np.exp(-((t - t_peak)/30)**2)
a_est_x = a_est * np.cos(theta)
a_est_y = a_est * np.sin(theta)

# Compute the mock calculated velocity (This is a physics based estimation of the rocket's velocity) (Error is still present from acceleration calc) 
v_est = np.cumsum(a_est*dt)
v_est_x = v_est * np.cos(theta)
v_est_y = v_est * np.sin(theta)

# Compute the mock calculated position (This is a physics based estimation of the rocket's position) (Error is still present from acceleration calc) 
p_est = np.cumsum(v_est*dt)
p_est_x = p_est * np.cos(theta)
p_est_y = p_est * np.sin(theta)


# ---------------------------------------- Add measurement noise ----------------------------------------
# This section is created to add noise to the senor data. The sensors should be based off the true data but should not be exact. This error will be 
# corrected by the filter

# Set a randomizer seed for reproducibility while still getting seemingly randomized data
np.random.seed(42)  

# Set the standard deviation of both acceleration and position error. This is how the error magnitude is set
accel_noise_std = 2.5
velo_noise_std = 60
pos_noise_std = 500

# Now randomize the acceleration measured data this is simulating noise
accel_meas_x = np.random.normal(a_true_x, accel_noise_std, size=len(t))
accel_meas_y = np.random.normal(a_true_y, accel_noise_std, size=len(t))

# Now randomize the acceleration measured data this is simulating noise
velo_meas_x = np.random.normal(v_true_x, velo_noise_std, size=len(t))
velo_meas_y = np.random.normal(v_true_y, velo_noise_std, size=len(t))

# Do the same for position
pos_meas_x = np.random.normal(p_true_x, pos_noise_std, size=len(t))
pos_meas_y = np.random.normal(p_true_y, pos_noise_std, size=len(t))


# ---------------------------------------- Write out Data ----------------------------------------
# This section uses the write out function to write data to files. Files will be included in submission

# Write out sensor measurement data
write_sensor_file("data", "measures_accel.txt", accel_meas_x, accel_meas_y,"accel_x", "accel_y")
write_sensor_file("data", "measures_velo.txt", velo_meas_x, velo_meas_y,"velo_x", "velo_y")
write_sensor_file("data", "measures_pos.txt", pos_meas_x, pos_meas_y,"pos_x", "pos_y")

# Write out the model estimated data
write_sensor_file("data", "estimated_accel.txt", a_est_x, a_est_y,"accel_x", "accel_y")
write_sensor_file("data", "estimated_velo.txt", v_est_x, v_est_y,"velo_x", "velo_y")
write_sensor_file("data", "estimated_pos.txt", p_est_x, p_est_y,"pos_x", "pos_y")

# Wite out the rocket's true data
write_sensor_file("data", "true_accel.txt", a_true_x, a_true_y,"accel_x", "accel_y")
write_sensor_file("data", "true_velo.txt", v_true_x, v_true_y,"velo_x", "velo_y")
write_sensor_file("data", "true_pos.txt", p_true_x, p_true_y,"pos_x", "pos_y")


# ---------------------------------------- Plot the acceleration in x and y ----------------------------------------
# Here is where acceleration plots are made for true data, noisy sensor readings, and model estimation. This is a good way to see error before integration
# and filtering

# Plot acceleration in the x direction (True data, noisy sensor readings, and model estimation)
plt.figure()
plt.plot(t, a_true_x, label='True Accel')
plt.plot(t, (accel_meas_x), 'r.', label='Noisy Measurements')
plt.plot(t, (a_est_x), label='Model Estimation')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Rocket Acceleration Curve (x direction)')
plt.grid(True)
plt.show()

# Plot acceleration in the y direction (True data, noisy sensor readings, and model estimation)
plt.figure()
plt.plot(t, a_true_y, label='True Accel')
plt.plot(t, (accel_meas_y), 'r.', label='Noisy Measurements')
plt.plot(t, (a_est_y), label='Model Estimation')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Rocket Acceleration Curve (y direction)')
plt.grid(True)
plt.show()


# ---------------------------------------- Kalman Filter Setup ----------------------------------------
# This step is setting up all matricies that are used in kalman filtering. This is set up before running the filter in the loop

# Set initial states (everything starts at 0) 
# This vector is consitant throughout(x_pos, x_vel, y_pos, y_vel)
x = np.array([[0], [0], [0], [0]])  

# State transition matrix 
# Position is iterated by pos_old + v * dt
# While velocity is assumed constant in this step
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

# Control input matrix
# Position 
B = np.array([[0.5*dt**2, 0], [dt, 0], [0, 0.5*dt**2], [0, dt]])

# Observation matrix (we measure altitude directly)
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

# Covariance matrices
P = np.eye(4) * 100  # initial estimate covariance

q = 5.0
Q = np.array([[dt**4/4, dt**3/2, 0, 0],[dt**3/2, dt**2, 0, 0],[0, 0, dt**4/4, dt**3/2],[0, 0, dt**3/2, dt**2]]) * q

R = np.array([[pos_noise_std**2, 0],[0, pos_noise_std**2]])

# Allocate storage for estimates
x_estimates = np.zeros((4, len(t)))


# ---------------------------------------- Run Kalman Filter ----------------------------------------
# This is where the kalman filter is run. It is run at each timestep and it takes in both the physics based model and the noisy sensor data. 
# It uses both of these values to find a middle ground estimation of where the rocket is at each timestep. 

for k in range(len(t)):
    # Prediction step
    u = np.array([[accel_meas_x[k]], [accel_meas_y[k]]]) # acceleration measurement as control input
    x_pred = F @ x + B @ u
    P_pred = F @ P @ F.T + Q
    z = np.array([[pos_meas_x[k]], [pos_meas_y[k]]]) 
         
    # Update step
    u = np.array([[accel_meas_x[k]], [accel_meas_y[k]]])
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    x = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred
    
    x_estimates[:, k] = x.flatten()


# ---------------------------------------- Plot the velocity in x and y ----------------------------------------
# Here is where velocity plots are made for true data, noisy sensor readings, model estimation, and kalman filter output. This is a good way to see 
# error after one round of integration (acceleration > velocity) and filtering.

# X-velocity
plt.figure(figsize=(10,6))
plt.plot(t, v_true_x, 'k-', label='True Velocity')
plt.plot(t, x_estimates[1,:], 'b-', label='Estimated Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity (x) [m/s]')
plt.title('Rocket Velocity (x) Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Y-velocity
plt.figure(figsize=(10,6))
plt.plot(t, v_true_y, 'k-', label='True Velocity')
plt.plot(t, x_estimates[3,:], 'b-', label='Estimated Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity (y) [m/s]')
plt.title('Rocket Velocity (y) Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------- Plot the position in x and y ----------------------------------------
# Here is where position plots are made for true data, noisy sensor readings, and model estimation. This is a good way to see error after two rounds of 
# integration (acceleration > velocity > position) and filtering.

# Plot position in the x direction (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, p_true_x, 'k-', label='True X-position')
plt.plot(t, pos_meas_x, 'r.', label='Noisy Measurements')
plt.plot(t, p_est_x, label='Model Estimation')
plt.plot(t, x_estimates[0,:], 'b-', label='Kalman Estimate')
plt.xlabel('Time [s]')
plt.ylabel('X-position [m]')
plt.title('Rocket Position (x) Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot position in the y direction (altitude) (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, p_true_y, 'k-', label='True Altitude')
plt.plot(t, pos_meas_y, 'r.', label='Noisy Measurements')
plt.plot(t, p_est_y, label='Model Estimation')
plt.plot(t, x_estimates[2,:], 'b-', label='Kalman Estimate')
plt.xlabel('Time [s]')
plt.ylabel('Y-position [m]')
plt.title('Rocket Position (y) Estimation vs Time')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------- Plot the error of both the velocity and position in x and y ----------------------------------------
# These plots are used to evaluate the effectiveness of the kalman filter. The filter's error should be less than the sensor data and the model that 
# is how success is judged. 

# Plot error in the X-direction velocity (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, np.abs(velo_meas_x-v_true_x), label='Noisy Measurements Error')
plt.plot(t, np.abs(x_estimates[1,:]-v_true_x), 'b-', label='Kalman Estimate Error')
plt.plot(t, np.abs(v_est_x-v_true_x), label='Model Estimation Error')
plt.xlabel('Time [s]')
plt.ylabel('X-velocity [m/s]')
plt.title('Rocket Velocity (x) Error vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot error in the Y-direction velocity (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, np.abs(velo_meas_y-v_true_y), label='Noisy Measurements Error')
plt.plot(t, np.abs(x_estimates[3,:]-v_true_y), 'b-', label='Kalman Estimate Error')
plt.plot(t, np.abs(v_est_y-v_true_y), label='Model Estimation Error')
plt.xlabel('Time [s]')
plt.ylabel('Y-velocity [m/s]')
plt.title('Rocket Velocity (y) Error vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot error in the X-direction position (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, np.abs(pos_meas_x-p_true_x), label='Noisy Measurements Error')
plt.plot(t, np.abs(x_estimates[0,:]-p_true_x), 'b-', label='Kalman Estimate Error')
plt.plot(t, np.abs(p_est_x-p_true_x), label='Model Estimation Error')
plt.xlabel('Time [s]')
plt.ylabel('X-position [m]')
plt.title('Rocket Position (x) Error vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot error in the Y-direction position (altitude) (True data, noisy sensor readings, model estimation, and kalman filter output)
plt.figure(figsize=(10,6))
plt.plot(t, np.abs(pos_meas_y-p_true_y), label='Noisy Measurements Error')
plt.plot(t, np.abs(x_estimates[2,:]-p_true_y), 'b-', label='Kalman Estimate Error')
plt.plot(t, np.abs(p_est_y-p_true_y), label='Model Estimation Error')
plt.xlabel('Time [s]')
plt.ylabel('Y-position [m]')
plt.title('Rocket Position (y) Error vs Time')
plt.legend()
plt.grid(True)
plt.show()

