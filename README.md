This project implements a 2D Kalman Filter to estimate a rocket’s position and velocity using noisy sensor data and a simple physics-based model.

The simulation generates true motion, adds noise to acceleration, velocity, and position measurements, and compares them to a model with intentional error. The Kalman Filter fuses these inputs to produce a more accurate state estimate.

The state vector is [x, v_x, y, v_y], with acceleration as the control input and position as the measurement. Results show the filter significantly reduces error compared to both raw sensor data and the model alone.

Run rocket_kalman.py to generate plots and data files.
