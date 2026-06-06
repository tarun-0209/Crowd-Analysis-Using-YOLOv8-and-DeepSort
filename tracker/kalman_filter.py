import numpy as np

class KalmanFilter:
    """
    A mathematical engine that balances "What Physics Predicts" vs "What YOLO Sees" 
    to find the absolute best guess of the drone's true location.
    """

    def __init__(self, initial_bbox):
        # 1. THE STATE VECTOR: The Drone's True Identity.
        # We track/update 8 things: [Center X, Center Y, Width, Height, Vel X, Vel Y, Vel W, Vel H]
        self.state_vector = np.zeros(8)
        self.state_vector[:4] = initial_bbox # First 4 slots are positions/size. Velocities start at 0 because this is the first ever detection.

        
        # 2. THE PHYSICS MATRIX: This defines the rules of motion i.e => future value = current value + old velocity.
        self.physics_matrix = np.eye(8) # Using matrix for these basic calculations makes the math run faster on hardware.
        # vx,vy,vb,vh in Pixels Per Frame and is the last calculated speed of previous frame, later we update the vx,vy,vb,vh based on the 
        # current position vs old position,so we get the next accurate estimate for the velocities so we can calculate the next cx,cy,bw,bw accurately.
        self.physics_matrix[0, 4] = 1   # future cx = Current cx + old vx 
        self.physics_matrix[1, 5] = 1   # future cy = Current cy + old vy 
        self.physics_matrix[2, 6] = 1   # future bw = Current bw + old vw
        self.physics_matrix[3, 7] = 1   # future bh = Current bh + old vh

        
        # 3. THE OBSERVATION MATRIX: The 8D -> 4D "Mask"
        # Maps our internal state [cx, cy, w, h, vx, vy, vw, vh] 
        # To YOLO's measurement    [cx, cy, w, h]
        #
        # Logic: [1 0 0 0 | 0 0 0 0]  <- Keep Positions
        #        [0 1 0 0 | 0 0 0 0]  
        #        [0 0 1 0 | 0 0 0 0]
        #        [0 0 0 1 | 0 0 0 0]  <- Ignore Velocities
        self.observation_matrix = np.eye(4, 8)

        
        # Extract the current width (w) and height (h) to use as our "scale"
        w = initial_bbox[2]
        h = initial_bbox[3]
        # These are standard ratios to define what a "normal" error looks like.
        # Position error is usually about 5% (1/20) of the bounding box size.
        std_weight_position = 1.0 / 20  
        # Velocity error is much smaller, usually about 0.6% (1/160) of the bounding box size.
        std_weight_velocity = 1.0 / 160

        
        # 4. SENSOR NOISE (R): YOLO's Pixel-Error Budget
        # We assume YOLO is off by ~5% of the drone's actual size.
        #
        # Logic: [err_x^2  0      0      0    ]  <- x position variance
        #        [0      err_y^2  0      0    ]  <- y position variance
        #        [0      0      err_w^2  0    ]
        #        [0      0      0      err_h^2]
        std_sensor = np.array([
            std_weight_position * w,
            std_weight_position * h,
            std_weight_position * w,
            std_weight_position * h
        ])
        self.sensor_noise = np.diag(np.square(std_sensor))

        # 5. PHYSICS NOISE (Q): The "Wind & Drift" Factor
        # How much the drone might deviate from our straight-line prediction.
        # Includes drift for both position (5%) and velocity (0.6%).
        #
        # Layout: [Pos Drift (4 slots) | Vel Drift (4 slots)]
        std_physics = np.array([
            std_weight_position * w, std_weight_position * h,
            std_weight_position * w, std_weight_position * h,
            std_weight_velocity * w, std_weight_velocity * h,
            std_weight_velocity * w, std_weight_velocity * h
        ])
        self.physics_noise = np.diag(np.square(std_physics))

        # 6. STATE UNCERTAINTY (P): The "Frame 1 Doubt" Matrix
        # We trust the position (from YOLO), but we have NO CLUE about the speed.
        #
        # Logic: [Pos Uncertainty (Normal) | Vel Uncertainty (10x Massive)]
        # This allows the filter to quickly "learn" the speed in the first few frames.
        std_initial_uncertainty = np.array([
            std_weight_position * w, std_weight_position * h,
            std_weight_position * w, std_weight_position * h,
            10.0 * std_weight_velocity * w, 10.0 * std_weight_velocity * h,
            10.0 * std_weight_velocity * w, 10.0 * std_weight_velocity * h
        ])
        self.state_uncertainty = np.diag(np.square(std_initial_uncertainty))        


    def predict(self):
        # 1. PROJECT STATE: Apply physics to guess the new position.
        # x = F @ x
        self.state_vector = self.physics_matrix @ self.state_vector 
        
        # 2. PROJECT UNCERTAINTY: Doubt grows as time passes without a camera update (YOLO Detection). allows doubt about speed to turn into doubt about position
        # P = F @ P @ F.T + Q
        # Because the uncertainty eventually becomes huge, the next time YOLO sees a drone (the "Surprise"), the Kalman Gain will be very high,
        # forcing the filter to immediately "snap" to the new visual detection instead of stubbornly sticking to its old predicted path.
        self.state_uncertainty = (self.physics_matrix @ self.state_uncertainty @ self.physics_matrix.T) + self.physics_noise

    def update(self, yolo_bbox):
        """
        The update function is the "Correction" phase of the filter, where the system measures the "surprise" gap between
        its physics-based guess and YOLO's actual visual evidence. It calculates a Kalman Gain to act as a mathematical trust factor,
        balancing the reliability of the camera against the certainty of its own internal math. Finally, it uses this balance to "snap" the
        drone's position and velocity to the most likely reality, while simultaneously shrinking the "doubt bubble" (uncertainty) around the track.
        """

        # 1. THE SURPRISE (Innovation): How wrong was my guess?  y = z - (H @ x)
        expected_measurement = self.observation_matrix @ self.state_vector
        # The filter takes its 8D internal guess (position + velocity) and uses the observation_matrix to strip it down to a 4D box. It then subtracts this "expected" box from the real YOLO box.
        # If physics predicted the drone at X=100, but YOLO sees it at X=110, the "Surprise" is +10 pixels
        measurement_residual = yolo_bbox - expected_measurement                        

        # 2. TOTAL SYSTEM UNCERTAINTY (S): Combine internal doubt + sensor noise. How much noise is in the room?  S = (H @ P @ H.T) + R
        # This combines the internal State Uncertainty (P) with YOLO's Sensor Noise (R)
        # It calculates the "Total Error Budget" for the current frame. If YOLO is blurry (high R) and the physics guess is shaky (high P), the total uncertainty S will be very large.
        total_uncertainty = (self.observation_matrix @ self.state_uncertainty @ self.observation_matrix.T) + self.sensor_noise        

        # 3. KALMAN GAIN (K): The "Weight of Trust" (0.0 to 1.0).  K = P @ H.T @ inv(S)
        # This is a value that decides which source of information to believe more 
        # High Gain ( ~1.0): "I don't trust my math; I trust the camera." The drone was likely just re-acquired after being lost.
        # Low Gain ( ~0.0): "I trust my physics; the camera might be jittery." The drone is being tracked steadily.
        kalman_gain = self.state_uncertainty @ self.observation_matrix.T @ np.linalg.inv(total_uncertainty)       

        # 4. CORRECTION: Adjust our 8D state based on the 4D surprise.  x = x + (K @ y)
        # The "Surprise" is multiplied by the "Trust Factor" and added to the state.
        # This doesn't just fix the position (cx, cy); it also fixes the velocity (vx, vy).
        # If the drone moved +10 pixels more than expected, the filter increases the velocity estimate so the next prediction will be even further to the right. 
        self.state_vector = self.state_vector + (kalman_gain @ measurement_residual)                        
        
        # 5. REDUCE DOUBT: We just saw the drone, so our confidence increases.   P = (I - K @ H) @ P
        # Because we just successfully matched a YOLO detection, our "Doubt Bubble" shrinks. This resets the uncertainty that grew during the predict() phase.
        identity_matrix = np.eye(8)
        self.state_uncertainty = (identity_matrix - (kalman_gain @ self.observation_matrix)) @ self.state_uncertainty     


    def get_state(self):
        # Return current estimated [cx, cy, w, h]
        return self.state_vector[:4].copy()

    def predict_future(self, steps=60):
        # A utility function to draw the dotted future trajectory line.
        current_position = self.state_vector[:2] # [cx, cy]
        current_velocity = self.state_vector[4:6] # [vx, vy]
        
        future_points = []
        for step_multiplier in range(1, steps + 1):
            future_points.append(current_position + (current_velocity * step_multiplier))
            
        return future_points