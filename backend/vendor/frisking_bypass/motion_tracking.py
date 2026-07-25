import numpy as np


class KalmanBBoxTracker:
    """Constant-velocity Kalman filter for an (x, y, width, height) bbox."""

    def __init__(
        self,
        bbox,
        frame_count,
        process_noise=1.0,
        measurement_noise=10.0,
        initial_covariance=100.0,
    ):
        x, y, width, height = [float(value) for value in bbox]
        self.state = np.zeros(8, dtype=np.float64)
        self.state[:4] = [x + width / 2.0, y + height / 2.0, width, height]
        self.covariance = np.eye(8, dtype=np.float64) * max(1.0, float(initial_covariance))
        self.covariance[4:, 4:] *= 4.0
        self.process_noise = max(1e-6, float(process_noise))
        self.measurement_noise = max(1e-6, float(measurement_noise))
        self.last_frame = int(frame_count)
        self.measurement_matrix = np.zeros((4, 8), dtype=np.float64)
        self.measurement_matrix[:4, :4] = np.eye(4, dtype=np.float64)

    def _bbox(self):
        center_x, center_y, width, height = self.state[:4]
        width = max(2.0, float(width))
        height = max(2.0, float(height))
        return (
            float(center_x - width / 2.0),
            float(center_y - height / 2.0),
            width,
            height,
        )

    def _sanitize_size(self):
        self.state[2] = max(2.0, float(self.state[2]))
        self.state[3] = max(2.0, float(self.state[3]))

    def predict(self, frame_count):
        frame_count = int(frame_count)
        delta = frame_count - self.last_frame
        if delta <= 0:
            return self._bbox()

        dt = float(delta)
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt

        position_noise = self.process_noise * max(1.0, dt * dt)
        velocity_noise = self.process_noise * max(1.0, dt)
        process_covariance = np.diag(
            [position_noise] * 4 + [velocity_noise] * 4
        )

        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + process_covariance
        )
        self.last_frame = frame_count
        self._sanitize_size()
        return self._bbox()

    def update(self, bbox, frame_count):
        self.predict(frame_count)
        x, y, width, height = [float(value) for value in bbox]
        measurement = np.array(
            [x + width / 2.0, y + height / 2.0, width, height],
            dtype=np.float64,
        )
        measurement_covariance = np.eye(4, dtype=np.float64) * self.measurement_noise
        innovation = measurement - self.measurement_matrix @ self.state
        innovation_covariance = (
            self.measurement_matrix
            @ self.covariance
            @ self.measurement_matrix.T
            + measurement_covariance
        )
        projected_covariance = self.covariance @ self.measurement_matrix.T
        try:
            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                projected_covariance.T,
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = projected_covariance @ np.linalg.pinv(innovation_covariance)

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        correction = identity - kalman_gain @ self.measurement_matrix
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self._sanitize_size()
        return self._bbox()
