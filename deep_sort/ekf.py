import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

class ExtendedKalmanFilter:
    def __init__(self):
        self.dim_x = 8  # state dimension
        self.dim_z = 4  # measurement dimension
        self.dt = 1.0   # time step

        # Create Kalman filter model matrices.
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = self.dt
        self.Q = np.eye(self.dim_x) * 0.01

        # Measurement matrix (H) and measurement noise covariance (R)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(self.dim_z) * 0.1

        # Position and velocity noise weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.zeros(self.dim_x)
        mean[:4] = measurement
        covariance = np.eye(self.dim_x) * 10
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.dot(self.F, mean)
        covariance = np.dot(np.dot(self.F, covariance), self.F.T) + self.Q
        return mean, covariance

    def update(self, mean, covariance, measurement):
        y = measurement - np.dot(self.H, mean)
        S = np.dot(self.H, np.dot(covariance, self.H.T)) + self.R
        K = np.dot(np.dot(covariance, self.H.T), np.linalg.inv(S))

        mean = mean + np.dot(K, y)
        I = np.eye(self.dim_x)
        covariance = np.dot(I - np.dot(K, self.H), covariance)
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self.H, mean)
        covariance = np.linalg.multi_dot((self.H, covariance, self.H.T))
        return mean, covariance + innovation_cov
