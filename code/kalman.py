"""Smoothing the body points over time (report sections 3.4 and 4.3).

The pose model's dots wobble a little from frame to frame, and sometimes a joint
disappears completely when it is hidden under the other athlete. This file fixes that:
  1. Going forwards through the video, it guesses where each joint should be next
     ("it was moving right at this speed, so it is probably about here now") and
     mixes that guess with what the camera actually saw.
  2. If the pose model is not sure about a joint (confidence below 0.35), the camera
     reading is ignored and only the guess is used - the joint is treated as hidden.
  3. Then it goes backwards through the video, so that what happened later can
     correct the earlier guesses. This removes the slight delay a forward-only
     pass would have.
"""

import numpy as np


class BidirectionalJointKalmanFilter:
    """Smooths the path of ONE joint (e.g. the left ankle) through the whole video.

    For every frame it keeps track of four numbers: the joint's position (x, y) in
    pixels and its speed (how many pixels per second it moves left/right and up/down).
    """

    def __init__(self, dt, conf_threshold=0.35, accel_noise=400.0, meas_noise=8.0):
        # dt = time between two analysed frames, in seconds (e.g. every 3rd frame at 30 fps = 0.1 s)
        self.dt = dt
        self.conf_threshold = conf_threshold

        # The "movement rule": next position = current position + speed x time,
        # and the speed stays the same from one frame to the next
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # The camera only tells us where the joint is, not how fast it is moving
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # How much an athlete is allowed to speed up or slow down between two frames.
        # A bigger number lets the smoothed path follow sudden movements more closely.
        q = accel_noise ** 2
        block = np.array([[dt ** 4 / 4, dt ** 3 / 2],
                          [dt ** 3 / 2, dt ** 2]])
        self.Q = np.zeros((4, 4))
        self.Q[np.ix_([0, 2], [0, 2])] = q * block  # left/right movement
        self.Q[np.ix_([1, 3], [1, 3])] = q * block  # up/down movement

        # How much the pose model's dots usually wobble, in pixels
        self.R = np.eye(2) * meas_noise ** 2

    def _usable(self, z, c):
        """Only use a camera reading if the joint was found and the model is confident enough."""
        return (not np.any(np.isnan(z))) and (not np.isnan(c)) and c >= self.conf_threshold

    def forward(self, positions, confidences):
        """Forward pass through the video. Also remembers its guesses, for the backward pass."""
        T = len(positions)
        # Start from the first frame where the joint was clearly visible
        first = next((t for t in range(T) if self._usable(positions[t], confidences[t])), None)
        if first is None:
            return None  # this joint was never seen clearly, so there is nothing to smooth

        x = np.array([positions[first][0], positions[first][1], 0.0, 0.0])
        # How unsure we are about each of the four numbers (very unsure about the speed at first)
        P = np.diag([self.R[0, 0], self.R[1, 1], 500.0 ** 2, 500.0 ** 2])

        xs_filt = np.zeros((T, 4)); Ps_filt = np.zeros((T, 4, 4))
        xs_pred = np.zeros((T, 4)); Ps_pred = np.zeros((T, 4, 4))

        for t in range(T):
            # 1) GUESS: move the joint along at its current speed (we become a bit less sure)
            if t > 0:
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q
            xs_pred[t], Ps_pred[t] = x, P

            # 2) CORRECT: only if the camera clearly saw the joint (confidence 0.35 or more)
            if self._usable(positions[t], confidences[t]):
                z = np.asarray(positions[t], dtype=float)
                y = z - self.H @ x                          # how far off our guess was
                S = self.H @ P @ self.H.T + self.R
                K = P @ self.H.T @ np.linalg.inv(S)         # how much to trust the camera vs. our guess
                x = x + K @ y
                P = (np.eye(4) - K @ self.H) @ P
            # (if the joint is hidden, we simply keep the guess - see report section 4.3)

            xs_filt[t], Ps_filt[t] = x, P

        return xs_filt, Ps_filt, xs_pred, Ps_pred

    def backward_smooth(self, xs_filt, Ps_filt, xs_pred, Ps_pred):
        """Backward pass: use what happened later in the video to improve earlier positions."""
        T = len(xs_filt)
        xs = xs_filt.copy()
        Ps = Ps_filt.copy()
        for t in range(T - 2, -1, -1):
            # C decides how much the later frame is allowed to correct this one
            C = Ps_filt[t] @ self.F.T @ np.linalg.inv(Ps_pred[t + 1])
            xs[t] = xs_filt[t] + C @ (xs[t + 1] - xs_pred[t + 1])
            Ps[t] = Ps_filt[t] + C @ (Ps[t + 1] - Ps_pred[t + 1]) @ C.T
        return xs, Ps

    def smooth(self, positions, confidences):
        """Run both passes for one joint.

        positions: one (x, y) per analysed frame (empty when not found); confidences: one per frame.
        Returns the smoothed (x, y) for every frame (all empty if the joint was never seen).
        """
        positions = np.asarray(positions, dtype=float)
        confidences = np.asarray(confidences, dtype=float)
        fwd = self.forward(positions, confidences)
        if fwd is None:
            return np.full((len(positions), 2), np.nan)
        xs, _ = self.backward_smooth(*fwd)
        return xs[:, :2]


def smooth_all_keypoints(keypoints, dt, conf_threshold=0.35):
    """Smooth every joint (17 per athlete) of both athletes.

    keypoints holds (x, y, confidence) for each analysed frame, athlete and joint,
    with empty values where nothing was found. The result has the same layout,
    with smoothed x and y and the original confidence values.
    """
    smoothed = keypoints.copy()
    kf = BidirectionalJointKalmanFilter(dt, conf_threshold=conf_threshold)
    T, n_athletes, n_joints, _ = keypoints.shape
    for a in range(n_athletes):
        for j in range(n_joints):
            smoothed[:, a, j, :2] = kf.smooth(keypoints[:, a, j, :2], keypoints[:, a, j, 2])
    return smoothed
