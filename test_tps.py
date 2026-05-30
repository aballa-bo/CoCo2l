import numpy as np
from scipy.interpolate import RBFInterpolator

print("Testing RBFInterpolator with 1.38M points...")
training_rgb = np.random.rand(24, 3)
training_xyz = np.random.rand(24, 3)

interp = RBFInterpolator(training_rgb, training_xyz, kernel="thin_plate_spline", smoothing=0.0)

flat = np.random.rand(960 * 1440, 3)
try:
    out = interp(flat)
    print("Success! Shape:", out.shape)
except Exception as e:
    print("Failed with exception:", e)
