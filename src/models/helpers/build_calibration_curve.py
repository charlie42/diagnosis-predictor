import numpy as np
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from util import *

def build_calibration_curve(y_true, y_pred, output_dir, diag):

    disp = CalibrationDisplay.from_predictions(y_true, y_pred, n_bins=5)

    # Save plot 
    path = output_dir + "calibration_curves/"
    create_dir_if_not_exists(path)
    plt.savefig(path + f"{diag}.png")
