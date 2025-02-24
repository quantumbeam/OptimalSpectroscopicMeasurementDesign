import numpy as np
import pandas as pd
from scipy import interpolate
import glob


def calc_interpolate(x, y, target_x):
    fitted = interpolate.interp1d(x, y, fill_value='extrapolate')
    return fitted(target_x)


def calc_interpolate_MultiStandard(x, y_mat, target_x):
    retval = np.zeros((target_x.shape[0], y_mat.shape[1]))

    for i in range(y_mat.shape[1]):
        retval[:, i] = calc_interpolate(x, y_mat[:, i], target_x)

    return retval


def NIMSStandard2Standardmatrix(path_folder, interpolate_energy):

    all_path = glob.glob(path_folder + "/*")


    Standardmatrix = np.zeros((interpolate_energy.shape[0], len(all_path)))
    for i in range(len(all_path)):
        df = pd.read_csv(all_path[i])
        Energy = df["Energy"]
        normalize_mut = df["normalize_mut"]

        interpolate_mut = calc_interpolate(Energy, normalize_mut, interpolate_energy)

        Standardmatrix[:, i] = interpolate_mut

    return interpolate_energy, Standardmatrix


def Load_SingleStandard_NIMS(path, interpolate_energy, mul = 1.0):
    df = pd.read_csv(path)

    energy = df["Energy"].values
    normalized_mut = df["normalize_mut"].values * mul

    return calc_interpolate(energy, normalized_mut, interpolate_energy)


def Load_MultiStandard(path_list, interpolate_energy, mul = 1.0):
    Standard = np.zeros((interpolate_energy.shape[0], len(path_list)))

    for i in range(len(path_list)):
        Standard[:, i] = Load_SingleStandard_NIMS(path_list[i], interpolate_energy)

    return Standard


def Load_csv_file(path, label):
    df = pd.read_csv(path)
    return df[label].values

