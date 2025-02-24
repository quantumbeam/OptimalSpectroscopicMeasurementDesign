import numpy as np
import pandas as pd
import os
import tifffile

def MakeCovarianceMatrix(sigma, x, cutoff):
    CovarianceMatrix = np.diag(sigma ** 2)

    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[0]):
            CovarianceMatrix[i, j] = np.sqrt(CovarianceMatrix[i, i] * CovarianceMatrix[j, j]) * np.exp(-((x[i] - x[j]) ** 2) / (cutoff * cutoff))
    return CovarianceMatrix


def Background_CovarianceMatrix(energy, sigma_a=5, sigma_b=5, sigma_c=5):
    energy = 7110.8 / energy
    energy_power3 = np.power(energy, 3).reshape(-1, 1)
    energy_power4 = np.power(energy, 4).reshape(-1, 1)
    constant = np.ones_like(energy_power3).reshape(-1, 1)

    Background_CovMat = np.dot(energy_power3, energy_power3.T) * sigma_a * sigma_a + np.dot(energy_power4, energy_power4.T) * sigma_b * sigma_b + np.dot(constant, constant.T) * sigma_c * sigma_c

    return Background_CovMat

def AddMulFactor(mean, cov, mul_mean=1.75, mul_sigma=5):
    cov_result = (mul_mean * mul_mean + mul_sigma * mul_sigma) * cov + mul_sigma * mul_sigma * np.dot(np.array([mean]).T, np.array([mean]))

    return cov_result

def Calc_likelihood(mean, cov, standard):
    standard_num = standard.shape[1]
    mean = np.broadcast_to(mean, (standard_num, mean.shape[0])).T

    _, logdet = np.linalg.slogdet(cov)

    cov_inv = np.linalg.inv(cov)
    tmp = np.dot(cov_inv, standard - mean)
    tmp = np.dot((standard - mean).T, tmp)

    return -np.trace(tmp) - logdet * standard_num

def Calc_mean_sigma(standard):
    mean = np.mean(standard, axis = 1)
    sigma = np.std(standard, axis = 1)

    return mean, sigma



if __name__ == "__main__":
    path_energy = "MDRStandard_energy.csv"
    path_standard = "MDRStandard.csv"
    cutoff = 4.35 #Use the result value of CutoffOptimization.py. In XAS, we use 4.35 for cutoff.

    path_save_folder = "results/Kernel"

    df = pd.read_csv(path_energy)
    Energy = df["Energy"].values

    standard = pd.read_csv(path_standard).values[:, 1:]
    mean, sigma = Calc_mean_sigma(standard)

    cov = MakeCovarianceMatrix(sigma, Energy, cutoff)

    #For XAS application. In XAS, there is a contribution of background. We added it to covariance matrix.
    cov = AddMulFactor(mean, cov)
    background = Background_CovarianceMatrix(Energy)
    cov = cov + background
    #If you use this method to your own data, please comment this out.


    os.mkdir(path_save_folder)
    df = pd.DataFrame()
    df["KernelEnergy"] = Energy
    df["mean"] = mean
    df.to_csv(path_save_folder + "/KernelEnergy_mean.csv")

    tifffile.imsave(path_save_folder + "/Kernel.tif", cov)

    print("Kernel size" + str(cov.shape))
    print("Kernel Energy" + str(Energy.shape))
