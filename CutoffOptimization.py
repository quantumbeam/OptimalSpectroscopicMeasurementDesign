import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import MakeCovarianceMatrix
import utils
from tqdm import tqdm


def calc_interpolate(x, y, target_x):

    fitted = interpolate.interp1d(x, y, fill_value='extrapolate')
    return fitted(target_x)



def Calc_likelihood(mean, cov, standard):
    standard_num = standard.shape[1]
    mean = np.broadcast_to(mean, (standard_num, mean.shape[0])).T

    _, logdet = np.linalg.slogdet(cov)

    delta_standard = standard - mean
    C_N_inv_t = np.linalg.solve(cov, delta_standard)
    tmp = np.dot(delta_standard.T, C_N_inv_t)

    return -np.trace(tmp) - logdet * standard_num


def Calc_expected_likelihood_single(mean, cov, standard, noise_val):
    cov = cov + np.diag(noise_val)
    cov_inv = np.linalg.inv(cov)
    _, cov_logdet = np.linalg.slogdet(cov)
    noise_term = np.diag(noise_val)

    tmp = np.dot(cov_inv, mean - standard)
    quad_term = np.dot(mean - standard, tmp)


    expected_liklihood = -cov_logdet - np.trace(np.dot(noise_term, cov_inv)) - quad_term

    return expected_liklihood

def Calc_expected_likelihood(mean, cov, standard, noise_val):
    standard_num = standard.shape[1]
    cov = cov + np.diag(noise_val)
    cov_inv = np.linalg.inv(cov)

    _, cov_logdet = np.linalg.slogdet(cov)
    noise_term = np.diag(noise_val)

    trace_term = np.trace(np.dot(noise_term, cov_inv))

    mean_broadcast = np.broadcast_to(mean, (standard_num, mean.shape[0])).T
    tmp = np.dot(cov_inv, mean_broadcast - standard)
    quad_term = np.trace(np.dot(mean_broadcast.T - standard.T, tmp))


    return (-standard_num * trace_term - quad_term - standard_num * cov_logdet) / standard_num



def MakeKernel_and_CalcLikelihood(Energy, standard, cutoff):
    mean = np.mean(standard, axis = 1)
    sigma = np.std(standard, axis = 1)

    cov = MakeCovarianceMatrix.MakeCovarianceMatrix(sigma, Energy, cutoff)

    noise_val = utils.spectrum2mutnoise(mean) ** 2

    return Calc_expected_likelihood(mean, cov, standard, noise_val)



if __name__ == "__main__":
    energy = np.arange(7076.2, 7181.22, 0.1) #Set your own discretize points

    path_save = "results/CutoffVSLikelihood.csv"

    #Load XAS standard spectra data
    spectra = pd.read_csv("MDRStandard.csv").values[:, 1:]
    standard_energy = pd.read_csv("MDRStandard_energy.csv").values
    spectra = np.array([calc_interpolate(standard_energy[:, 0], spectra[:, i], energy) for i in range(spectra.shape[1])]).T
    #

    #spectra = "Load your own data" #Load your own database of spectra


    ret_list = []
    cutoff_list = []
    for i in tqdm(range(20)):
        cutoff = i * 0.05 + 4.0 #Set suitable range of your data
        retval = MakeKernel_and_CalcLikelihood(energy, spectra, cutoff)
        ret_list.append(retval)
        cutoff_list.append(cutoff)

    ret_list = np.array(ret_list)

    df = pd.DataFrame()
    df["cutoff"] = np.array(cutoff_list)
    df["Likelihood"] = np.array(ret_list)
    df.to_csv(path_save)

    print("Cutoff Parameter is: " + str(np.array(cutoff_list)[np.argmax(np.array(ret_list))])) #In our example, you can get c=4.35
    plt.figure()
    plt.plot(np.array(cutoff_list), ret_list)
    plt.show()
