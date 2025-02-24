import numpy as np
import pandas as pd
import tifffile

def Count2Noise(count):
    tmp = 9.3131 * count - 63.003
    if tmp < 0:
        return 0
    else:
        return np.sqrt(tmp)

def Spectrum2Noise(spectrum, ref=23760, mul=1.75, offset=0.45): #Arrange to your noise function
    spectrum_add = spectrum * mul + offset
    obj_count = ref * np.exp(-spectrum_add)

    ref_sigma = Count2Noise(ref)

    obj_sigma = np.zeros_like(spectrum)
    for i in range(obj_sigma.shape[0]):
        obj_sigma[i] = Count2Noise(obj_count[i])

    mut_noise = (ref_sigma / ref) * (ref_sigma / ref) + (obj_sigma / obj_count) * (obj_sigma / obj_count)

    return mut_noise / (mul * mul)


def Get_Kernel(kernel, x1, x2):
    tmp = kernel[x1, :]
    return tmp[:, x2]


def Calc_A_optimality(kernel, noise, index, index_tot):
    Kernel_nxn = Get_Kernel(kernel, index_tot, index_tot)
    kernel_mxn = Get_Kernel(kernel, index, index_tot)
    kernel_mxm = Get_Kernel(kernel, index, index)
    C_mxm = kernel_mxm + np.diag(noise[index])
    C_mxm_inv = np.linalg.inv(C_mxm)

    tmp_bias = Kernel_nxn + np.dot(kernel_mxn.T, np.dot((np.dot(C_mxm_inv, np.dot(kernel_mxm, C_mxm_inv)) - 2 * C_mxm_inv), kernel_mxn))
    tmp_variance = np.dot(kernel_mxn.T, np.dot(C_mxm_inv, np.dot(np.diag(noise[index]), np.dot(C_mxm_inv, kernel_mxn))))

    return np.diag(tmp_bias), np.diag(tmp_variance), np.diag(tmp_bias + tmp_variance)

def Get_NearestIndex(energy, target_energy):
    index_list = np.zeros_like(target_energy)
    for i in range(index_list.shape[0]):
        index_list[i] = np.argmin(np.abs(energy - target_energy[i]))

    return index_list.astype(np.int64)


def LoadKernel(path_of_kernel_folder, reg_energy):
    path_kernel = path_of_kernel_folder + "/Kernel.tif"
    path_energy_mean = path_of_kernel_folder + "/KernelEnergy_mean.csv"

    kernel = np.array(tifffile.imread(path_kernel))

    df = pd.read_csv(path_energy_mean)
    energy = df["KernelEnergy"].values
    mean = df["mean"].values

    noise = Spectrum2Noise(mean)

    reg_index = Get_NearestIndex(energy, reg_energy)

    return kernel, noise, reg_index, energy

def GetIndex_SamplingPoints(kernel_energy, sampling_energy):
    sampling_index = Get_NearestIndex(kernel_energy, sampling_energy)

    return sampling_index


def Calc_A_optimality_AllPoints(path_of_kernel, path_of_sampling_points, reg_energy, path_save):
    kernel, noise, reg_index, kernel_energy = LoadKernel(path_of_kernel, reg_energy)

    df = pd.read_csv(path_of_sampling_points)
    SamplingPoints = df["SamplingPoints"].values
    num = df["num"].values

    A_optimality_tot_sup_list = np.zeros_like(SamplingPoints)
    A_optimality_tot_sum_list = np.zeros_like(SamplingPoints)
    A_optimality_bias_sup_list = np.zeros_like(SamplingPoints)
    A_optimality_bias_sum_list = np.zeros_like(SamplingPoints)
    A_optimality_variance_sup_list = np.zeros_like(SamplingPoints)
    A_optimality_variance_sum_list = np.zeros_like(SamplingPoints)

    for i in range(SamplingPoints.shape[0]):
        SamplingEnergy = SamplingPoints[:i + 1]
        sampling_index = GetIndex_SamplingPoints(kernel_energy, SamplingEnergy)

        A_optimality_bias, A_optimality_variance, A_optimality_tot = Calc_A_optimality(kernel, noise, sampling_index, reg_index)

        A_optimality_tot_sup_list[i] = np.max(A_optimality_tot)
        A_optimality_tot_sum_list[i] = np.sum(A_optimality_tot)
        A_optimality_bias_sup_list[i] = np.max(A_optimality_bias)
        A_optimality_bias_sum_list[i] = np.sum(A_optimality_bias)
        A_optimality_variance_sup_list[i] = np.max(A_optimality_variance)
        A_optimality_variance_sum_list[i] = np.sum(A_optimality_variance)

    df = pd.DataFrame()
    df["num"] = np.array(num)
    df["A_optimality_tot_sum"] = np.array(A_optimality_tot_sum_list)
    df["A_optimality_bias_sum"] = np.array(A_optimality_bias_sum_list)
    df["A_optimality_variance_sum"] = np.array(A_optimality_variance_sum_list)

    df.to_csv(path_save)




if __name__ == "__main__":
    path_of_kernel = "results/Kernel"
    path_of_sampling_points = "results/OptimalPoints/SamplingPoints.csv"

    reg_energy = np.arange(7076.2, 7181.21, 0.1)
    path_save = "results/OptimalPoints/BiasVariance_Decomposition.csv"

    Calc_A_optimality_AllPoints(path_of_kernel, path_of_sampling_points, reg_energy, path_save)



