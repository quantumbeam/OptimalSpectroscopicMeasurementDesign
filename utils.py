import numpy as np


def count2noise(count):
    return np.sqrt(np.max(9.3131 * count - np.ones_like(count) * 63.003, 0))



def spectrum2mutnoise(spectrum, ref=23760, mul=1.75, offset=0.45): #Arrange for your data
    spectrum_add = spectrum * mul + offset
    obj_count = ref * np.exp(-spectrum_add)


    ref_sigma = count2noise(ref)
    obj_sigma = count2noise(obj_count)

    mut_noise = np.sqrt((ref_sigma / ref) * (ref_sigma / ref) + (obj_sigma / obj_count) * (obj_sigma / obj_count))

    return mut_noise / mul





