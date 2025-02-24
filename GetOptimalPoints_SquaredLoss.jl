using LinearAlgebra
using ProgressBars
using TiffImages
using DataFrames
using CSV
using PyCall
@pyimport os
@pyimport shutil


function GetKernel_value_value(x_1::Int64, x_2::Int64, kernel :: Matrix{Float64})::Float64
    return kernel[x_1, x_2]
end


function GetKernel_value_vector(x_1::Int64, x_2::Vector{Int64}, kernel :: Matrix{Float64})::Vector{Float64}
    return kernel[x_1, x_2]
end


function GetKernel_vector_vector(x_1::Vector{Int64}, x_2::Vector{Int64}, kernel :: Matrix{Float64})::Matrix{Float64}
    tmp = kernel[x_1, :]
    return tmp[:, x_2]
end

function GetEvaluateValue_NextStep(x_measure::Vector{Int64}, x_reg::Vector{Int64}, sigma_measure::Vector{Float64}, sigma_reg::Vector{Float64}, kernel :: Matrix{Float64})::Vector{Float64}
    size_n = Int32(size(x_reg)[1])
    size_m = Int32(size(x_measure)[1])

    EvaluateValue_NextStep = zeros(Float64, size_n)
    K_nxn = GetKernel_vector_vector(x_reg, x_reg, kernel)
    K_nxn_term = tr(K_nxn)
    K_nxmp1 = zeros(Float64, size_n, size_m + 1)
    K_nxmp1[:,begin:end - 1] = GetKernel_vector_vector(x_reg, x_measure, kernel)
    K_mp1xmp1 = zeros(Float64, size_m + 1, size_m + 1)
    K_mp1xmp1[begin:end - 1, begin:end - 1] = GetKernel_vector_vector(x_measure, x_measure, kernel)
    sigma_measurep1 = zeros(Float64, size_m + 1)
    sigma_measurep1[begin:end - 1] = sigma_measure

    for k :: Int64 = 1:size(x_reg)[1]
        K_nxmp1[:, end] = GetKernel_value_vector(x_reg[k], x_reg, kernel)
        K_mp1xmp1[end, begin:end-1] = GetKernel_value_vector(x_reg[k], x_measure, kernel)
        K_mp1xmp1[begin:end-1, end] = GetKernel_value_vector(x_reg[k], x_measure, kernel)
        K_mp1xmp1[end, end] = GetKernel_value_value(x_reg[k], x_reg[k], kernel)
        sigma_measurep1[end] = sigma_reg[k]

        C_mp1xmp1_inv = inv(K_mp1xmp1 .+ Diagonal(sigma_measurep1))
        alpha = K_nxmp1 * C_mp1xmp1_inv

        EvaluateValue_NextStep[k] = K_nxn_term - sum(alpha .* K_nxmp1)
    end
    return EvaluateValue_NextStep
end


function OptimizeSampling(x_measure_init :: Vector{Int64}, x_reg :: Vector{Int64}, sigma_measure_init :: Vector{Float64}, sigma_reg::Vector{Float64}, SamplingNum::Int64, kernel :: Matrix{Float64})
    Evaluation_list = zeros(Float64, SamplingNum)
    x_measure = x_measure_init
    sigma_measure = sigma_measure_init
    for k :: Int64 = tqdm(1:SamplingNum)
        Evaluation_NextStep = GetEvaluateValue_NextStep(x_measure, x_reg, sigma_measure, sigma_reg, kernel)
        argmin_eval = argmin(Evaluation_NextStep)
        Evaluation_list[k] = Evaluation_NextStep[argmin_eval]
        push!(x_measure, x_reg[argmin_eval])
        push!(sigma_measure, sigma_reg[argmin_eval])
    end
    return x_measure, Evaluation_list
end

function LoadKernel(path)
    img = Matrix{Float64}(TiffImages.load(path))
    return img
end

function Count2Noise(count)
    tmp = 9.3131 * count - 63.003
    if tmp < 0
        return 0
    else
        return sqrt(tmp)
    end
end


function Spectrum2Noise(spectrum, ref, mul, offset)
    spectrum_add = spectrum * mul .+ offset
    obj_count = ref * exp.(-spectrum_add)

    ref_sigma = Count2Noise(ref)

    obj_sigma = zeros(Float64, size(obj_count)[1])
    for k :: Int64 = 1:size(obj_sigma)[1]
        obj_sigma[k] = Count2Noise(obj_count[k])
    end

    mut_noise = (ref_sigma / ref) .^ 2 .+ (obj_sigma ./ obj_count) .^ 2

    return mut_noise / (mul^2)
end


function make_use_kernel(x_reg, kernel_mean, kernel, kernel_energy)
    index_list = zeros(Int64, size(x_reg))
    for k :: Int64 = 1:size(x_reg)[1]
        index_list[k] = argmin(abs.(kernel_energy .- x_reg[k]))
    end
    use_kernel = kernel[index_list, :]
    use_kernel = use_kernel[:, index_list]

    return use_kernel, kernel_energy[index_list], kernel_mean[index_list]
end


function LoadKernelEnergy(path_kernel_energy)
    df = CSV.read(path_kernel_energy, DataFrame)
    kernel_energy = Vector{Float64}(df[!, "KernelEnergy"])
    mean_spectrum = Vector{Float64}(df[!, "mean"])

    return kernel_energy, mean_spectrum
end


function do_Optimize(path_kernel, reg_energy, initial_sampling_num, SamplingNum, path_save_folder)
    os.mkdir(path_save_folder)
    kernel = LoadKernel(path_kernel * "/Kernel.tif")
    path_kernel_energy = path_kernel * "/KernelEnergy_mean.csv"
    kernel_energy, mean_spectrum = LoadKernelEnergy(path_kernel_energy)
    use_kernel, use_kernel_energy, mean_spectrum = make_use_kernel(reg_energy, mean_spectrum, kernel, kernel_energy)
    init_index = Vector{Int64}(round.(Vector{Float64}(range(1, size(reg_energy)[1], length = initial_sampling_num))))
    reg_index = Vector(1:size(use_kernel)[1])

    reg_sigma = Spectrum2Noise(mean_spectrum, 23760, 1.75, 0.45) #Use your own noise function

    init_sigma = reg_sigma[init_index]

    x_measure, Evaluation_list = OptimizeSampling(init_index, reg_index, init_sigma, reg_sigma, SamplingNum, use_kernel)

    path_save = path_save_folder * "/SamplingPoints.csv"
    df = DataFrame()
    df[!, "num"] = Vector(1:SamplingNum + initial_sampling_num)
    df[!, "SamplingPoints"] = use_kernel_energy[x_measure]
    df |> CSV.write(path_save)

    path_save = path_save_folder * "/EvaluationValues.csv"
    df = DataFrame()
    df[!, "num"] = Vector(1:SamplingNum)
    df[!, "Eval"] = Evaluation_list
    df |> CSV.write(path_save)
end

path_kernel = "results/Kernel"

reg_energy = Vector{Float64}(7076.2:0.1:7181.2)
initial_sampling_num = 2
SamplingNum = 320
path_save_folder = "results/OptimalPoints_SquaredLoss"

do_Optimize(path_kernel, reg_energy, initial_sampling_num, SamplingNum, path_save_folder)