## Optimal spectroscopic measurement design: Bayesian framework for rational data acquisition
Yusei Ito, Yasuo Takeichi, Hideitsu Hino and Kanta Ono

Machine Learning: Science and Technology 6, 025037 (2025). https://doi.org/10.1088/2632-2153/add0f6

## Citation
```text
@article{ito2025experimentaldesign,
  year      = {2025},
  title     = {Optimal spectroscopic measurement design: Bayesian framework for rational data acquisition},
  author    = {Yusei Ito and 
               Yasuo Takeichi and
               Hideitsu Hino and
               Kanta Ono},
  journal   = {Machine Learning: Science and Technology}, 
  doi       = {10.1088/2632-2153/add0f6}, 
  volume    = {6},
  number    = {2}, 
  pages     = {025037}
}
```

## Reproduce the result of our paper
Run the following command: 
```bash
python CutoffOptimization.py
```
```bash
python MakeCovarianceMatrix.py
```
```bash
julia GetOptimalPoints_SquaredLoss.jl
```
```bash
julia GetOptimalPoints_KLDivergence.jl
```
```bash
python BiasVarianceDecomposition.py
```

## Steps of our proposed method
For fast processing, we implemented the part of optimal measurement points in Julia.

Others are implemented in Python
1. Calculate the parameter $c$ in the covariance matrix of a prior distribution.
   The implementation is in CutoffOptimization.py
   
2. Get prior distribution
   The implementation is in the MakeCovarianceMatrix.py

3. Get optimal measurement points
   The implementation is in the GetOptimalPoints.jl

4. Calculate minimal measurement points by bias-variance decomposition.
   The implementation is in the BiasVarianceDecomposition.py
   
Details of each step are provided in the following sections.  

<br>

## How to calculate the parameter $c$
For XAS application provided in our paper, just run   
```bash
python CutoffOptimization.py
```
If you want to try your own measurement, load the standard data and arrange the noise function.


<br>

## Get prior distribution
For the XAS application, we take into account the background that is specific to the XAS problem.

To reproduce our results provided in the paper, run
```bash
python MakeCovarianceMatrix.py
```
For your own application, please comment out the part of the background.


## Get optimal measurement points
You can decide the following parameters:
- `reg_energy`: All of the discritized points, which is the candidate of the measurement points.
- `initial_sampling_num`: Number of initial measurement points.
- `SamplingNum`: The total number of measurement points.

Run
```bash
julia GetOptimalPoints_SquaredLoss.jl
```
```bash
julia GetOptimalPoints_KLDivergence.jl
```
then you can get optimal measurement points. 

## Bias-variance decomposition
Run
```bash
python BiasVarianceDecomposition.py
```
Make sure that you need to use same covariance matrix and candidate measurement points as the part of obtaining the optimal measurement points.

