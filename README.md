# Boltzmann Sampling
## Name
Restricted Boltzmann Machine and Ising Machine-Based Fast Sampling
## Description
This project consists of two parts:
* Reproduced the RBM training from [1], along with ablation experiments where the final logistic regression was replaced with other models for training and testing.
* Reproduced the Ising machine-based sampling from [1], and compared this sampling method with MCMC sampling.

[1] Böhm F, Alonso-Urquijo D, Verschaffelt G, et al. Noise-injected analog Ising machines enable ultrafast statistical sampling and machine learning[J]. Nature Communications, 2022, 13(1): 5847.

## Usage Instructions
* Place all .py files and notebook files in the same folder.
* [RBM_example.ipynb](RBM_example.ipynb) contains an example reproducing the training in [1]:
  * Modify modelname to train and test RBM+model combinations. You can also test the performance of standalone models.
  * Perform ablation experiments by modifying the model name (limited to models specified in the notebook) to train and test RBM+model combinations.
* [sample_verify.ipynb](sample_verify.ipynb) contains an example reproducing the sampling method in [1], which tests ultrafast sampling based on Ising machine dynamics.
  * Adjust parameters a, b, and noise variance noise for fast sampling, and parameter T for MCMC sampling. By tuning noise and T, you can find parameter pairs that yield consistent results.
  * Use the generate_data function in Sample_verify.py to generate an adjacency matrix for an N×N 2D spin lattice with coupling coefficients of -1.
  * Modify NN to change the number of sampling steps for both MCMC and ultrafast sampling.
* [small_verify.ipynb](small_verify.ipynb) contains a small-scale spin graph sampling example:
* Attempt to fit Boltzmann distributions:
  * Plot fitting curves and KL divergence curves between P-E distributions under different noise variances.
  * Plot the relationship between fitted parameter T and noise variance.
  * The structure and coupling coefficients of the small-scale spin graph can be found in [Sample_verify_small.py](Sample_verify_small.py).

