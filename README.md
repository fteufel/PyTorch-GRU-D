# PyTorch-GRU-D
PyTorch Implementation of GRU-D from "Recurrent Neural Networks for Multivariate Time Series with Missing Values" https://arxiv.org/abs/1606.01865

Code based on
https://github.com/Han-JD/GRU-D

Adapted for batchwise training, GPU support and fixed bugs.
PyTorch Version 1.3.1

Model takes input of shape ( n_samples, 3, features, seq_length ).
Dimension 1 is (input_matrix, masking_matrix, delta_t_matrix). Input_matrix has 0 where values are missing.
