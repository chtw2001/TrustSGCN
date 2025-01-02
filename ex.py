import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

original_matrix = np.array([
    [10, 10, 10, 30],
    [10, 10, 20, 10],
    [10, 10, 20, 10],
    [10, 30, 10, 10]
])

csr_matrix_rep = csr_matrix(original_matrix)

data = csr_matrix_rep.data
row_indices = csr_matrix_rep.indices
col_pointers = csr_matrix_rep.indptr

print(original_matrix, data, row_indices, col_pointers)

import torch
combined = torch.cat([torch.rand([100, 10]), torch.rand([100, 30])], dim=1)
print(combined.shape)