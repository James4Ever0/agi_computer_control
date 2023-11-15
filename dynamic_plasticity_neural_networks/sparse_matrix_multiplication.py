import torch
import sparse
import time
large_number = 1_000_000
average_synapses = 3 # 1.7s, 0.146s
# average_synapses = 10 # 11.8s, 0.505s
# average_synapses = 100 # ..., 13.258s
# average_synapses = 1_000

np_sparse = sparse.random((large_number, large_number), nnz=large_number*average_synapses)

torch_sparse = torch.sparse_coo_tensor(np_sparse.coords, np_sparse.data, np_sparse.shape)

torch_dense = torch.randn(large_number, dtype=torch.double)

# before = time.time()
# result = torch_sparse@torch_sparse
# after = time.time()
# print(result)
# print(f"calculation time: {after-before:.3f}s")

before = time.time()
result_dense = torch_dense@torch_sparse
after = time.time()
print(result_dense)
print(f"calculation time: {after-before:.3f}s")
