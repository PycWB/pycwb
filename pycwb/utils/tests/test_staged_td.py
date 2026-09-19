"""Sparse extraction must reproduce dense samples, including whole-bin shifts."""
import numpy as np
import pytest
from pycwb.types.td_batch_inputs import TDBatchInputs

@pytest.mark.parametrize('M', [4, 16, 64])
@pytest.mark.parametrize('stride', [1, 2, 4])
def test_sparse_samples_are_bitwise_dense(M, stride):
    rng = np.random.default_rng(771)
    nc, nt = 3, 32
    J = M * stride
    inputs = TDBatchInputs(
        rng.normal(size=(nt+2*nc, M+1)).astype(np.float32),
        rng.normal(size=(nt+2*nc, M+1)).astype(np.float32),
        rng.normal(size=(2*J+1, 2*nc+1)),
        rng.normal(size=(2*J+1, 2*nc+1)), M, nc, J)
    indices = np.array([n*(M+1)+m for n in [8,9,20,21]
                        for m in [0,1,M//2,M-1,M]], dtype=np.int32)
    K = 2*J+3  # both shift signs, odd/even shifts, non-divisible half-range
    dense = inputs.extract_td_vecs(indices, K).reshape(len(indices),2,2*K+1)
    coarse_K = K//stride
    sparse = inputs.extract_td_vecs(indices, coarse_K, stride).reshape(len(indices),2,-1)
    selected = K + np.arange(-coarse_K,coarse_K+1)*stride
    np.testing.assert_array_equal(sparse.view(np.uint32),dense[:,:,selected].view(np.uint32))
