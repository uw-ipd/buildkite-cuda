def test_numba_cuda_is_available():
    import numba.cuda
    assert numba.cuda.is_available()

def test_numba_cuda_smoke():
    import math
    import numba.cuda as cuda
    import numpy
    import numpy.testing

    @cuda.jit
    def matmul(A, B, C):
        """Perform square matrix multiplication of C = A * B
        """
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

    rs = (100, 100)
    a = numpy.random.random(rs)
    b = numpy.random.random(rs)
    c = numpy.empty(rs)

    threadsperblock = (16, 16)
    blockspergrid = tuple(math.ceil(s / t ) for s, t in zip(a.shape, threadsperblock))

    matmul[blockspergrid, threadsperblock](a, b, c)

    numpy.testing.assert_allclose(a @ b, c)

def test_torch_cuda_is_available():
    import torch
    assert torch.cuda.is_available()

def test_torch_cuda_smoke():
    import torch

    rs = (100, 100)
    a = torch.rand(rs)
    b = torch.rand(rs)

    c = a.cuda() @ b.cuda()

    torch.testing.assert_allclose(a @ b, c.cpu())
