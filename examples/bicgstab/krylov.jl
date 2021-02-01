using Krylov
using LinearAlgebra
using SparseArrays
using CUDA

function square_preconditioned(n :: Int=10)
        A   = ones(n, n) + (n-1) * Matrix(I, n, n)
        b   = 10.0 * [1:n;]
        M⁻¹ = 1/n * Matrix(I, n, n)
        return A, b, M⁻¹
end

A, b, N = square_preconditioned(100)
spA = sparse(A)
spN = sparse(N)
y, status = Krylov.bicgstab(spA, copy(b), N=spN)
@show length(status.residuals)
@show y

cuspA = CUDA.CUSPARSE.CuSparseMatrixCSR(spA)
cuspN = CUDA.CUSPARSE.CuSparseMatrixCSR(spN)
cub = CuVector(b)
cuy, status = Krylov.bicgstab(cuspA, cub, N=cuspN)
@show cuy
@show length(status.residuals)