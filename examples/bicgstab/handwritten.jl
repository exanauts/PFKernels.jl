using LinearAlgebra
using SparseArrays
using CUDA
using Printf

include("bicgstab.jl")

function square_preconditioned(n :: Int=10)
        A   = ones(n, n) + (n-1) * Matrix(I, n, n)
        b   = 10.0 * [1:n;]
        M⁻¹ = 1/n * Matrix(I, n, n)
        return A, b, M⁻¹
end

A, b, N = square_preconditioned(100)
x = similar(b); r = similar(b)
spA = sparse(A)
spN = sparse(N)
y, iter, status = bicgstab(spA, b, spN, x)
@show y
@show iter
@show status

cuspA = CUDA.CUSPARSE.CuSparseMatrixCSR(spA)
cuspN = CUDA.CUSPARSE.CuSparseMatrixCSR(spN)
cux = CuVector(x)
cub = CuVector(b)
cuy, iter, status = bicgstab(cuspA, cub, cuspN, cux)
@show cuy
@show iter
@show status