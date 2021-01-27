using PFKernels
using PFKernels.PowerSystem
using PFKernels.ForwardDiff
using LinearAlgebra

AT = PFKernels.AT

function mynorm(x)
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    n = length(x)
    FT = t1s{n}
    V = Vector
    F = zeros(n)
    adF = V{FT}(undef, n)
    adx = V{FT}(undef, n)
    # ForwardDiff.seed!(adx, x, one(ForwardDiff.Partials{n,Float64}))
    ForwardDiff.seed!(adx, x, ForwardDiff.construct_seeds(ForwardDiff.Partials{n,Float64}))
    dF = AT(adF)
    dx = AT(adx)
    PFKernels.kernel!(PFKernels.backend, dF, dx) 
    nrm2 = t1s{n}(0.0)
    for v in dF
      nrm2 += v
    end
    @show typeof(nrm2)
    @show nrm2
    return ForwardDiff.value(nrm2), ForwardDiff.partials(nrm2)
end

# Inject GPU kernels
PFKernels.include(joinpath(dirname(@__FILE__), "kernels.jl"))

x = rand(10)

nrm2 = x -> norm(x)^2
res = nrm2(x)
dres = ForwardDiff.gradient(nrm2, x)
@show x
F, dF = mynorm(x)
@assert(F ≈ res)
@assert(dF ≈ dres)
