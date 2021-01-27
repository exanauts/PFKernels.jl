using PFKernels
using PFKernels.PowerSystem
using PFKernels.ForwardDiff

AT = PFKernels.AT

function identity(x)
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    FT = t1s{2}
    V = Vector
    F = copy(x)
    adF = V{FT}(undef, length(F))
    adx = V{FT}(undef, length(x))
    ForwardDiff.seed!(adx, x, one(ForwardDiff.Partials{2,Float64}))
    dF = AT(adF)
    dx = AT(adx)
    PFKernels.kernel!(PFKernels.backend, dF, dx) 
    function pvalues(x)
        return x.values
    end
    return Array(ForwardDiff.values.(dF)), Array(pvalues.(ForwardDiff.partials.(dF)))
end

# Inject GPU kernels
PFKernels.include(joinpath(dirname(@__FILE__), "kernels.jl"))

x = ones(10)
@show x
F, dF = identity(x)
# Value should be value of x = 1.0
@assert(F == ones(10))
# Partials of x will be set to 1.0, so dF should be 1.0 too. x is set to have 2 partials
@show dF
@assert(dF == [(1.0,1.0) for i in 1:10]) 
