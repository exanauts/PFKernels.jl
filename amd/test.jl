using ForwardDiff
using AMDGPU

function f(x)
    x = x.^2 
    nrm2 = eltype(x)(0.0)
    for v in x
        nrm2 += v
    end
    return nrm2
end


n = 10

# Works
t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
x = rand(n)
adx = Array{t1s{n}}(undef, n)
ForwardDiff.seed!(adx, x, ForwardDiff.construct_seeds(ForwardDiff.Partials{n,Float64}))
rocadx = ROCVector{t1s{n}}(adx)
rocadg = f(rocadx)

# Doesn't works 
x = ROCVector{Float64}(undef, n)
adx = ROCVector{t1s{n}}(undef, n)
ForwardDiff.seed!(adx, x, ForwardDiff.construct_seeds(ForwardDiff.Partials{n,Float64}))
adg = adf(x)

# Doesn't work
df = x -> ForwardDiff.jacobian(f,x)
g = df(x) # gradient of F
