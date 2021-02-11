using PFKernels
using PFKernels.PowerSystem
using PFKernels.ForwardDiff
using UnicodePlots

PFKernels.CUDA.device!(1)

AT = PFKernels.AT

# Matlab-like residual function
function residual(V, Ybus, Sbus, pv, pq)
    # form mismatch vector
    mis = V .* conj(Ybus * V) - Sbus
    # form residual vector
    F = [real(mis[pv])
        real(mis[pq])
        imag(mis[pq]) ]
    return F
end

function recursive_value(dual::ForwardDiff.Dual{Nothing, T, N}) where {T,N}
    if T == Float64
        return ForwardDiff.value(dual)
    else
        recursive_value(dual.value)
    end
end

# Driver for dual numbers
function residual!(::Type{DT}, F::Vector{Float64},
            v_m::Vector{Float64}, v_a::Vector{Float64},
            ybus_re::PFKernels.Spmat{Vector{Int64}, Vector{Float64}},
            ybus_im::PFKernels.Spmat{Vector{Int64}, Vector{Float64}},
            pinj::Vector{Float64}, qinj::Vector{Float64},
            pv::Vector{Int64}, pq::Vector{Int64},
            nbus::Int64) where DT 
    npv = length(pv)
    npq = length(pq)
    adF = Vector{DT}(undef, length(F))
    adv_m = Vector{DT}(undef, length(v_m))
    adv_a = Vector{DT}(undef, length(v_a))
    adybus_re_nzval = Vector{DT}(undef, length(ybus_re.nzval))
    adybus_re_colptr = Vector{Int64}(undef, length(ybus_re.colptr))
    adybus_re_rowval = Vector{Int64}(undef, length(ybus_re.rowval))
    adybus_im_nzval = Vector{DT}(undef, length(ybus_im.nzval))
    adybus_im_colptr = Vector{Int64}(undef, length(ybus_im.colptr))
    adybus_im_rowval = Vector{Int64}(undef, length(ybus_im.rowval))
    adpinj = Vector{DT}(undef, length(pinj))
    adqinj = Vector{DT}(undef, length(qinj))
    adpv = Vector{DT}(undef, length(pv))
    adpq = Vector{DT}(undef, length(pq))

    adF .= F
    adv_m .= v_m
    adv_a .= v_a
    adybus_re_nzval .= ybus_re.nzval
    adybus_im_nzval .= ybus_im.nzval
    adpinj .= pinj
    adqinj .= qinj
    adpv .= pv
    adpq .= pq

    dF = AT(adF)
    dv_m = AT(adv_m)
    dv_a = AT(adv_a)
    dybus_re_nzval = AT(adybus_re_nzval)
    dybus_re_colptr = AT(ybus_re.colptr)
    dybus_re_rowval = AT(ybus_re.rowval)
    dybus_im_nzval = AT(adybus_im_nzval)
    dybus_im_colptr = AT(ybus_im.colptr)
    dybus_im_rowval = AT(ybus_im.rowval)
    dpinj = AT(adpinj)
    dqinj = AT(adqinj)
    dpv = AT(pv)
    dpq = AT(pq)

    PFKernels.kernel!(PFKernels.backend, dF, dv_m, dv_a,
                    dybus_re_nzval, dybus_re_colptr, dybus_re_rowval,
                    dybus_im_nzval, dybus_im_colptr, dybus_im_rowval,
                    dpinj, dqinj, dpv, dpq, nbus)
    F .= recursive_value.(dF)
end

const PS = PowerSystem

# Inject GPU kernels
PFKernels.include(joinpath(dirname(@__FILE__), "kernels.jl"))

# Get the powerflow stuff out of the way
local_case = "case14.raw"
V, Ybus, Sbus, Sload, ref, pv, pq = PFKernels.loaddata(local_case)
npv = size(pv, 1)
npq = size(pq, 1)
F = zeros(Float64, npv + 2*npq)
Fnew = zeros(Float64, npv + 2*npq)
F♯ = residual(V, Ybus, Sbus, pv, pq)
Vm = abs.(V)
Va = angle.(V)
ybus_re, ybus_im = PFKernels.Spmat{Vector{Int}, Vector{Float64}}(Ybus)
pbus = real(Sbus)
qbus = imag(Sbus)

# Let's run some differentiated kernels
t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where {N}
t2s{N} = ForwardDiff.Dual{Nothing,t1s{N}, N} where {N}
t3s{N} = ForwardDiff.Dual{Nothing,t2s{N}, N} where {N}

N = 2
F1 = similar(F)
F2 = similar(F)
F3 = similar(F)
F1 .= 0.0
residual!(t1s{N}, F1, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
F2 .= 0.0
@time residual!(t2s{N}, F2, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
F3 .= 0.0
residual!(t3s{N}, F3, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
@assert(F1 ≈ F♯)
@assert(F2 ≈ F♯)
@assert(F3 ≈ F♯)

# And now we crank up the partials
# times = Vector{Float64}(undef, 10)
const T1 = t2s{10}
const T2 = t2s{20}
ctimes = zeros(10)
ctimes[1] = @elapsed residual!(T1, F2, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
ctimes[2] = @elapsed residual!(T2, F2, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
@show ctimes
rtimes = zeros(10)
rtimes[1] = @elapsed residual!(T1, F2, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
rtimes[2] = @elapsed residual!(T2, F2, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
@show rtimes