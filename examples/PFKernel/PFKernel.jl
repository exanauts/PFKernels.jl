using PFKernels
using PFKernels.PowerSystem
using PFKernels.ForwardDiff

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

# Driver for dual numbers
function residual!(F, v_m, v_a,
                                ybus_re, ybus_im,
                                pinj, qinj, pv, pq, nbus)
    npv = length(pv)
    npq = length(pq)
    t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
    FT = t1s{2}
    V = Vector
    adF = V{FT}(undef, length(F))
    adv_m = V{FT}(undef, length(v_m))
    adv_a = V{FT}(undef, length(v_a))
    adybus_re_nzval = V{FT}(undef, length(ybus_re.nzval))
    adybus_re_colptr = V{Int64}(undef, length(ybus_re.colptr))
    adybus_re_rowval = V{Int64}(undef, length(ybus_re.rowval))
    adybus_im_nzval = V{FT}(undef, length(ybus_im.nzval))
    adybus_im_colptr = V{Int64}(undef, length(ybus_im.colptr))
    adybus_im_rowval = V{Int64}(undef, length(ybus_im.rowval))
    adpinj = V{FT}(undef, length(pinj))
    adqinj = V{FT}(undef, length(qinj))
    adpv = V{FT}(undef, length(pv))
    adpq = V{FT}(undef, length(pq))

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
    F .= ForwardDiff.value.(dF)
end


const PS = PowerSystem

# Inject GPU kernels
PFKernels.include(joinpath(dirname(@__FILE__), "kernels.jl"))

local_case = "case14.raw"
V, Ybus, Sbus, Sload, ref, pv, pq = PFKernels.loaddata(local_case)
npv = size(pv, 1)
npq = size(pq, 1)
F = zeros(Float64, npv + 2*npq)
F♯ = residual(V, Ybus, Sbus, pv, pq)
Vm = abs.(V)
Va = angle.(V)
ybus_re, ybus_im = PFKernels.Spmat{Vector{Int}, Vector{Float64}}(Ybus)
pbus = real(Sbus)
qbus = imag(Sbus)
residual!(F, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
@assert(F ≈ F♯)