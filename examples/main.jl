using GPUArrays
import PFkernel: ParsePSSE, PowerSystem, IndexSet, Spmat, residual, residual!, CUDABackend, AMDGPUBackend, oneAPIBackend, loaddata, identity

const PS = PowerSystem

local_case = "case14.raw"
V, Ybus, Sbus, Sload, ref, pv, pq = loaddata(local_case)
npv = size(pv, 1)
npq = size(pq, 1)
F = zeros(Float64, npv + 2*npq)
F♯ = residual(V, Ybus, Sbus, pv, pq)
Vm = abs.(V)
Va = angle.(V)
ybus_re, ybus_im = Spmat{Vector{Int}, Vector{Float64}}(Ybus)
pbus = real(Sbus)
qbus = imag(Sbus)
residual!(F, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, length(V))
#@assert(F ≈ F♯)
x = ones(10)
@show x
F, dF = identity(x)
# Value should be value of x = 1.0
@assert(F == ones(10))
# Partials of x will be set to 1.0, so dF should be 1.0 too. x is set to have 2 partials
@show dF
@assert(dF == [(1.0,1.0) for i in 1:10]) 
