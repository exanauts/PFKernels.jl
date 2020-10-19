using PFkernel
import PFkernel: ParsePSSE, PowerSystem, IndexSet, Spmat, residualFunction, residualFunction_polar!

const PS = PowerSystem

local_case = "case14.raw"

datafile = joinpath(dirname(@__FILE__), "..", "data", local_case)
data_raw = ParsePSSE.parse_raw(datafile)
data, bus_to_indexes = ParsePSSE.raw_to_exapf(data_raw)

BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN,
LAM_P, LAM_Q, MU_VMAX, MU_VMIN = IndexSet.idx_bus()
bus = data["bus"]
gen = data["gen"]
SBASE = data["baseMVA"][1]
nbus = size(bus, 1)
Ybus = PS.makeYbus(data, bus_to_indexes)
V = Array{Complex{Float64}}(undef, nbus)
T = Vector
for i in 1:nbus
    V[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
end

Vm = abs.(V)
Va = angle.(V)
bus = data["bus"]
gen = data["gen"]
nbus = size(bus, 1)
ngen = size(gen, 1)

ybus_re, ybus_im = Spmat{T{Int}, T{Float64}}(Ybus)
SBASE = data["baseMVA"][1]
Sbus, Sload = PS.assembleSbus(gen, bus, SBASE, bus_to_indexes)
pbus = real(Sbus)
qbus = imag(Sbus)
ref, pv, pq = PS.bustypeindex(bus, gen, bus_to_indexes)
npv = size(pv, 1)
npq = size(pq, 1)
F = zeros(Float64, npv + 2*npq)
F♯ = residualFunction(V, Ybus, Sbus, pv, pq)
residualFunction_polar!(F, Vm, Va,
    ybus_re, ybus_im,
    pbus, qbus, pv, pq, nbus)
@show F
@show F♯
@assert(F ≈ F♯)
