__precompile__(false)
module PFKernels
    using SparseArrays
    using Printf
    include("utils.jl")

    include("indexes.jl")
    using .IndexSet
    include("parsers/parse_mat.jl")
    using .ParseMAT
    include("parsers/parse_psse.jl")
    using .ParsePSSE
    include("PowerSystem/PowerSystem.jl")
    using .PowerSystem

    const PS = PowerSystem

    using ForwardDiff
    abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct CUDABackend <: AbstractBackend end
    struct AMDGPUBackend <: AbstractBackend end
    struct oneAPIBackend <: AbstractBackend end
    backend = nothing
    AT = Array
    try
        using CUDA
        @info("CUDA loaded")
        global backend = CUDABackend()
        global AT = CuVector

    catch
        try
            using AMDGPU
            @info("AMDGPU loaded")
            global backend = AMDGPUBackend()
            global AT = ROCVector
        catch
            try
                using oneAPI
                @info("oneAPI loaded")
                global backend = oneAPIBackend()
                global AT = oneArray
            catch
                @info("No GPU backend could be loaded, switching to CPU")
                global backend = CPUBackend()
            end
        end
    end

    # Defining dummy macros for the code to run
    if !(@isdefined CUDA)
        macro cuda(ex)
            return :( error("CUDA module not loaded") )
        end
    end

    if !(@isdefined AMDGPU)
        macro roc(ex)
            return :( error("AMDGPU module not loaded") )
        end
    end

    if !(@isdefined oneAPI)
        macro oneapi(ex...)
            return :( error("oneAPI module not loaded") )
        end
    end

    function kernel!(::AMDGPUBackend, F, args...)
        F_ = copy(F)
        io = open("code_lowered.txt", "w")
        print(io, @device_code_lowered @roc kernel_roc!(F_, args...))
        close(io)
        F_ = copy(F)
        io = open("code_typed.txt", "w")
        print(io, @device_code_lowered @roc kernel_roc!(F_, args...))
        close(io)
        F_ = copy(F)
        io = open("code_llvm.txt", "w")
        print(io, @device_code_lowered @roc kernel_roc!(F_, args...))
        close(io)
        wait(@roc kernel_roc!(F, args...))
    end

    function kernel!(::CUDABackend, F, args...)
    F_ = copy(F)
    io = open("code_lowered.txt", "w")
        @sync begin
            print(io, @device_code_lowered @cuda kernel_cuda!(F_, args...))
        end
    close(io)
    F_ = copy(F)
    io = open("code_typed.txt", "w")
        @sync begin
            print(io, @device_code_lowered @cuda kernel_cuda!(F_, args...))
        end
    close(io)
    io = open("code_llvm.txt", "w")
        @sync begin
            print(io, @device_code_lowered @cuda kernel_cuda!(F, args...))
        end
    close(io)
    end

    function kernel!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus, ::oneAPIBackend)
        n = length(pv) + length(pq)
        @oneapi items=n kernel_oneapi!(F, v_m, v_a,
                    ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                    ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                    pinj, qinj, pv, pq, nbus)
    end

    function loaddata(case)
        datafile = joinpath(dirname(@__FILE__), "..", "data", case)
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
        for i in 1:nbus
            V[i] = bus[i, VM]*exp(1im * pi/180 * bus[i, VA])
        end

        Vm = abs.(V)
        Va = angle.(V)
        bus = data["bus"]
        gen = data["gen"]
        nbus = size(bus, 1)
        ngen = size(gen, 1)

        SBASE = data["baseMVA"][1]
        Sbus, Sload = PS.assembleSbus(gen, bus, SBASE, bus_to_indexes)
        ref, pv, pq = PS.bustypeindex(bus, gen, bus_to_indexes)
        return V, Ybus, Sbus, Sload, ref, pv, pq
    end
end
