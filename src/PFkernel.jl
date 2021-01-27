__precompile__(false)
module PFkernel
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
    try
        using CUDA
        @info("CUDA loaded")
        global backend = CUDABackend()
    catch
        try
            using AMDGPU
            @info("AMDGPU loaded")
            global backend = AMDGPUBackend()
        catch
            try
                using oneAPI
                @info("oneAPI loaded")
                global backend = oneAPIBackend()
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


    
    function residual(V, Ybus, Sbus, pv, pq)
        # form mismatch vector
        mis = V .* conj(Ybus * V) - Sbus
        # form residual vector
        F = [real(mis[pv])
            real(mis[pq])
            imag(mis[pq]) ]
        return F
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

    function kernel!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus, ::CPUBackend)
        npv = length(pv)
        npq = length(pq)
        n = npq + npv
        # i = @index(Global, Linear)
        for i in 1:n
            # REAL PV: 1:npv
            # REAL PQ: (npv+1:npv+npq)
            # IMAG PQ: (npv+npq+1:npv+2npq)
            fr = (i <= npv) ? pv[i] : pq[i - npv]
            F[i] -= pinj[fr]
            if i > npv
                F[i + npq] -= qinj[fr]
            end
            @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
                to = ybus_re_rowval[c]
                aij = v_a[fr] - v_a[to]
                # f_re = a * cos + b * sin
                # f_im = a * sin - b * cos
                coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
                coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
                cos_val = cos(aij)
                sin_val = sin(aij)
                F[i] += coef_cos * cos_val + coef_sin * sin_val
                if i > npv
                    F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
                end
            end
        end
    end

    function kernel_roc!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus)
        npv = length(pv)
        npq = length(pq)
        n = npq + npv
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        for i in index:stride:n
            # REAL PV: 1:npv
            # REAL PQ: (npv+1:npv+npq)
            # IMAG PQ: (npv+npq+1:npv+2npq)
            fr = (i <= npv) ? pv[i] : pq[i - npv]
            F[i] -= pinj[fr]
            if i > npv
                F[i + npq] -= qinj[fr]
            end
            for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
                to = ybus_re_rowval[c]
                aij = v_a[fr] - v_a[to]
                # f_re = a * cos + b * sin
                # f_im = a * sin - b * cos
                coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
                coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
                cos_val = AMDGPU.GPUArrays.cos(aij)
                sin_val = AMDGPU.GPUArrays.sin(aij)
                F[i] += coef_cos * cos_val + coef_sin * sin_val
                if i > npv
                    F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
                end
            end
        end
        return nothing
    end

    function kernel_oneapi!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus)
        npv = length(pv)
        npq = length(pq)
        n = npq + npv
        i = get_global_id()
        # REAL PV: 1:npv
        # REAL PQ: (npv+1:npv+npq)
        # IMAG PQ: (npv+npq+1:npv+2npq)
        fr = (i <= npv) ? pv[i] : pq[i - npv]
        F[i] -= pinj[fr]
        if i > npv
            F[i + npq] -= qinj[fr]
        end
        @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
            to = ybus_re_rowval[c]
            aij = v_a[fr] - v_a[to]
            # f_re = a * cos + b * sin
            # f_im = a * sin - b * cos
            coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
            coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
            # TODO: support cos and sin in oneAPI
            #cos_val = oneAPI.GPUArrays.cos(aij)
            #sin_val = oneAPI.GPUArrays.sin(aij)
            cos_val = aij
            sin_val = aij 
            F[i] += coef_cos * cos_val + coef_sin * sin_val
            if i > npv
                F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
            end
        end
        return nothing
    end

    function kernel_cuda!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus)
        npv = length(pv)
        npq = length(pq)
        n = npq + npv
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        for i in index:stride:n
            # REAL PV: 1:npv
            # REAL PQ: (npv+1:npv+npq)
            # IMAG PQ: (npv+npq+1:npv+2npq)
            fr = (i <= npv) ? pv[i] : pq[i - npv]
            F[i] -= pinj[fr]
            if i > npv
                F[i + npq] -= qinj[fr]
            end
            @inbounds for c in ybus_re_colptr[fr]:ybus_re_colptr[fr+1]-1
                to = ybus_re_rowval[c]
                aij = v_a[fr] - v_a[to]
                # f_re = a * cos + b * sin
                # f_im = a * sin - b * cos
                coef_cos = v_m[fr]*v_m[to]*ybus_re_nzval[c]
                coef_sin = v_m[fr]*v_m[to]*ybus_im_nzval[c]
                cos_val = CUDA.cos(aij)
                sin_val = CUDA.sin(aij)
                F[i] += coef_cos * cos_val + coef_sin * sin_val
                if i > npv
                    F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
                end
            end
        end
        return nothing
    end

    function residual!(F, v_m, v_a,
                                    ybus_re, ybus_im,
                                    pinj, qinj, pv, pq, nbus)
        npv = length(pv)
        npq = length(pq)
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        FT = t1s{2}
        V = Vector
        if backend == AMDGPUBackend() 
            T = ROCVector
        elseif backend == CUDABackend()
            T = CuVector
        elseif backend == oneAPIBackend() 
            T = oneArray
        else
            error("Unsupported backend")
        end
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

        dF = T(adF)
        dv_m = T(adv_m)
        dv_a = T(adv_a)
        dybus_re_nzval = T(adybus_re_nzval)
        dybus_re_colptr = T(ybus_re.colptr)
        dybus_re_rowval = T(ybus_re.rowval)
        dybus_im_nzval = T(adybus_im_nzval)
        dybus_im_colptr = T(ybus_im.colptr)
        dybus_im_rowval = T(ybus_im.rowval)
        dpinj = T(adpinj)
        dqinj = T(adqinj)
        dpv = T(pv)
        dpq = T(pq)

        kernel!(backend, dF, dv_m, dv_a,
                        dybus_re_nzval, dybus_re_colptr, dybus_re_rowval,
                        dybus_im_nzval, dybus_im_colptr, dybus_im_rowval,
                        dpinj, dqinj, dpv, dpq, nbus)
        F .= ForwardDiff.value.(dF)
    end

    function kernel_roc!(F, x) 
        n = length(x)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        for i in index:stride:n
            F[i] = x[i] 
        end
    end

    function kernel_cuda!(F, x) 
        n = length(x)
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        for i in index:stride:n
            F[i] = x[i] 
        end
    end
    
    function identity(x)
        t1s{N} = ForwardDiff.Dual{Nothing,Float64, N} where N
        FT = t1s{2}
        V = Vector
        if backend == AMDGPUBackend() 
            T = ROCVector
        elseif backend == CUDABackend()
            T = CuVector
        elseif backend == oneAPIBackend() 
            T = oneArray
        else
            error("Unsupported backend")
        end
        F = copy(x)
        adF = V{FT}(undef, length(F))
        adx = V{FT}(undef, length(x))
        ForwardDiff.seed!(adx, x, one(ForwardDiff.Partials{2,Float64}))
        dF = T(adF)
        dx = T(adx)
        kernel!(backend, dF, dx) 
        function pvalues(x)
            return x.values
        end
        return Array(ForwardDiff.values.(dF)), Array(pvalues.(ForwardDiff.partials.(dF)))
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
