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
    
    function residualFunction(V, Ybus, Sbus, pv, pq)
        # form mismatch vector
        mis = V .* conj(Ybus * V) - Sbus
        # form residual vector
        F = [real(mis[pv])
            real(mis[pq])
            imag(mis[pq]) ]
        return F
    end

    function residual_kernel!(F, v_m, v_a,
                                  ybus_re_nzval, ybus_re_colptr, ybus_re_rowval,
                                  ybus_im_nzval, ybus_im_colptr, ybus_im_rowval,
                                  pinj, qinj, pv, pq, nbus)
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

    function residual_kernel_roc!(F, v_m, v_a,
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
                cos_val = AMDGPU.cos(aij)
                sin_val = AMDGPU.sin(aij)
                F[i] += coef_cos * cos_val + coef_sin * sin_val
                if i > npv
                    F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
                end
            end
        end
    end

    function residual_kernel_oneapi!(F, v_m, v_a,
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
            cos_val = oneAPI.cos(aij)
            sin_val = oneAPI.sin(aij)
            F[i] += coef_cos * cos_val + coef_sin * sin_val
            if i > npv
                F[npq + i] += coef_cos * sin_val - coef_sin * cos_val
            end
        end
    end

    function residual_kernel_cuda!(F, v_m, v_a,
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
    end

    function residualFunction_polar!(F, v_m, v_a,
                                    ybus_re, ybus_im,
                                    pinj, qinj, pv, pq, nbus, gpu)
        npv = length(pv)
        npq = length(pq)
        if gpu == "amd" 
            T = HSAArray
        end
        # if gpu == "nvidia" 
        #     T = CuArray
        # end
        # if gpu == "intel" 
        #     T = oneArray
        # end
        dF = T(F)
        dv_m = T(v_m)
        dv_a = T(v_a)
        dybus_re_nzval = T(ybus_re.nzval)
        dybus_re_colptr = T(ybus_re.colptr)
        dybus_re_rowval = T(ybus_re.rowval)
        dybus_im_nzval = T(ybus_im.nzval)
        dybus_im_colptr = T(ybus_im.colptr)
        dybus_im_rowval = T(ybus_im.rowval)
        dpinj = T(pinj)
        dqinj = T(qinj)
        dpv = T(pv)
        dpq = T(pq)

        if gpu == "amd" 
            wait(@roc residual_kernel_roc!(dF, dv_m, dv_a,
                        dybus_re_nzval, dybus_re_colptr, dybus_re_rowval,
                        dybus_im_nzval, dybus_im_colptr, dybus_im_rowval,
                        dpinj, dqinj, dpv, dpq, nbus))
        end
        # if gpu == "nvidia" 
        #     wait(@cuda residual_kernel_cuda!(dF, dv_m, dv_a,
        #                 dybus_re_nzval, dybus_re_colptr, dybus_re_rowval,
        #                 dybus_im_nzval, dybus_im_colptr, dybus_im_rowval,
        #                 dpinj, dqinj, dpv, dpq, nbus))
        # end
        # if gpu == "intel" 
        #     @oneapi residual_kernel_cuda!(dF, dv_m, dv_a,
        #                 dybus_re_nzval, dybus_re_colptr, dybus_re_rowval,
        #                 dybus_im_nzval, dybus_im_colptr, dybus_im_rowval,
        #                 dpinj, dqinj, dpv, dpq, nbus)
        # end
        F .= dF
    end
end
