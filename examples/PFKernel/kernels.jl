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