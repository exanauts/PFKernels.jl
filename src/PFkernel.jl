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

    function residualFunction_polar!(F, v_m, v_a,
                                    ybus_re, ybus_im,
                                    pinj, qinj, pv, pq, nbus)
        npv = length(pv)
        npq = length(pq)
        residual_kernel!(F, v_m, v_a,
                    ybus_re.nzval, ybus_re.colptr, ybus_re.rowval,
                    ybus_im.nzval, ybus_im.colptr, ybus_im.rowval,
                    pinj, qinj, pv, pq, nbus)
    end
end