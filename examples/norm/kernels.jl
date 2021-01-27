function kernel_roc!(F, x) 
    n = length(x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:n
        F[i] = x[i]*x[i] 
    end
end

function kernel_cuda!(F, x) 
    n = length(x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:n
        F[i] = x[i]*x[i] 
    end
end
