# PFKernels.jl
This is a power flow kernel implemented using [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is used to compute a Jacobian vector product.

This prototype is used as a blue print to integrate oneAPI.jl and AMDGPU.jl into the powerflow solver [ExaPF.jl](https://github.com/exanauts/ExaPF.jl) for reduced methods in optimization.

# Usage
Due to dependency conflicts each target architecture has a `Project.toml` in the corresponding folder (`amd`, `oneapi`, `cuda`). In the architecture folder instantiate the environment and run one of the examples.

```bash
julia --project examples/identity/identity.jl
```

# Examples

Core functions required in [ExaPF.jl](https://github.com/exanauts/ExaPF.jl)

## Identity

Basic kernel implementing the identity

```julia
y .= x
```

The driver computes the identity and its derivative using ForwardDiff.jl

## Norm

Kernel implementing squared norm followed through the point-wise square followed by a summation using `GPUArrays.jl`.

```julia
y .= sum(x.^2)
```

The driver computes the identity and its derivative using ForwardDiff.jl.

## BiCGSTAB

This calls the bicgstab (krylov.jl) of [Kyrlov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) or executes a custom BiCGSTAB implementation (handwritten.jl).

## PFKernel

`PFKernel.jl` implements the balance equations of a power flow kernel.