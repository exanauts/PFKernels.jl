# PFkernel.jl
This is a power flow kernel implemented using [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is used to compute a Jacobian vector product.

This prototype is used as a blue print to integrate oneAPI.jl and AMDGPU.jl into the powerflow solver [ExaPF.jl](https://github.com/exanauts/ExaPF.jl) for reduced methods in optimization.

# Usage
Due to dependency conflicts each target architecture has a `Project.toml` in the corresponding folder (`amd`, `oneapi`, `cuda`). In the architecture folder instantiate the environment and run

```bash
julia --project examples/main.jl
```

By default, the code tries to load one of the GPU packages `AMDGPU.jl`, `oneAPI.jl` or `CUDA.jl`.
