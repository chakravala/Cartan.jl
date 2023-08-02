<p align="center">
  <img src="./docs/src/assets/logo.png" alt="TensorFields.jl"/>
</p>

# TensorFields.jl

*TensorField with product topology using [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) element parameters*

[![DOI](https://zenodo.org/badge/223493781.svg)](https://zenodo.org/badge/latestdoi/223493781)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/dev)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build status](https://ci.appveyor.com/api/projects/status/oxi2qutlsaytloap?svg=true)](https://ci.appveyor.com/project/chakravala/tensorfields-jl)

Provides `TensorField{R,B,T,N} <: FiberBundle{Section{R,T},N}` implementation for both a local `ProductSpace` topology and the simplicial mesh topologies imported with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl).
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
Utility package for differential geometry and tensor calculus intended for packages such as [Adapode.jl](https://github.com/chakravala/Adapode.jl).

```Julia
MeshFunction (alias for TensorField{R, B, T, 1} where {R, B<:ChainBundle, T<:Real})
ElementFunction (alias for TensorField{R, B, T, 1} where {R, B<:AbstractVector{R}, T<:Real})
IntervalMap{R} where R<:Real (alias for TensorField{R, B, T, 1} where {R<:Real, B<:AbstractArray{R, 1}, T})
RealFunction (alias for TensorField{R, B, T, 1} where {R<:Real, B<:AbstractVector{R}, T<:Union{Real, Single, Chain{V, G, <:Real, 1} where {V, G}}})
PlaneCurve (alias for TensorField{R, B, T, 1} where {R<:Real, B<:AbstractVector{R}, T<:(Chain{V, G, Q, 2} where {V, G, Q})})
SpaceCurve (alias for TensorField{R, B, T, 1} where {R<:Real, B<:AbstractVector{R}, T<:(Chain{V, G, Q, 3} where {V, G, Q})})
GridSurface (alias for TensorField{R, B, T, 2} where {R, B<:AbstractMatrix{R}, T<:Real})
GridParametric{R} where R (alias for TensorField{R, B, T} where {R, B<:(AbstractArray{R}), T<:Real})
ComplexMapping (alias for TensorField{R, B, T} where {R, B, T<:Complex})
ComplexMap (alias for TensorField{R, B, T} where {R, B, T<:Couple})
ScalarField (alias for TensorField{R, B, T} where {R, B, T<:Single})
VectorField (alias for TensorField{R, B, T} where {R, B, T<:(Chain{V, 1} where V)})
SpinorField (alias for TensorField{R, B, T} where {R, B, T<:Spinor})
TensorField{R, B, T} where {R, B, T<:(Quaternion)}
CliffordField (alias for TensorField{R, B, T} where {R, B, T<:Multivector})
BivectorField (alias for TensorField{R, B, T} where {R, B, T<:(Chain{V, 2} where V)})
TrivectorField (alias for TensorField{R, B, T} where {R, B, T<:(Chain{V, 3} where V)})
```
This package is intended to standardize the composition of various methods and functors applied to specialized categories transformed with a unified representation over a product topology.
```
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
Interval (alias for ProductSpace{V, T, 1, 1} where {V, T})
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
```

Construct `rect` and `fun` to initialize an example
```julia
julia> using Grassmann, TensorFields

julia> rect = (0:0.1:2π)⊕(0:0.1:2π);

julia> fun(t::Chain) = (t[1]-t[2])*cos(t[2]);

julia> tf = TensorField(rect,fun);

julia> typeof(tf)
TensorField{Chain{⟨++⟩, 1, Float64, 2}, ProductSpace{⟨++⟩, Float64, 2, 2, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, Float64, 2}

julia> typeof(tangent(tf))
VectorField{Chain{⟨++⟩, 1, Float64, 2}, ProductSpace{⟨++⟩, Float64, 2, 2, StepRangeLen{Float64, TwicePrecision{Float64}, TwicePrecision{Float64}, Int64}}, Chain{⟨××⟩, 1, Float64, 2}, 2} (alias for TensorField{Chain{⟨++⟩, 1, Float64, 2}, ProductSpace{⟨++⟩, Float64, 2, 2, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, Chain{⟨××⟩, 1, Float64, 2}, 2})
```
Many of these methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
```Julia
julia> F(t) = Chain(cos(t)+t*sin(t),sin(t)-t*cos(t),t^2)

julia> TensorField(0:0.01:2π,F)
```
Visualizing `TensorField` reperesentations can be standardized in combination with packages such as [Makie.jl](https://github.com/MakieOrg/Makie.jl) or [UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl).
