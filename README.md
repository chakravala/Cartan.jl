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
MeshFunction (alias for TensorField{B, T, F, 1} where {B, T<:ChainBundle, F<:Real})
ElementFunction (alias for TensorField{B, T, F, 1} where {B, T<:AbstractVector{B}, F<:Real})
IntervalMap{B} where B<:Real (alias for TensorField{B, T, F, 1} where {B<:Real, T<:AbstractArray{B, 1}, F})
RealFunction (alias for TensorField{B, T, F, 1} where {B<:Real, T<:AbstractVector{B}, F<:Union{Real, Single, Chain{V, G, <:Real, 1} where {V, G}}})
PlaneCurve (alias for TensorField{B, T, F, 1} where {B<:Real, T<:AbstractVector{B}, F<:(Chain{V, G, Q, 2} where {V, G, Q})})
SpaceCurve (alias for TensorField{B, T, F, 1} where {B<:Real, T<:AbstractVector{B}, F<:(Chain{V, G, Q, 3} where {V, G, Q})})
GridSurface (alias for TensorField{B, T, F, 2} where {B, T<:AbstractMatrix{B}, F<:Real})
GridParametric{B} where B (alias for TensorField{B, T, F} where {B, T<:(AbstractArray{B}), F<:Real})
ComplexMapping (alias for TensorField{B, T, F} where {B, T, F<:Complex})
ComplexMap (alias for TensorField{B, T, F} where {B, T, F<:Couple})
GradedField{G} where G (alias for TensorField{B, T, F} where {G, B, T, F<:(Chain{V, G} where V)})
ScalarField (alias for TensorField{B, T, F} where {B, T, F<:Single})
VectorField (alias for TensorField{B, T, F} where {B, T, F<:(Chain{V, 1} where V)})
SpinorField (alias for TensorField{B, T, F} where {B, T, F<:Spinor})
QuaternionField (alias for TensorField{B, T, F} where {B, T, F<:(Quaternion)})
CliffordField (alias for TensorField{B, T, F} where {B, T, F<:Multivector})
BivectorField (alias for TensorField{B, T, F} where {B, T, F<:(Chain{V, 2} where V)})
TrivectorField (alias for TensorField{B, T, F} where {B, T, F<:(Chain{V, 3} where V)})
```
This package is intended to standardize the composition of various methods and functors applied to specialized categories transformed with a unified representation over a product topology.
```
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
Interval (alias for ProductSpace{V, T, 1, 1} where {V, T})
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
```

Construct `dom → fun == dom → fun.(dom)` category with `\rightarrow` to initialize an example
```julia
julia> using Grassmann, TensorFields, UnicodePlots

julia> dom = (2π:0.01:4π)⊕(0:0.01:2π);

julia> fun(v) = (v[1]-v[2])*cos(v[1]*v[2]/2);

julia> cat = dom → fun; # dom → fun.(dom)

julia> typeof(cat)
TensorField{Chain{⟨++⟩, 1, Float64, 2}, ProductSpace{⟨++⟩, Float64, 2, 2, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}, Float64, 2}

julia> supertype(ans)
FiberBundle{Section{Chain{⟨++⟩, 1, Float64, 2}, Float64}, 2}

julia> contourplot(cat)
     ┌────────────────────────────────────────┐  500
   7 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ ┌──┐
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠤⣄⡀⠠⢤⣀⠀⠀⠀⠀⠀⠀⠠⠤⠄⠀⠀⠀⠀⠀⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠲⣍⡓⠦⢄⣀⠀⠀⠀⠀⠉⠓⠲⢬⣙⡲⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠦⣌⡙⠲⢤⣀⡀⠀⠀⠀⠀⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⢠⣖⠆⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠘⢿⣒⠦⢤⣀⠀⠀⠀⠀⠀⠀⠉⠙⠒⠦⢭⣓⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠈⠓⠦⢬⣉⠓⠦⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠯⣍⣉⠓⠒⠂⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠓⠒⠮⠽⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⡀⠀⠀⠈⠉⠙⠂⠀⠀│ │▄▄│
     │⠀⢐⡒⠦⢤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠲⠤⣄⣉⡉⠉⠓⠒⠲⠄⠀⠀│ │▄▄│
     │⠀⠀⠈⠙⠒⠦⠭⢽⣲⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠉⠉⠙⠒⣒⡆⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⠲⠤⢤⣀⣈⡉⠉⠉⠓⠒⠒⠲⠤⠭⠇⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⢓⣒⣒⣲⣤⣬⡇⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠾⣍⣉⣉⠉⠉⠉⠉⠙⠒⠒⠒⠒⠒⠦⠬⠭⠷⠶⢶⣦⡄⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠓⠒⠒⠒⠒⢦⣤⣤⣤⣼⣿⣿⣯⣭⡅⠀⠀│ │▄▄│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⡠⠤⠤⠤⠤⣄⣀⣀⣀⣀⣀⣀⣛⣛⣒⣦⣤⣤⣤⣬⣭⣍⡁⠀⠀│ │▄▄│
   0 │⠀⠀⠀⠀⠀⠀⠀⠀⠐⠯⢥⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣐⣶⣯⣭⣭⣭⢍⣉⣉⣉⡁⠀⠀│ └──┘
     └────────────────────────────────────────┘ -400
     ⠀6⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀13⠀     
```
Visualizing `TensorField` reperesentations can be standardized in combination with [Makie.jl](https://github.com/MakieOrg/Makie.jl) or [UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl).
```Julia
julia> using Grassmann, TensorFields, UnicodePlots

julia> t = 0:0.01:2π → identity
TensorField{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Float64, 1}
     ┌────────────────────────────────────────┐ 
   7 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠋⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⣀⡴⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⢀⡤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   0 │⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     └────────────────────────────────────────┘ 
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀7⠀ 

julia> cos(3t) + im*sin(2t)
TensorField{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, ComplexF64, 1}
      ┌────────────────────────────────────────┐ 
    1 │⡴⠊⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠑⠒⠒⠤⠤⢤⣀⣀⣇⣀⡠⠤⠔⠒⠒⠊⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠑⢢│ 
      │⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠔⠒⠉⠉⡏⠉⠒⠢⠤⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡼│ 
      │⠀⠙⢦⡀⠀⠀⠀⠀⢀⡠⠔⠒⠋⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠈⠉⠒⠤⣀⠀⠀⠀⠀⠀⣀⡴⠋⠀│ 
      │⠀⠀⠀⠈⠒⣄⡤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠒⢤⣴⠊⠁⠀⠀⠀│ 
      │⠀⠀⢀⠔⠊⠀⠈⠑⠢⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠁⠀⠑⠢⡀⠀⠀│ 
      │⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠈⠒⠤⣀⡀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⣀⠤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀│ 
      │⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠒⠤⣀⠀⠀⡇⠀⣀⠤⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳│ 
      │⡧⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣭⠶⡷⣭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢼│ 
      │⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉⠀⠀⡇⠀⠉⠒⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼│ 
      │⠈⠳⡄⠀⠀⠀⠀⠀⠀⠀⣀⠤⠔⠉⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠑⠒⠤⣀⠀⠀⠀⠀⠀⠀⠀⢀⠞⠁│ 
      │⠀⠀⠈⠲⢄⡀⢀⡠⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⢄⡀⠀⡠⠖⠁⠀⠀│ 
      │⠀⠀⠀⢀⡤⠛⠣⢄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠜⠛⠤⡀⠀⠀⠀│ 
      │⠀⡠⠖⠁⠀⠀⠀⠀⠀⠉⠒⠤⣀⡀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⠤⠒⠊⠁⠀⠀⠀⠀⠈⠳⣄⠀│ 
      │⡞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠒⠢⠤⣄⣀⣇⣀⠤⠴⠒⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳│ 
   -1 │⠣⢄⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠤⠤⠤⠔⠒⠒⠉⠉⡏⠉⠒⠒⠲⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⡠⠞│ 
      └────────────────────────────────────────┘ 
      ⠀-1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀1⠀ 
```
In the above example, a technique is demonstrated where an identity `TensorField` is constructed from an interval, resulting in `t` which can be used to parametrize functions on the complex plane.
Constructing a `TensorField` can be accomplished in various ways,
there are explicit techniques to construct a `TensorField` as well as implicit methods.
Additional packages such as `Adapode` can build on the `TensorField` concept by generating them from differential equations.
```Julia
julia> using Grassmann, TensorFields, Adapode, UnicodePlots

julia> Lorenz(x) = Chain(
               10.0(x[2]-x[1]),
               x[1]*(28.0-x[3])-x[2],
               x[1]*x[2]-(8/3)*x[3]);

julia> sol = odesolve(Lorenz,Chain(10.,10.,10.))
TensorField{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Chain{⟨×××⟩, 1, Float64, 3}, 1}
            ┌────────────────────────────────────────┐   
    42.9279 │⢰⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡄⠀⠀⠀⠀⠀⠀⠀⠀│ y1
            │⢸⢧⠀⠀⢰⡆⠀⠀⢠⢧⠀⠀⠀⠀⣿⡀⠀⠀⣀⠀⠀⢀⣄⠀⠀⠀⣷⠀⠀⠀⢸⡇⠀⠀⢸⣇⠀⠀⠀⠀│ y2
            │⢸⢸⠀⠀⢸⢳⠀⠀⢸⢸⠀⠀⠀⠀⡇⡇⠀⢰⢻⠀⠀⢸⢸⠀⠀⢸⢹⡀⠀⠀⣸⢳⠀⠀⢸⢸⠀⠀⠀⠀│ y3
            │⢸⠸⡄⠀⡏⢸⡀⠀⢸⠘⡆⠀⠀⢠⠇⢧⠀⢸⠈⡇⠀⢸⠘⡆⠀⢸⠀⡇⠀⠀⡇⢸⡀⠀⣸⠈⠀⠀⠀⠀│   
            │⣼⠀⣇⠀⡇⠀⡇⠀⡼⠀⢧⠀⠀⢸⠀⢸⡀⡞⠀⢳⠀⡏⠀⢧⠀⣸⠀⢳⠀⠀⡇⠀⡇⠀⡇⠀⠀⠀⠀⠀│   
            │⣿⠀⠸⣴⠃⠀⢹⡀⡇⠀⠸⡄⠀⢸⠀⠀⠧⠇⠀⠘⣦⠇⠀⠸⡄⣇⠀⠸⡄⠀⣷⠀⢹⠀⡇⠀⠀⠀⠀⠀│   
            │⣿⡇⠀⠉⠀⠀⠀⠳⠃⠀⠀⢧⠀⢸⠀⠀⠀⣶⡀⠀⠀⣿⡀⠀⠙⣿⡆⠀⢳⢸⣿⡄⠈⠿⠁⠀⠀⠀⠀⠀│   
            │⠏⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⡞⠀⠀⢰⡟⣷⠀⢸⡏⣧⠀⢰⡟⣷⠀⠈⣻⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀│   
            │⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⠁⢿⣆⣾⠁⡿⡄⣼⠃⣿⡄⢀⣿⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀│   
            │⣀⣿⣄⣀⣀⣀⣀⣀⣀⣠⣄⣀⣀⣀⣸⣸⣁⣀⣘⣚⣁⣀⣹⣟⣋⣀⣸⣳⣾⣃⣀⣿⣄⣀⣀⣠⣀⣀⣀⣀│   
            │⠀⠳⢽⡄⠀⣯⠿⣦⠀⣞⡏⠈⢿⡄⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⠻⣆⠀⣏⠀⠀⠀⠀│   
            │⠀⠀⠀⣿⣀⡿⠀⢻⡆⣿⠀⠀⠘⣧⣼⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⣿⠀⠀⠀⠀│   
            │⠀⠀⠀⢸⣿⠇⠀⠘⣧⡏⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣷⡇⠀⠀⠀⠀│   
            │⠀⠀⠀⠀⠿⠀⠀⠀⣿⠁⠀⠀⠀⢹⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⠀⠀⠀⠀│   
   -22.4896 │⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀│   
            └────────────────────────────────────────┘   
            ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀7⠀   

julia> typeof(sol) <: SpaceCurve
true

julia> speed(sol)
TensorField{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Single{⟨×××⟩, 0, v, Float64}, 1}
       ┌────────────────────────────────────────┐ 
   300 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⡏⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⡇⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⢧⠀⠀⠀⠀⠀⠀⣸⡄⠀⠀⠀⢸⢸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⡇⡇⠀⠀⣾⡀⠀⠀⠀⠀│ 
       │⠀⢸⠀⠀⣀⠀⠀⠀⡇⡇⠀⠀⠀⡏⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡇⠀⠀⢸⠁⡇⠀⠀⡇⡇⠀⠀⠀⠀│ 
       │⠀⢸⠀⠀⣿⡀⠀⠀⡇⢹⠀⠀⠀⡇⠀⡇⠀⠀⠀⠀⠀⣴⡀⠀⠀⡇⢳⠀⠀⢸⠀⢳⠀⠀⡇⢧⠀⠀⠀⠀│ 
       │⠀⢸⠀⢠⠇⣇⠀⠀⡇⢸⠀⠀⠀⡇⠀⡇⠀⡴⡄⠀⠀⡇⢧⠀⠀⡇⢸⠀⠀⢸⠀⢸⠀⢠⠇⢸⠀⠀⠀⠀│ 
       │⠀⠘⡆⢸⠀⢸⠀⢸⠁⠘⡆⠀⠀⡇⠀⣇⠀⡇⢳⠀⢀⡇⢸⡀⢀⡇⠈⡇⠀⢸⠀⢸⠀⢸⠀⠘⠀⠀⠀⠀│ 
       │⠀⠀⡇⣸⠀⠘⡆⢸⠀⠀⣇⠀⢰⠃⠀⢸⢠⠇⠈⡇⢸⠀⠀⡇⢸⠀⠀⢧⠀⡏⠀⠈⡇⢸⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⠹⠇⠀⠀⢳⡞⠀⠀⠸⡄⢸⠀⠀⠘⡾⠀⠀⠹⠞⠀⠀⠹⠼⠀⠀⠘⣆⡇⠀⠀⢳⡏⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       └────────────────────────────────────────┘ 
       ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀7⠀ 
```
Many of these methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
