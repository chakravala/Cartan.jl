<p align="center">
  <img src="./docs/src/assets/logo.png" alt="Cartan.jl"/>
</p>

# Cartan.jl

*Maurer-Cartan-Lie frame connections ∇ [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) TensorField derivations*

[![DOI](https://zenodo.org/badge/673606851.svg)](https://zenodo.org/badge/latestdoi/673606851)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/dev)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build status](https://ci.appveyor.com/api/projects/status/klhdg493nvs0oi7h?svg=true)](https://ci.appveyor.com/project/chakravala/cartan-jl)

Provides `TensorField{B,F,N} <: GlobalFiber{LocalTensor{B,F},N}` implementation for both a local `ProductSpace` topology and the simplicial mesh topologies imported with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl).
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
Building on this, `Cartan` provides an algebra for any `GlobalSection` and associated bundles on a manifold, such as general `Connection` and `CovariantDerivative` operators in terms of `Grassmann` elements.
Utility package for differential geometry and tensor calculus intended for packages such as [Adapode.jl](https://github.com/chakravala/Adapode.jl).

The `Cartan` package is intended to standardize the composition of various methods and functors applied to specialized categories transformed with a unified representation over a product topology, especially having fibers of the `Grassmann` algebra.
Initial topologies include `ProductSpace` types and implicitly simplicial manifolds.
```
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
Interval (alias for ProductSpace{V, T, 1, 1} where {V, T})
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
```
Visualizing `TensorField` reperesentations can be standardized in combination with [Makie.jl](https://github.com/MakieOrg/Makie.jl) or [UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl).

Due to the versatility of the `TensorField` type instances, it's possible to disambiguate them into these type alias specifications with associated methods:
```Julia
MeshFunction (alias for TensorField{B, F, 1, BA} where {B, F<:AbstractReal, BA<:ChainBundle})
ElementFunction (alias for TensorField{B, F, 1, PA} where {B, F<:AbstractReal, PA<:(AbstractVector)})
IntervalMap (alias for TensorField{B, F, 1, PA} where {B, F, PA<:(AbstractArray{<:Union{Real, Single{V, G, B, <:Real} where {V, G, B}, Chain{V, G, <:Real, 1} where {V, G}}, 1})})
RectangleMap (alias for TensorField{B, F, 2, BA} where {B, F, BA<:(ProductSpace{V, T, 2, 2} where {V, T})})
HyperrectangleMap (alias for TensorField{B, F, 3, PA} where {B, F, PA<:(ProductSpace{V, T, 3, 3} where {V, T})})
ParametricMap (alias for TensorField{B, F, N, PA} where {B, F, N, PA<:(ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})})
RealFunction (alias for TensorField{B, F, 1, PA} where {B, F<:AbstractReal, PA<:(AbstractVector{<:AbstractReal})})
PlaneCurve (alias for TensorField{B, F, 1, PA} where {B, F<:(Chain{V, G, Q, 2} where {V, G, Q}), PA<:(AbstractVector{<:AbstractReal})})
SpaceCurve (alias for TensorField{B, F, 1, PA} where {B, F<:(Chain{V, G, Q, 3} where {V, G, Q}), PA<:(AbstractVector{<:AbstractReal})})
SurfaceGrid (alias for TensorField{B, F, 2, PA} where {B, F<:AbstractReal, PA<:(AbstractMatrix)})
VolumeGrid (alias for TensorField{B, F, 3, PA} where {B, F<:AbstractReal, PA<:(AbstractArray{P, 3} where P)})
ScalarGrid (alias for TensorField{B, F, N, PA} where {B, F<:AbstractReal, N, PA<:AbstractArray})
GlobalFrame{B, N, N} where {B<:(LocalFiber{P, <:TensorNested} where P), N, N} (alias for Cartan.GlobalSection{B, N, N1, BA, FA} where {B<:(LocalFiber{P, <:TensorNested} where P), N, N1, BA, FA<:AbstractArray{N, N1}})
DiagonalField (alias for TensorField{B, F} where {B, F<:DiagonalOperator})
EndomorphismField (alias for TensorField{B, F} where {B, F<:(TensorOperator{V, V, T} where {V, T<:(TensorAlgebra{V, <:TensorAlgebra{V}})})})
OutermorphismField (alias for TensorField{B, F} where {B, F<:Outermorphism})
CliffordField (alias for TensorField{B, F} where {B, F<:Multivector})
QuaternionField (alias for TensorField{B, F} where {B, F<:(Quaternion)})
ComplexMap (alias for TensorField{B, F} where {B, F<:(Union{Complex{T}, Single{V, G, B, Complex{T}} where {V, G, B}, Chain{V, G, Complex{T}, 1} where {V, G}, Couple{V, B, T} where {V, B}, Phasor{V, B, T} where {V, B}} where T<:Real)})PhasorField (alias for TensorField{B, T, F} where {B, T, F<:Phasor})
SpinorField (alias for TensorField{B, F} where {B, F<:AbstractSpinor})
GradedField{G} where G (alias for TensorField{B, F} where {G, B, F<:(Chain{V, G} where V)})
ScalarField (alias for TensorField{B, F} where {B, F<:Union{Real, Single{V, G, B, <:Real} where {V, G, B}, Chain{V, G, <:Real, 1} where {V, G}}})
VectorField (alias for TensorField{B, F} where {B, F<:(Chain{V, 1} where V)})
BivectorField (alias for TensorField{B, F} where {B, F<:(Chain{V, 2} where V)})
TrivectorField (alias for TensorField{B, F} where {B, F<:(Chain{V, 3} where V)})
```

In the `Cartan` package, a technique is employed where an identity `TensorField` is constructed from an interval or product manifold, to generate an algebra of sections which can be used to compose parametric maps on manifolds.
Constructing a `TensorField` can be accomplished in various ways,
there are explicit techniques to construct a `TensorField` as well as implicit methods.
Additional packages such as `Adapode` build on the `TensorField` concept by generating them from differential equations.
Many of these methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.

```
 _________                __                  __________
 \_   ___ \_____ ________/  |______    ____   \\       /
 /    \  \/\__  \\_  __ \   __\__  \  /    \   \\     /
 \     \____/ __ \|  | \/|  |  / __ \|   |  \   \\   /
  \______  (____  /__|   |__| (____  /___|  /    \\ /
         \/     \/                 \/     \/      \/
```
developed by [chakravala](https://github.com/chakravala) with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl)
