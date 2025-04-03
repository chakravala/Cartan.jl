<p align="center">
  <img src="./docs/src/assets/logo.png" alt="Cartan.jl"/>
</p>

# Cartan.jl

*TensorField topology over FrameBundle ∇ with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) elements*

[![DOI](https://zenodo.org/badge/673606851.svg)](https://zenodo.org/badge/latestdoi/673606851)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cartan.crucialflow.com)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/chakravala/Cartan.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/klhdg493nvs0oi7h?svg=true)](https://ci.appveyor.com/project/chakravala/cartan-jl)

Provides `TensorField{B,F,N} <: GlobalFiber{LocalTensor{B,F},N}` implementation for both a local `ProductSpace` and general `ImmersedTopology` specifications on any `FrameBundle` expressed with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) algebra.
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
Building on this, `Cartan` provides an algebra for `FiberBundle` sections and associated bundles on a manifold, such as general `Connection`, `LieDerivative`, and `CovariantDerivative` operators in terms of `Grassmann` elements.
Calculus of `Variation` fields can also be generated with the combined topology of a `FiberProductBundle`.
Furthermore, the `FiberProduct` structure enables construction of `HomotopyBundle` types.
Utility package for differential geometry and tensor calculus intended for [Adapode.jl](https://github.com/chakravala/Adapode.jl).

The `Cartan` package is intended to standardize the composition of various methods and functors applied to specialized categories transformed with a unified representation over a product topology, especially having fibers of the `Grassmann` algebra.
Initial topologies include `ProductSpace` types and in general the `ImmersedTopology`.
```
Positions{P, G} where {P<:Chain, G} (alias for AbstractArray{<:Coordinate{P, G}, 1} where {P<:Chain, G})
Interval{P, G} where {P<:AbstractReal, G} (alias for AbstractArray{<:Coordinate{P, G}, 1} where {P<:Union{Real, Single{V, G, B, <:Real} where {V, G, B}, Chain{V, G, <:Real, 1} where {V, G}}, G})
IntervalRange{P, G, PA, GA} where {P<:Real, G, PA<:AbstractRange, GA} (alias for GridBundle{1, Coordinate{P, G}, <:PointArray{P, G, 1, PA, GA}} where {P<:Real, G, PA<:AbstractRange, GA})
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
RealSpace{N} where N (alias for AbstractArray{<:Coordinate{P, G}, N} where {N, P<:(Chain{V, 1, <:Real} where V), G})
AlignedRegion{N} where N (alias for GridBundle{N, Coordinate{P, G}, PointArray{P, G, N, PA, GA}} where {N, P<:Chain, G<:InducedMetric, PA<:(ProductSpace{V, <:Real, N, N, <:AbstractRange} where V), GA<:Global})
AlignedSpace{N} where N (alias for GridBundle{N, Coordinate{P, G}, PointArray{P, G, N, PA, GA}} where {N, P<:Chain, G<:InducedMetric, PA<:(ProductSpace{V, <:Real, N, N, <:AbstractRange} where V), GA})
FrameBundle{Coordinate{B,F},N} where {B,F,N}
GridBundle{N,C,PA<:FiberBundle{C,N},TA<:ImmersedTopology} <: FrameBundle{C,N}
SimplexBundle{N,C,PA<:FiberBundle{C,1},TA<:ImmersedTopology} <: FrameBundle{C,1}
FaceBundle{N,C,PA<:FiberBundle{C,1},TA<:ImmersedTopology} <: FrameBundle{C,1}
FiberProductBundle{P,N,SA<:AbstractArray,PA<:AbstractArray} <: FrameBundle{Coordinate{P,InducedMetric},N}
HomotopyBundle{P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray,TA<:ImmersedTopology} <: FrameBundle{Coordinate{P,InducedMetric},N}
```
Visualizing `TensorField` reperesentations can be standardized in combination with [Makie.jl](https://github.com/MakieOrg/Makie.jl) or [UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl).

Due to the versatility of the `TensorField` type instances, it's possible to disambiguate them into these type alias specifications with associated methods:
```Julia
ElementMap (alias for TensorField{B, F, 1, P, A} where {B, F, P<:ElementBundle, A})
SimplexMap (alias for TensorField{B, F, 1, P, A} where {B, F, P<:SimplexBundle, A})
FaceMap (alias for TensorField{B, F, 1, P, A} where {B, F, P<:FaceBundle, A})
IntervalMap (alias for TensorField{B, F, 1, P, A} where {B, F, P<:(AbstractArray{<:Coordinate{P, G}, 1} where {P<:Union{Real, Single{V, G, B, <:Real} where {V, G, B}, Chain{V, G, <:Real, 1} where {V, G}}, G}), A})
RectangleMap (alias for TensorField{B, F, 2, P, A} where {B, F, P<:(AbstractMatrix{<:Coordinate{P, G}} where {P<:(Chain{V, 1, <:Real} where V), G}), A})
HyperrectangleMap (alias for TensorField{B, F, 3, P, A} where {B, F, P<:(AbstractArray{<:Coordinate{P, G}, 3} where {P<:(Chain{V, 1, <:Real} where V), G}), A})
ParametricMap (alias for TensorField{B, F, N, P, A} where {B, F, N, P<:(AbstractArray{<:Coordinate{P, G}, N} where {N, P<:(Chain{V, 1, <:Real} where V), G}), A})
Variation (alias for TensorField{B, F, N, P, A} where {B, F<:TensorField, N, P, A})
RealFunction (alias for TensorField{B, F, 1, P, A} where {B, F<:AbstractReal, PA<:(AbstractVector{<:AbstractReal}), A})
PlaneCurve (alias for TensorField{B, F, N, P, A} where {B, F, N, P<:(AbstractArray{<:Coordinate{P, G}, N} where {N, P<:(Chain{V, 1, <:Real} where V), G}), A})
SpaceCurve (alias for TensorField{B, F, 1, P, A} where {B, F<:(Chain{V, G, Q, 3} where {V, G, Q}), P<:(AbstractVector{<:Coordinate{P, G}} where {P<:AbstractReal, G}), A})
AbstractCurve (alias for TensorField{B, F, 1, P, A} where {B, F<:Chain, P<:(AbstractVector{<:Coordinate{P, G}} where {P<:AbstractReal, G}), A})
SurfaceGrid (alias for TensorField{B, F, 2, P, A} where {B, F<:AbstractReal, P<:(AbstractMatrix{<:Coordinate{P, G}} where {P<:(Chain{V, 1, <:Real} where V), G}), A})
VolumeGrid (alias for TensorField{B, F, 3, P, A} where {B, F<:AbstractReal, P<:(AbstractArray{<:Coordinate{P, G}, 3} where {P<:(Chain{V, 1, <:Real} where V), G}), A})
ScalarGrid (alias for TensorField{B, F, N, P, A} where {B, F<:AbstractReal, N, P<:(AbstractArray{<:Coordinate{P, G}, N} where {P<:(Chain{V, 1, <:Real} where V), G}), A})
DiagonalField (alias for TensorField{B, F, N, P, A} where {B, F<:DiagonalOperator, N, P, A})
EndomorphismField (alias for TensorField{B, F, N, P, A} where {B, F<:(TensorOperator{V, V, T} where {V T<:(TensorAlgebra{V, <:TensorAlgebra{V}})}), N, P, A})
OutermorphismField (alias for TensorField{B, F, N, P, A} where {B, F<:Outermorphism, N, P, A})
CliffordField (alias for TensorField{B, F, N, P, A} where {B, F<:Multivector, N, P, A})
QuaternionField (alias for TensorField{B, F, N, P, A} where {B, F<:(Quaternion), N, P, A})
ComplexMap (alias for TensorField{B, F, N, P, A} where {B, F<:(Union{Complex{T}, Single{V, G, B, Complex{T}} where {V, G, B}, Chain{V, G, Complex{T}, 1} where {V, G}, Couple{V, B, T} where {V, B}, Phasor{V, B, T} where {V, B}} where T<:Real), N, P, A})
PhasorField (alias for TensorField{B, F, N, P, A} where {B, F<:Phasor, N, P, A})
SpinorField (alias for TensorField{B, F, N, P, A} where {B, F<:AbstractSpinor, N, P, A})
GradedField{G} where G (alias for TensorField{B, F, N, P, A} where {G, B, F<:(Chain{V, G} where V), N, P, A})
ScalarField (alias for TensorField{B, F, N, P, A} where {B, F<:Union{Real, Single{V, G, B, <:Real} where {V, G, B}, Chain{V, G, <:Real, 1} where {V, G}}, N, P, A})
VectorField (alias for TensorField{B, F, N, P, A} where {B, F<:(Chain{V, 1} where V), N, P, A})
BivectorField (alias for TensorField{B, F, N, P, A} where {B, F<:(Chain{V, 2} where V), N, P, A})
TrivectorField (alias for TensorField{B, F, N, P, A} where {B, F<:(Chain{V, 3} where V),N, P, A})
```

In the `Cartan` package, a technique is employed where a `TensorField` is constructed from an interval, product manifold, or topology, to generate an algebra of sections which can be used to compose parametric maps on manifolds.
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
