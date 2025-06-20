<p align="center">
  <img src="./docs/src/assets/logo.png" alt="Cartan.jl"/>
</p>

# Cartan.jl

*TensorField topology over FrameBundle ∇ with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) elements*

[![DOI](https://zenodo.org/badge/673606851.svg)](https://zenodo.org/badge/latestdoi/673606851)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cartan.crucialflow.com)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/chakravala/Cartan.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/klhdg493nvs0oi7h?svg=true)](https://ci.appveyor.com/project/chakravala/cartan-jl)
[![PDF 2021](https://img.shields.io/badge/PDF-2021-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=differential-geometric-algebra-2021.pdf)
[![PDF 2025](https://img.shields.io/badge/PDF-2025-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf)

*Cartan.jl* introduces a pioneering unified numerical framework for comprehensive differential geometric algebra, purpose-built for the formulation and solution of partial differential equations on manifolds with non-trivial topological structure and [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) algebra.
Written in Julia, [Cartan.jl](https://github.com/chakravala/Cartan.jl) unifies differential geometry, geometric algebra, and tensor calculus with support for fiber product topology; enabling directly executable generalized treatment of geometric PDEs over grids, meshes, and simplicial decompositions.

The system supports intrinsic formulations of differential operators (including the exterior derivative, codifferential, Lie derivative, interior product, and Hodge star) using a coordinate-free algebraic language grounded in Grassmann-Cartan multivector theory.
Its core architecture accomodates numerical representations of fiber bundles, product manifolds, and submanifold immersion, providing native support for PDE models defined on structured or unstructured domains.

*Cartan.jl* integrates naturally with simplex-based finite element exterior calculus, allowing for geometrical discretizations of field theories and conservation laws.
With its synthesis of symbolic abstraction and numerical execution, *Cartan.jl* empowers researchers to develop PDE models that are simultaneously founded in differential geometry, algebraically consistent, and computationally expressive, opening new directions for scientific computing at the interface of geometry, algebra, and analysis.

```
 _________                __                  __________
 \_   ___ \_____ ________/  |______    ____   \\       /
 /    \  \/\__  \\_  __ \   __\__  \  /    \   \\     /
 \     \____/ __ \|  | \/|  |  / __ \|   |  \   \\   /
  \______  (____  /__|   |__| (____  /___|  /    \\ /
         \/     \/                 \/     \/      \/
```
developed by [chakravala](https://github.com/chakravala) with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl)

## Tensor field topology and fiber bundles

Provides `TensorField{B,F,N} <: GlobalFiber{LocalTensor{B,F},N}` implementation for both a local `ProductSpace` and general `ImmersedTopology` specifications on any `FrameBundle` expressed with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) algebra.
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
Building on this, `Cartan` provides an algebra for `FiberBundle` sections and associated bundles on a manifold, such as general `Connection`, `LieDerivative`, and `CovariantDerivative` operators in terms of `Grassmann` elements.
Calculus of `Variation` fields can also be generated with the combined topology of a `FiberProductBundle`.
Furthermore, the `FiberProduct` structure enables construction of `HomotopyBundle` types.
Utility package for differential geometry and tensor calculus intended for [Adapode.jl](https://github.com/chakravala/Adapode.jl).

**Definition**. Commonly used fundamental building blocks are
* `ProductSpace{V,K,N} <: AbstractArray{Chain{V,1,K,N},N}`
    * uses Cartesian products of interval subsets of real line products
    * generates lazy array of `Chain{V,1}` point vectors from input ranges
* `Global{N,T}` represents array with same `T` value at all indices
* `LocalFiber{B,F}` has a local `basetype` of `B` and `fibertype` of `F`
    * `Coordinate{P,G}` has `pointtype` of `P` and `metrictype` of `G`
* `ImmersedTopology{N,M} = AbstractArray{Values{N,Int},M}`
    * `ProductTopology` generates basic product topologies for grids
    * `SimplexTopology` defines continuous simplex immersion
    * `DiscontinuousTopology` disconnects for discontinuous
    * `LagrangeTopology` extends for Lagrange polynomial base
    * `QuotientTopology` defines classes of quotient identification

Generalizing upon `ProductTopology`, the `QuotientTopology` defines a quotient identification across the boundary fluxes of the region,
which then enables different specializations of `QuotientTopology` as

* `OpenTopology`: all boundaries don't have accumulation points,
* `CompactTopology`: all points have a neighborhood topology,
* `CylinderTopology`: closed ribbon with two edge open endings,
* `MobiusTopology`: twisted ribbon with one edge open ending,
* `WingTopology`: upper and lower surface topology of wing,
* `MirrorTopology`: reflection boundary along mirror (basis) axis,
* `ClampedTopology`: each boundary face is reflected to be compact,
* `TorusTopology`: generalized compact torus up to 5 dimensions,
* `HopfTopology`: compact topology of the Hopf fibration in 3D,
* `KleinTopology`: compact topology of the Klein bottle domain,
* `PolarTopology`: polar compactification with open edge boundary,
* `SphereTopology`: generalized mathematical sphere, compactified,
* `GeographicTopology`: axis swapped from `SphereTopology` in 2D.

Combination of `PointArray <: Coordinates` and `ImmersedTopology` leads into definition of `TensorField` as a global section of a `FrameBundle`.

Utility methods for `QuotientTopology` include `isopen` and `iscompact`, while utility methods for `SimplexTopology` include `nodes`, `sdims`, `subelements`, `subimmersion`, `topology`, `totalelements`, `totalnodes`, `vertices`, `elements`, `isfull`, `istotal`, `iscover`, `isdiscontinuous`, `isdisconnected`, `continuous`, `disconnect`, `getfacet`, `getimage`, `edges`, `facets`, `complement`, `interior`, `∂`, `degrees`, `weights`, `adjacency`, `antiadjacency`, `incidence`, `laplacian`, `neighbors`.

**Definition**. A *fiber bundle* is a manifold `E` having projection which commutes with local trivializations paired to neighborhoods of manifold `B`, where `B` is the `basetype` and `F` is the `fibertype` of `E`.

`FiberBundle{E,N} <: AbstractArray{E,N}` where `E` is the `eltype`
* `Coordinates{P,G,N} <: FiberBundle{Coordinate{P,G},N}`
    * `PointArray{P,G,N}` has `pointtype` of `P`, `metrictype` of `G`
    * `FiberProduct` introduces fiber product structure for manifolds
* `FrameBundle{C,N}` has `coordinatetype` of `C` and `immersion`
    * `GridBundle{N,C}` `N`-grid with `coordianates` and `immersion`
    * `SimplexBundle{N,C}` defines `coordinates` and an `immersion`
    * `FaceBundle{N,C}` defines `element` faces and their `immersion`
    * `FiberProductBundle` for extruding dimensions from simplices
    * `HomotopyBundle` encapsulates a variation as `FrameBundle`
* `TensorField` defines fibers in a global section of a `FrameBundle`

When a `TensorField` has a `fibertype` from `<:TensorGraded{V,g}`, then it is a grade `g` differential form.
In general the `TensorField` type can deal with more abstract `fibertype` varieties than only those used for differential forms, as it unifies many different forms of tensor analysis.

By default, the `InducedMetric` is defined globally in each `PointArray`, unless a particular metric tensor specification is provided.
When the default `InducedMetric` is invoked, the metric tensor from the `TensorAlgebra{V}` type is used for the global manifold, instead of the extra allocation to specify metric tensors at each point.
`FrameBundle` then defines local charts along with metric tensor in a `PointArray` and pairs it with an `ImmersedTopology`.
Then the fiber of a `FrameBundle` section is a fiber of a `TensorField`.

These methods relate to `FrameBundle` and `TensorField` instances
* `coordinates(m::FiberBundle)` returns `Coordinates` data type
* `coordinatetype` return applies to `FiberBundle` or `LocalFiber`
* `immersion(m::FiberBundle)` returns `ImmersedTopology` data
* `immersiontype` return applies to `FiberBundle` or `LocalFiber`
* `base` returns the `B` element of a `LocalFiber{B,F}` or `FiberBundle`
* `basetype` returns type `B` of a `LocalFiber{B,F}` or `FiberBundle`
* `fiber` returns the `F` element of `LocalFiber{B,F}` or `FiberBundle`
* `fibertype` returns the `F` type of `LocalFiber{B,F}` or `FiberBundle`
* `points` returns `AbstractArray{P}` data for `Coordinates{P,G}`
* `pointtype` is type `P` of `Coordinate{P,G}` or `Coordinates{P,G}`
* `metrictensor` returns the grade 1 block of the `metricextensor`
* `metricextensor` is `AbstractArray{G}` data for `Coordinates{P,G}`
* `metrictype` is type `G` of `Coordinate{P,G}` or `Coordinates{P,G}`
* `fullcoordinates` returns full `FiberBundle{Coordinate{P,G}}`
* `fullimmersion` returns superset `ImmersedTopology` which `isfull`
* `fulltopology` returns composition of `topology ∘ fullimmersion`
* `fullvertices` list of `vertices` associated to the `fullimmersion`
* `fullpoints` is full `AbstractArray{P}` instead of possibly subspace
* `fullmetricextensor` is full `AbstractArray{G}` instead of subspace
* `isinduced` is true if the `metrictype` is an `InducedMetric` type
* `bundle` returns the integer identification of bundle cache

Various interpolation methods are also supported and can be invoked by applying `TensorField` instances as function evaluations on base manifold or applying some form of resampling method to the manifold topology.
Some utility methods include `volumes`, `initmesh`, `refinemesh`, `affinehull`, `affineframe`, `gradienthat`, etc.

For `GridBundle` initialization it is typical to invoke a combination of `ProductSpace` and `QuotientTopology`, while optional Julia packages extend `SimplexBundle` initialization, such as
[Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl),
[GeometryBasics.jl](https://github.com/JuliaGeometry/GeometryBasics.jl),
[Delaunay.jl](https://github.com/eschnett/Delaunay.jl),
[QHull.jl](https://github.com/JuliaPolyhedra/QHull.jl),
[MiniQhull.jl](https://github.com/gridap/MiniQhull.jl),
[Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl),
[TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl),
[MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl),
[FlowGeometry.jl](https://github.com/chakravala/FlowGeometry.jl).

Standard differential geometry methods for curves includes `integral`, `integrate`, `arclength`, `tangent`, `unittangent`, `speed`, `normal`, `unitnormal`, `curvature`, `radius`, `evolute`, `involute`, `osculatingplane`, `unitosculatingplane`, `binormal`, `unitbinormal`, `torsion`, `frame`, `unitframe`, `curvatures`, `bishopframe`, `bishopunitframe`, `bishoppolar`, `bishop`, `planecurve`, `linkmap`, and `linknumber`.
Standard differential geometry methods for surfaces includees `graph`, `normal`, `unitnormal`, `normalnorm`, `jacobian`, `weingarten`, `sectordet`, `sectorintegral`, `sectorintegrate`, `surfacearea`, `surfacemetric`, `surfacemetricdiag`, `surfaceframe`, `frame`, `unitframe`, `firstform`, `firstformdiag`, `secondform`, `thirdform`, `shape`, `principals`, `principalaxes`, `curvatures`, `meancurvature`, `gaussintrinsic`, `gaussextrinsic`, `gausssign`.

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

## References

* Michael Reed, [Differential geometric algebra with Leibniz and Grassmann](https://crucialflow.com/grassmann-juliacon-2019.pdf) (2019)
* Michael Reed, [Foundatons of differential geometric algebra](https://vixra.org/abs/2304.0228) (2021)
* Michael Reed, [Multilinear Lie bracket recursion formula](https://vixra.org/abs/2412.0034) (2024)
* Michael Reed, [Differential geometric algebra: compute using Grassmann.jl and Cartan.jl](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf) (2025)
* Emil Artin, [Geometric Algebra](https://archive.org/details/geometricalgebra033556mbp) (1957)
* John Browne, [Grassmann Algebra, Volume 1: Foundations](https://www.grassmannalgebra.com/) (2011)
* C. Doran, D. Hestenes, F. Sommen, and N. Van Acker, [Lie groups as spin groups](http://geocalc.clas.asu.edu/pdf/LGasSG.pdf), J. Math Phys. (1993)
* David Hestenes, [Tutorial on geometric calculus](http://geocalc.clas.asu.edu/pdf/Tutorial%20on%20Geometric%20Calculus.pdf). Advances in Applied Clifford Algebra (2013)
* Vladimir and Tijana Ivancevic, [Undergraduate lecture notes in DeRahm-Hodge theory](https://arxiv.org/abs/0807.4991). arXiv (2011)

