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
Its core architecture accomodates numerical representations of principal G-fiber bundles, product manifolds, and submanifold immersion, providing native support for PDE models defined on structured or unstructured domains.

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

Provides `TensorField{B,F,N} <: FiberBundle{LocalTensor{B,F},N}` implementation for both a local `ProductSpace` and general `ImmersedTopology` specifications on any `FrameBundle` expressed with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) algebra.
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
Building on this, `Cartan` provides an algebra for `FiberBundle` sections and associated bundles on a manifold, such as general `PrincipalFiber`, `Connection`, `LieDerivative`, and `CovariantDerivative` operators in terms of `Grassmann` elements.
Calculus of `Variation` fields can also be generated with the combined topology of a `FiberProductBundle`.
Furthermore, the `FiberProduct` structure enables construction of `HomotopyBundle` types.
Utility package for differential geometry and tensor calculus intended for [Adapode.jl](https://github.com/chakravala/Adapode.jl).

The `Cartan` package is intended to standardize the composition of various methods and functors applied to specialized categories transformed with a unified representation over a product topology, especially having fibers of the `Grassmann` algebra.
Initial topologies include `ProductSpace` types and in general the `ImmersedTopology`.

```@contents
Pages = ["fiber.md","videos.md","library.md","plot.md"]
```

This `Cartan` package for the Julia language was created by [github.com/chakravala](https://github.com/chakravala) for mathematics and computer algebra research with differential geometric algebras.
These projects and repositories were started entirely independently and are available as free software to help spread the ideas to a wider audience.
Please consider donating to show your thanks and appreciation to this project at [liberapay](https://liberapay.com/chakravala), [GitHub Sponsors](https://github.com/sponsors/chakravala), [Patreon](https://patreon.com/dreamscatter), [Tidelift](https://tidelift.com/funding/github/julia/Grassmann), [Bandcamp](https://music.crucialflow.com) or [contribute](https://github.com/chakravala/Grassmann.jl/graphs/contributors) (documentation, tests, examples) in the repositories.
