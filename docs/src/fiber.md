# Cartan.jl language design

*TensorField topology over FrameBundle ∇ with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) elements*

[![DOI](https://zenodo.org/badge/673606851.svg)](https://zenodo.org/badge/latestdoi/673606851)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cartan.crucialflow.com)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/chakravala/Cartan.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/klhdg493nvs0oi7h?svg=true)](https://ci.appveyor.com/project/chakravala/cartan-jl)
[![PDF 2021](https://img.shields.io/badge/PDF-2021-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=differential-geometric-algebra-2021.pdf)
[![PDF 2025](https://img.shields.io/badge/PDF-2025-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf)

```@contents
Pages = ["index.md","fiber.md","videos.md","library.md"]
```

Initially, *Grassmann.jl* and *Cartan.jl* introduced the *DirectSum* formalism into computational language design for differential geometric algebra, thus enabling the construction of custom algebras through metaprogramming.
*Grassmann.jl* pioneered a novel type system design along with its syntax and semantics, which has undergone significant refinement through many years of development and continuous improvement.
*Cartan.jl* represents a groundbreaking extension of capabilities offered by *Grassmann.jl*, marking a pioneering fully realized implementation of numerical differential geometric algebra, based on `TensorField` representations over a `FrameBundle` and the `ImmersedTopology` of the `FiberBundle`.
*Grassmann.jl* and *Cartan.jl* build on Julia's multiple dispatch and metaprogramming capabilities, presenting a new computational language design approach to interfacing with differential geometric algebra based on a new *sector integral theorem*.
This pioneering design not only actualizes but also elevates computational language syntax to new heights using the foundations of *Grassmann.jl* and *Cartan.jl*.
The *Grassmann.jl* and *Cartan.jl* packages introduce pioneering computational language designs, having inspired imitation projects and thereby validating the project's relevance as significant advance in computational mathematics.

Packages *Grassmann.jl* and *Cartan.jl* can be used as universal language for finite element methods based on a discrete manifold bundle.
Tools built on these foundations enable computations based on multi-linear algebra and spin groups using the geometric algebra known as Grassmann algebra or Clifford algebra.
This foundation is built on a `DirectSum` parametric type system for tangent bundles and vector spaces generating the algorithms for local tangent algebras in a global context.
Geometric algebra mathematical foundations for differential geometry can be used to simplify the Maxwell equations to a single wave equation due to the geometric product.
With this unifying mathematical foundation, it is possible to improve efficiency of multi-disciplinary research using geometric tensor calculus by relying on universal mathematical principles.
Tools built on this differential geometric algebra provide an excellent language for the newly presented *sector integral theorem*, the Helmholtz decomposition, and Hodge-DeRahm co/homology.

The *Grassmann.jl* package provides tools for computations based on multi-linear algebra and spin groups using the extended geometric algebra known as Grassmann-Clifford-Hodge algebra.
Algebra operations include exterior, regressive, inner, and geometric, along with the Hodge star and boundary operators.
Code generation enables concise usage of the algebra syntax.
*DirectSum.jl* multivector parametric type polymorphism is based on tangent vector spaces and conformal projective geometry.
Additionally, the universal interoperability between different sub-algebras is enabled by *AbstractTensors.jl*, on which the type system is built.
The design is based on `TensorAlgebra{V}` abstract type interoperability from *AbstractTensors.jl* with a ``\mathbb{K}``-module type parameter ``V`` from *DirectSum.jl*.
Abstract vector space type operations happen at compile-time, resulting in a differential geometric algebra of multivectors.

Building on the *Grassmann.jl* foundation, the *Cartan.jl* extension then defines `TensorField{B,F,N} <: GlobalFiber{LocalTensor{B,F},N}` for both a local `ProductSpace` and general `ImmersedTopology` specifications on any `FrameBundle` expressed with *Grassmann.jl* algebra.
Many of these modular methods can work on input meshes or product topologies of any dimension, although there are some methods which are specialized.
`Cartan` provides an algebra for `FiberBundle` sections and any associated bundles on a manifold in terms of `Grassmann` elements.
Calculus of `Variation` fields can also be generated with the combined topology of a `FiberProductBundle`.
Furthermore, the `FiberProduct` enables construction of `HomotopyBundle` types.
The `Cartan` package standardizes composition of various methods and functors applied to specialized categories transformed in terms of a unified representation over a product topology, especially having fibers of the `Grassmann` algebra and using `Cartan` methods over a `FrameBundle`.

## Grassmann.jl API design overview

The `AbstractTensors` package is intended for universal interoperation of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes have type parameter ``V``, used to store a `Submanifold{M}` value, which is parametrized by ``M`` the `TensorBundle` choice.
This means that different tensor types can have a commonly shared underlying ``\mathbb{K}``-module parametric type expressed by defining `V::Submanifold{M}`.
Each `TensorAlgebra` subtype must be accompanied by a corresponding `TensorBundle` parameter, which is fully static at compile time.
Due to the parametric type system for the ``\mathbb{K}``-module types, the compiler can fully pre-allocate and often cache.

Let ``M = T^\mu V`` be a ``\mathbb{K}``-module of rank ``n``, then an instance for
``T^\mu V`` can be the tuple ``(n,\mathbb{P},g,\nu,\mu)`` with ``\mathbb{P}\subseteq \langle v_\infty,v_\emptyset\rangle`` specifying the presence of the projective basis and ``g:V\times V\rightarrow\mathbb{K}`` is a metric tensor specification.
The type `TensorBundle{n,```\mathbb{P}```,g,```\nu```,```\mu```}` encodes this information as *byte-encoded* data available at pre-compilation,
where ``\mu`` is an integer specifying the order of the tangent bundle (i.e. multiplicity limit of the Leibniz-Taylor monomials).
Lastly, ``\nu`` is the number of tangent variables.
```math
\langle v_1,\dots,v_{n-\nu},\partial_1,\dots,\partial_\nu\rangle=M\leftrightarrow M' = \langle w_1,\dots,w_{n-\nu},\epsilon_1,\dots,\epsilon_\nu\rangle
```
where ``v_i`` and ``w_i`` are bases for the vectors and covectors, while ``\partial_i`` and ``\epsilon_j`` are bases for differential operators and scalar functions.
The purpose of the `TensorBundle` type is to specify the ``\mathbb{K}``-module basis at compile time.
When assigned in a workspace, `V = Submanifold(::TensorBundle)` is used.

The metric signature of the `Submanifold{V,1}` elements of a vector space ``V`` can be specified with the `V"..."` by using ``+`` or ``-`` to specify whether the `Submanifold{V,1}` element of the corresponding index squares to ``+1`` or ``-1``.
For example, `S"+++"` constructs a positive definite 3-dimensional `TensorBundle`, so constructors such as `S"..."` and `D"..."` are convenient.
```@setup ds
using DirectSum
```
It is also possible to change the diagonal scaling, such as with `D"1,1,1,0"`, although the `Signature` format has a more compact representation if limited to ``+1`` and ``-1``.
It is also possible to change the diagonal scaling, such as with `D"0.3,2.4,1"`.
Fully general `MetricTensor` as a type with non-diagonal components requires a matrix, e.g. `MetricTensor([1 2; 2 3])`.

Declaring an additional point at infinity is done by specifying it in the string constructor with ``\infty`` at the first index (i.e. Riemann sphere `S"∞+++"`).
The hyperbolic geometry can be declared by ``\emptyset`` subsequently (i.e. hyperbolic projection `S"∅+++"`).
Additionally, the *null-basis* based on the projective split for conformal geometric algebra would be specified with `S"∞∅+++"`.
These two declared basis elements are interpreted in the type system.
The `tangent(V,μ,ν)`  map can be used to specify ``\mu`` and ``\nu``.

To assign `V = Submanifold(::TensorBundle)` along with associated basis
elements of the `DirectSum.Basis` to the local Julia session workspace, it is typical to use `Submanifold` elements created by the `@basis` macro,
```@repl ds
using Grassmann; @basis S"-++" # macro or basis"-++"
```
the macro `@basis V` delcares a local basis in Julia.
As a result of this macro, all `Submanifold{V,G}` elements generated with `M::TensorBundle` become available in the local workspace with the specified naming arguments.
The first argument provides signature specifications, the second argument is the variable name for ``V`` the ``\mathbb{K}``-module, and the third and fourth argument are prefixes of the `Submanifold` vector names (and covector names).
Default is ``V`` assigned `Submanifold{M}` and ``v`` is prefix for the `Submanifold{V}`.

It is entirely possible to assign multiple different bases having different signatures without any problems.
The `@basis` macro arguments are used to assign the vector space name to ``V`` and the basis elements to ``v_i``, but other assigned names can be chosen so that their local names don't interfere:
If it is undesirable to assign these variables to a local workspace, the versatile constructs of `DirectSum.Basis{V}` can be used to contain or access them, which is exported to the user as the method `DirectSum.Basis(V)`.
```@repl ds
DirectSum.Basis(V)
```
`V(::Int...)` provides a convenient way to define a `Submanifold` by using integer indices to reference specific direct sums within the ambient space ``V``.

Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of ``V`` and has its interpretation only instantiated by context of `TensorAlgebra{V}` elements being operated on.
Interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `TensorBundle` form of any other `TensorAlgebra` element is handled globally.
This enables the usage of `I` from `LinearAlgebra` as a universal pseudoscalar element defined at every point ``x`` of a `Manifold`, which is mathematically denoted by ``I = I(x)`` and specified by the ``g(x)`` bilinear tensor field of ``TM``.

*Grassmann.jl* is a foundation which has been built up from a minimal ``\mathbb{K}``-module algebra kernel on which an entirely custom algbera specification is designed and built from scratch on the base Julia language.

**Definition**.
`TensorAlgebra{V,```\mathbb{K}```}` where `V::Submanifold{M}` for a generating ``\mathbb{K}``-module specified by a `M::TensorBundle` choice
* `TensorBundle` specifies generators of `DirectSum.Basis` algebra
    * `Int` value induces a Euclidean metric of counted dimension
    * `Signature` uses `S"..."` with + and - specifying the metric
    * `DiagonalForm` uses `D"..."` for defining any diagonal metric
    * `MetricTensor` can accept non-diagonal metric tensor array
* `TensorGraded{V,G,```\mathbb{K}```}` has `grade` ``G`` and element of ``\Lambda^GV`` subspace
    * `Chain{V,G,```\mathbb{K}```}` has a complete basis for ``\Lambda^GV`` with ``\mathbb{K}``-module
    * `Simplex{V}` alias column-module `Chain{V,1,Chain{V,1,```\mathbb{K}```}}`
* `TensorTerm{V,G,```\mathbb{K}```} <: TensorGraded{V,G,```\mathbb{K}```}` single coefficient
    * `Zero{V}` is a zero value which preserves ``V`` in its algebra type
    * `Submanifold{V,G,B}` ``\langle v_{i_1}\wedge\cdots\wedge v_{i_G}\rangle_G`` with sorted indices ``B``
    * `Single{V,G,B,```\mathbb{K}```}` where `B::Submanifold{V}` is paired to ``\mathbb{K}``
* `AbstractSpinor{V,```\mathbb{K}```}` subtypes are special sub-algebras of ``\Lambda V``
    * `Couple{V,B,```\mathbb{K}```}` is the sum of ``\mathbb{K}`` scalar with `Single{V,G,B,```\mathbb{K}```}`
    * `PseudoCouple{V,B,```\mathbb{K}```}` is pseudoscalar + `Single{V,G,B,```\mathbb{K}```}`
    * `Spinor{V,```\mathbb{K}```}` has complete basis for the `even` ``\mathbb{Z}_2``-graded terms
    * `CoSpinor{V,```\mathbb{K}```}` has complete basis for `odd` ``\mathbb{Z}_2``-graded terms
* `Multivector{V,```\mathbb{K}```}` has complete basis for all ``\Lambda V`` with ``\mathbb{K}``-module


**Definition**. `TensorNested{V,T}` subtypes are linear transformations
* `TensorOperator{V,W,T}` linear map ``V\rightarrow W`` with `T::DataType`
    * `Endomorphism{V,T}` linear map ``V\rightarrow V`` with `T::DataType`
* `DiagonalOperator{V,T}` diagonal map ``V\rightarrow V`` with `T::DataType`
    * `DiagonalMorphism{V,<:Chain{V,1}}` diagonal map ``V\rightarrow V``
    * `DiagonalOutermorphism{V,<:Multivector{V}}` ``:\Lambda V\rightarrow \Lambda V``
* `Outermorphism{V,T}` extends ``F\in`` `Endomorphism{V}` to full ``\Lambda V``
```math
F(v_1)\wedge\cdots\wedge F(v_n) = F(v_1\wedge\cdots\wedge v_n)
```
* `Projector{V,T}` linear map ``F:V\rightarrow V`` with ``F(F) = F`` defined
```math
\verb`Proj(x::TensorGraded)` = \frac{x}{|x|}\otimes\frac{x}{|x|}
```
* `Dyadic{V,X,Y}` linear map ``V\rightarrow V`` with `Dyadic(x,y)` ``= x\otimes y``

*Grassmann.jl* was first to define a comprehensive `TensorAlgebra{V}` type system from scratch around the idea of the `V::Submanifold{M}` value to express algebra subtypes for a specified ``\mathbb{K}``-module structure.

**Definition**. Common unary operations on `TensorAlgebra` elements
* `Manifold` returns the parameter `V::Submanifold{M}` ``\mathbb{K}``-module
* `mdims` dimensionality of the pseudoscalar ``V`` of that `TensorAlgebra`
* `gdims` dimensionality of the grade ``G`` of ``V`` for that `TensorAlgebra`
* `tdims`  dimensionality of `Multivector{V}` for that `TensorAlgebra`
* `grade` returns ``G`` for `TensorGraded{V,G}` while `grade(x,g)` is ``\langle x\rangle_g``
* `istensor` returns true for `TensorAlgebra` elements
* `isgraded` returns true for `TensorGraded` elements
* `isterm` returns true for `TensorTerm` elements
* `complementright` Euclidean metric Grassmann right complement
* `complementleft` Euclidean metric Grassmann left complement
* `complementrighthodge` Grassmann-Hodge right complement ``\widetilde\omega I``
* `complementlefthodge` Grassmann-Hodge left complement ``I\widetilde\omega``
* `metric` applies the `metricextensor` as outermorphism operator
* `cometric` applies complement `metricextensor` as outermorphism
* `metrictensor` returns ``g:V\rightarrow V`` associated to `TensorAlgebra{V}`
* `metrictextensor` returns ``\Lambda g:\Lambda V\rightarrow\Lambda V`` for `TensorAlgebra{V}`
* `involute` grade permutes basis with ``\langle\overline\omega\rangle_k = \sigma_1(\langle\omega\rangle_k) = (-1)^k\langle\omega\rangle_k``
* `reverse` permutes basis with ``\langle\widetilde\omega\rangle_k = \sigma_2(\langle\omega\rangle_k) = (-1)^{k(k-1)/2}\langle\omega\rangle_k``
* `clifford` conjugate of an element is composite `involute` ``\circ`` `reverse`
* `even` part selects ``\overline{\mathfrak{R}}\omega = (\omega + \overline\omega)/2`` and is defined by ``\Lambda^g`` for even ``g``
* `odd` part selects ``\overline{\mathfrak{I}}\omega = (\omega-\overline\omega)/2`` and is defined by ``\Lambda^g`` for odd ``g``
* `real` part selects ``\widetilde{\mathfrak{R}}\omega = (\omega+\widetilde\omega)/2`` and is defined by ``|\widetilde{\mathfrak{R}}\omega|^2 = (\widetilde{\mathfrak{R}}\omega)^2``
* `imag` part selects ``\widetilde{\mathfrak{I}}\omega = (\omega-\widetilde\omega)/2`` and is defined by ``|\widetilde{\mathfrak{I}}\omega|^2 = -(\widetilde{\mathfrak{I}}\omega)^2``
* `abs` is the absolute value ``|\omega|=\sqrt{\widetilde\omega\omega}`` and `abs2` is then ``|\omega|^2 = \widetilde\omega\omega``
* `norm` evaluates a positive definite norm metric on the coefficients
* `unit` applies normalization defined as `unit(t) = t/abs(t)`
* `scalar` selects grade 0 term of any `TensorAlgebra` element
* `vector` selects grade 1 terms of any `TensorAlgebra` element
* `bivector` selects grade 2 terms of any `TensorAlgebra` element
* `trivector` selects grade 3 terms of any `TensorAlgebra` element
* `pseudoscalar` max. grade term of any `TensorAlgebra` element
* `value` returns internal `Values` tuple of a `TensorAlgebra` element
* `valuetype` returns type of a `TensorAlgebra` element value's tuple

Binary operations commonly used in `Grassmann` algebra syntax
* `+` and `-` carry over from the ``\mathbb{K}``-module structure associated to ``\mathbb{K}``
* `wedge` is exterior product ``\wedge`` and `vee` is regressive product ``\vee``
* `>` is the right contraction and `<` is the left contraction for the algebra
* `*` is the geometric product and `/` uses `inv` algorithm for division
* ``\oslash`` is the `sandwich` and `>>>` is its alternate operator orientation

Custom methods related to tensor operators and roots of polynomials
* `inv` returns the inverse and `adjugate` returns transposed cofactor
* `det` returns the scalar determinant of an endomorphism operator
* `tr` returns the scalar trace of an endomorphism operator
* `transpose` operator has swapping of row and column indices
* `compound(F,g)` is multilinear endomorphism ``\Lambda^gF : \Lambda^g V\rightarrow\Lambda^g V``
* `outermorphism(A)` transforms ``A:V\rightarrow V`` into ``\Lambda A:\Lambda V\rightarrow\Lambda V``
* `operator` make linear representation of multivector outermorphism
* `companion` matrix of monic polynomial ``a_0+a_1z+\dots+a_nz^n + z^{n+1}``
* `roots(a...)` of polynomial with coefficients ``a_0 + a_1z + \dots + a_nz^n``
* `rootsreal` of polynomial with coefficients ``a_0 + a_1z + \dots + a_nz^n``
* `rootscomplex` of polynomial with coefficients ``a_0 + a_1z + \dots + a_nz^n``
* `monicroots(a...)` of monic polynomial ``a_0+a_1z+\dots+a_nz^n + z^{n+1}``
* `monicrootsreal` of monic polynomial ``a_0+a_1z+\dots+a_nz^n + z^{n+1}``
* `monicrootscomplex` of monic polynomial ``a_0+a_1z+\dots+a_nz^n + z^{n+1}``
* `characteristic(A)` polynomial coefficients from ``\det (A-\lambda I)``
* `eigvals(A)` are the eigenvalues ``[\lambda_1,\dots,\lambda_n]`` so that ``A e_i = \lambda_i e_i ``
* `eigvalsreal` are real eigenvalues ``[\lambda_1,\dots,\lambda_n]`` so that ``A e_i = \lambda_i e_i ``
* `eigvalscomplex` are complex eigenvalues ``[\lambda_1,\dots,\lambda_n]`` so ``A e_i = \lambda_i e_i ``
* `eigvecs(A)` are the eigenvectors ``[e_1,\dots,e_n]`` so that ``A e_i = \lambda_i e_i ``
* `eigvecsreal` are real eigenvectors ``[e_1,\dots,e_n]`` so that ``A e_i = \lambda_i e_i ``
* `eigvecscomplex` are complex eigenvectors ``[e_1,\dots,e_n]`` so ``A e_i = \lambda_i e_i ``
* `eigen(A)` spectral decomposition ``\sum_i \lambda_i\text{Proj}(e_i)`` with ``A e_i = \lambda_i e_i``
* `eigenreal` spectral decomposition ``\sum_i \lambda_i\text{Proj}(e_i)`` with ``A e_i = \lambda_i e_i``
* `eigencomplex` spectral decomposition ``\sum_i \lambda_i\text{Proj}(e_i)`` so ``A e_i = \lambda_i e_i``
* `eigpolys(A)` normalized symmetrized functions of `eigvals(A)`
* `eigpolys(A,g)` normalized symmetrized function of `eigvals(A)`
* `vandermonde` facilitates ``((X'X)^{-1} X')y`` for polynomial coefficients
* `cayley(V,```\circ```)` returns product table for ``V`` and binary operation ``\circ``

Accessing `metrictensor(V)` produces a linear map ``g: V\rightarrow V`` which can be extended to ``\Lambda g:\Lambda V\rightarrow\Lambda V`` outermorphism given by `metricextensor`.
To apply the `metricextensor` to any `Grassmann` element of ``\Lambda V``, the function `metric` can be used on the element, `cometric` applies a complement metric.

## Tensor field topology and fiber bundles

**Definition**. Commonly used fundamental building blocks are 
* `ProductSpace{V,```\mathbb{K}```,N} <: AbstractArray{Chain{V,1,```\mathbb{K}```,N},N}`
    * uses Cartesian products of interval subsets of `` \mathbb{R}\times\mathbb{R}\times\cdots\times\mathbb{R} = \Lambda^1 \mathbb{R}^n ``,
    * generates lazy array of `Chain{V,1}` point vectors from input ranges
    * `ProductSpace{V}(0:0.1:1,0:0.1:1) # specify V`
    * `ProductSpace(0:0.1:1,0:0.1:1) # auto-select V`
    * `ProductSpace{V}(r::AbstractVector{<:Real}...)` default
    * ``\oplus```(r::AbstractVector{<:Real}...)` for algebraic syntax
* `Global{N,T}` represents array with same ``T`` value at all indices
* `LocalFiber{B,F}` has a local `basetype` of ``B`` and `fibertype` of ``F``
    * `Coordinate{P,G}` has `pointtype` of ``P`` and `metrictype` of ``G``
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

All these methods apply to `SimplexTopology` except `isopen`, `iscompact`
* `isopen` is true if `QuotientTopology` is an `OpenTopology` instance
* `iscompact` is true if `QuotientTopology` is a `CompactTopology`
* `nodes` counts number of `vertices` associated to `SimplexTopology`
* `sdims` counts the number of vertices $N$ of a `SimplexTopology{N}`
* `subelements` subspace element indices associated to `fulltopology`
* `subimmersion` modified with vertices re-indexed based on subspace
* `topology` view into `fulltopology` based on `subelements` structure
* `totalelements` counts total number of elements in `fulltopology`
* `totalnodes` counts total number of nodes over all subspaces
* `vertices` list of indices associated to the subspace `immersion`
* `elements` counts the number of `subelements` in the `immersion`
* `isfull` is true if the `immersion` is a `fulltopology`, not a subspace
* `istotal` is true if `fulltopology` is covering `totalnodes` completely
* `iscover` is true if `isfull` and `istotal`, fully covering `totalnodes`
* `isdiscontinuous` is true if an instance of `DiscontinuousTopology`
* `isdisconnected` is true if `isdiscontinuous` and fully disconnected
* `continuous` returns original data from `DiscontinuousTopology`
* `discontinuous` allocates a derived `DiscontinuousTopology`
* `disconnect` allocates a disconnected `DiscontinuousTopology`
* `getfacet` indexing `subelements` in reference to the `fullimmersion`
* `getimage` indexing vertex subspace in reference to `fullimmersion`

* `edges` assembles `SimplexTopology{2}` of all unique edge elements
* `facets` assembles `SimplexTopology` of all unique facet elements
* `complement` returns a `SimplexTopology` based on `subelements`
* `interior` returns the interior components of a `SimplexTopology`
* ``\partial`` operator returns boundary components of a `SimplexTopology`
* `degrees` returns the (graph) degree for each incidence vertex
* `weights` divides each incidence vertex by the (graph) degree
* `adjacency` returns a symmetric sparse matrix with ones at vertices
* `antiadjacency` returns sparse matrix with vertex antisymmetry
* `incidence` returns heterogenous relation for vertices and elements
* `laplacian` returns the (graph) Laplacian with adjacent vertices
* `neighbors` finds neighboring elements per `SimplexTopology` facet

**Definition**. An ``n``-dimensional *manifold* ``M`` requires the existence of a neighborhood ``U`` for each ``p\in U\subseteq M`` with a local *chart* map ``\phi : U_\phi\rightarrow\mathbb{R}^n``.
Given another chart ``\psi: U_\psi\rightarrow\mathbb{R}^n``, then the combinatorial compositions
```math
\phi\circ\psi^{-1} : \psi(U_\phi\cap U_\psi)\rightarrow\phi(U_\phi\cap U_\psi), \quad \psi\circ\phi^{-1} : \phi(U_\phi\cap U_\psi) \rightarrow \psi(U_\phi\cap U_\psi)
```
are the transition maps.
If all the transition maps ``\phi`` are ``\mathcal{C}^r`` differentiable and ``\bigcup_\phi U_\phi = M``, then the collection of charts is called an *atlas* of a ``\mathcal{C}^r``-manifold.

**Definition**. A *fiber bundle* is a manifold ``E`` with projection ``\pi: E \rightarrow B`` commutes with local trivializations ``\phi`` paired to ``U_\phi`` of manifold ``B = \bigcup_\phi U_\phi``
```math
\phi: \pi^{-1}(U_\phi) \rightarrow U_\phi\times F, \qquad \pi^{-1}(U_\phi) \overset{\pi}{\longrightarrow} U_\phi \overset{\pi_1}{\longleftarrow} U_\phi\times F,
```
where ``B`` is the `basetype` and ``F`` is the `fibertype` of
`` E_p = \pi^{-1}(p)  = \{p\}\times F``,
```math
E = \bigcup_{p\in B}E_p = \bigcup_{p\in B} \{p\}\times F = B\times F, \qquad B = \bigcup_\phi U_\phi.
```

`FiberBundle{E,N} <: AbstractArray{E,N}` where ``E`` is the `eltype`
* `Coordinates{P,G,N} <: FiberBundle{Coordinate{P,G},N}`
    * `PointArray{P,G,N}` has `pointtype` of ``P``, `metrictype` of ``G``
    * `FiberProduct` introduces fiber product structure for manifolds
* `FrameBundle{C,N}` has `coordinatetype` of ``C`` and `immersion`
    * `GridBundle{N,C}` ``N``-grid with `coordianates` and `immersion`
    * `SimplexBundle{N,C}` defines `coordinates` and an `immersion`
    * `FaceBundle{N,C}` defines `element` faces and their `immersion`
    * `FiberProductBundle` for extruding dimensions from simplices
    * `HomotopyBundle` encapsulates a variation as `FrameBundle`
* `TensorField` defines fibers in a global section of a `FrameBundle`

When a `TensorField` has a `fibertype` from ``\Lambda^gTV`` then it is a grade ``g`` differential form on the tangent bundle of ``V``.
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
* `base` returns the ``B`` element of a `LocalFiber{B,F}` or `FiberBundle`
* `basetype` returns type ``B`` of a `LocalFiber{B,F}` or `FiberBundle`
* `fiber` returns the ``F`` element of `LocalFiber{B,F}` or `FiberBundle`
* `fibertype` returns the ``F`` type of `LocalFiber{B,F}` or `FiberBundle`
* `points` returns `AbstractArray{P}` data for `Coordinates{P,G}`
* `pointtype` is type ``P`` of `Coordinate{P,G}` or `Coordinates{P,G}`
* `metrictensor` returns the grade 1 block of the `metricextensor`
* `metricextensor` is `AbstractArray{G}` data for `Coordinates{P,G}`
* `metrictype` is type $G$ of `Coordinate{P,G}` or `Coordinates{P,G}`
* `fullcoordinates` returns full `FiberBundle{Coordinate{P,G}}`
* `fullimmersion` returns superset `ImmersedTopology` which `isfull`
* `fulltopology` returns composition of `topology` ``\circ`` `fullimmersion`
* `fullvertices` list of `vertices` associated to the `fullimmersion`
* `fullpoints` is full `AbstractArray{P}` instead of possibly subspace
* `fullmetricextensor` is full `AbstractArray{G}` instead of subspace
* `isinduced` is true if the `metrictype` is an `InducedMetric` type
* `bundle` returns the integer identification of bundle cache

Various interpolation methods are also supported and can be invoked by applying `TensorField` instances as function evaluations on base manifold or applying some form of resampling method to the manifold topology.
* `volumes` returns `FaceBundle` with simplex volume at each element
* `initmesh` provides API keyword for interfacing mesh initialization
* `refinemesh` provides API keyword for interfacing mesh refinement
* `affinehull` returns a localized affine simplex hull at each element
* `affineframe` returns a localized affine basis frame at each element
* `gradienthat` returns the hat gradients for the `SimplexBundle`

For `GridBundle` initialization it is typical to invoke a combination of `ProductSpace` and `QuotientTopology`, while optional Julia packages extend `SimplexBundle` initialization, such as
`Meshes`, 
`GeometryBasics`, 
`Delaunay`, 
`QHull`, 
`MiniQhull`, 
`Triangulate`, 
`TetGen`,
`MATLAB`,
`FlowGeometry`.

**Definition**. Let ``\gamma:[a,b] \rightarrow \mathbb R^n`` be a curve ``\gamma(t)`` with parameter ``t``.
* `integral(::IntervalMap)` cumulative trapezoidal sum ``\int_a^t\gamma(\xi)d\xi``
* `integrate(::IntervalMap)` final value of ``\int_a^b\gamma(t) dt`` on interval end
* `arclength(::IntervalMap)` curve parameter `` s(t) = \int_a^t |\frac{d\gamma(\xi)}{d\xi}|d\xi ``
* `tangent(::IntervalMap)` tangent speed curve `` \frac{d\gamma(t)}{dt} = \frac{ds}{dt} T(t) ``
* `unittangent(::IntervalMap)` unit tangent curve `` T(t) = \frac{d\gamma}{dt}\frac{dt}{ds} ``
* `speed(::IntervalMap)` tangent speed of a curve ``\frac{ds}{dt} = \left|\frac{d\gamma(t)}{dt}\right| ``
* `normal(::IntervalMap)` `` \kappa(t)N(t) = \frac{dT}{dt}\frac{dt}{ds} = \frac{d}{dt}\left(\frac{d\gamma}{dt}\frac{dt}{ds}\right)\frac{dt}{ds} ``
* `unitnormal(::IntervalMap)` `` N(t) = \frac{dT}{dt}\frac{dt}{ds}/\kappa(t) `` normalized
* `curvature(::AbstractCurve)` `` \kappa(t) = \left|\frac{dT}{dt}\frac{dt}{ds}\right| = \left|\frac{d}{dt}\left(\frac{d\gamma}{dt}\frac{dt}{ds}\right)\frac{dt}{ds}\right| ``
* `radius(::AbstractCurve)` of curvature `` \kappa(t)^{-1} = \left|\frac{d}{dt}\left(\frac{d\gamma}{dt}\frac{dt}{ds}\right)\frac{dt}{ds}\right|^{-1} ``
* `evolute(::AbstractCurve)` `` \gamma + \frac{N}{\kappa} = \gamma(t) + \frac{d}{dt}\left(\frac{d\gamma}{dt}\frac{dt}{ds}\right)\frac{dt}{ds}/(\kappa(t))^2  ``
* `involute(::AbstractCurve)` `` \gamma - Ts = \gamma(t) - \left(\frac{d\gamma(t)}{dt}\frac{dt}{ds}\right)\int_a^t|\frac{d\gamma(\xi)}{d\xi}|d\xi ``
* `osculatingplane(::AbstractCurve)` linear span of ``\left[\frac{ds}{dt}T,\kappa N\right]``
* `unitosculatingplane(::AbstractCurve)` linear span of ``\left[T,N\right]``
* `binormal(::SpaceCurve)` complement ``\frac{ds}{dt}\kappa B = \star (\frac{ds}{dt}T\wedge \kappa N)``
* `unitbinormal(::SpaceCurve)` complement of plane ``B = \star (T\wedge N)``
* `torsion(::SpaceCurve)` `` \tau(t) = \left|\frac{dB}{dt}\frac{dt}{ds}\right| = \left|\frac{d}{dt}\star\left(T\wedge N\right)\frac{dt}{ds}\right| ``

* `frame(::AbstractCurve)` scaled frame ``\left[\frac{ds}{dt}T,\kappa N,\star(\frac{ds}{dt}T\wedge\kappa N)\right]``
* `unitframe(::AbstractCurve)` Frenet frame ``\left[T,N,\star(T\wedge N)\right]``
* `curvatures(::AbstractCurve)` all degrees of freedom `` \left[\kappa,\tau,\dots\right] ``
* `curvatures(::AbstractCurve,i)` selects ``i``-th curvature degree
* `bishopframe(::SpaceCurve,```\theta_0```=0)` computes Bishop-style frame
* `bishopunitframe(::SpaceCurve,```\theta_0```=0)` Bishop-style unit frame
* `bishoppolar(::SpaceCurve,```\theta_0```=0)` Bishop polar ``(\kappa,\theta_0+\int_a^b\tau ds)``
* `bishop(::SpaceCurve,```\theta_0```=0)` `` \kappa(\cos(\theta_0+\int_a^b\tau ds),\sin(\theta_0+\int_a^b\tau ds)) ``
* `planecurve(::RealFunction,```\theta_0```=0)` from curvature ``\kappa(t)`` and ``\theta_0``
```math
\qquad \textstyle (\kappa(t),\theta_0) \mapsto \int_a^b \left[cos (\theta_0 + \int_a^b\kappa(t)dt),\sin(\theta_0+\int_a^b\kappa(t)dt)\right]dt
```
* `linkmap(f::SpaceCurve,g::SpaceCurve)` is ``\ell(t,s) = g(s)-f(t)``
* `linknumber(f,g)` of curves ``\propto`` `sectorintegrate` ``\circ`` `unit` ``\circ`` `linkmap`

**Definition**. Surfaces ``\gamma : \mathbb{R}^2\rightarrow\mathbb{R}^3 `` with parametric ``\gamma(x_1,x_2)`` methods
* `graph` outputs surface ``\gamma :\mathbb{R}^n\rightarrow\mathbb{R}^n\times\mathbb{R}`` from scalar field ``f:\mathbb{R}^n\rightarrow\mathbb{R}``
* `normal` vector ``N(x) = \star(\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n})`` or product ``\frac{\partial \gamma(x)}{\partial x_1}\times\frac{\partial \gamma(x)}{\partial x_2}``
* `unitnormal` ``\nu(x) = \star(\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n}) / |\star(\frac{\partial \gamma(x)}{x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n})|``
* `normalnorm` is the norm of normal ``|N(x)| = |\star(\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n})|``
* `jacobian` linear span of ``\left[\frac{\partial \gamma(x)}{\partial x_1},\dots,\frac{\partial \gamma(x)}{\partial x_n}\right]`` as `TensorOperator`
* `weingarten` linear span of ``\left[\frac{\partial \nu(x)}{\partial x_1},\dots,\frac{\partial \nu(x)}{\partial x_n}\right]`` as `TensorOperator`
* `sectordet` is the product ``\gamma(x)\wedge\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n}``, here with ``n = 2``
* `sectorintegral` ``\int \frac{\gamma(x)}{n+1}\wedge\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n}dx_1\cdots dx_n`` with ``n=2``
* `sectorintegrate` estimates the total value of `sectorintegral`
* `area` cumulative ``\int |\star(\frac{\partial \gamma(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \gamma(x)}{\partial x_n})|dx_1\cdots dx_n`` with ``n=2``
* `surfacearea` estimates total value of the (surface) `area` integral
* `surfacemetric` gets `GridBundle` with `firstform` as `metrictensor`
* `surfacemetricdiag` gets `GridBundle` with `firstformdiag` metric
* `surfaceframe` constructs intrinsic orthonormal surface frame
* `frame` scaled Darboux style frame ``\left[\frac{\partial\gamma(x)}{\partial x_1},\star\left(N(x)\wedge\frac{\partial\gamma(x)}{\partial x_1}\right),N(x)\right] ``
* `unitframe` is then ``\left[\frac{\partial\gamma(x)}{\partial x_1}/\left|\frac{\partial\gamma(x)}{\partial x_1}\right|,\star\left(\nu(x)\wedge\frac{\partial\gamma(x)}{\partial x_1}\right)/\left|\frac{\partial\gamma(x)}{\partial x_1}\right|,\nu(x)\right] ``
* `firstform` ``I = g = \begin{bmatrix} \frac{\partial\gamma(x)}{\partial x_1}\cdot\frac{\partial\gamma(x)}{\partial x_1} & \frac{\partial\gamma(x)}{\partial x_1}\cdot\frac{\partial\gamma(x)}{\partial x_2} \\ \frac{\partial\gamma(x)}{\partial x_1}\cdot\frac{\partial\gamma(x)}{\partial x_2} & \frac{\partial\gamma(x)}{\partial x_2}\cdot\frac{\partial\gamma(x)}{\partial x_2} \end{bmatrix}`` or `firstformdiag`
* `secondform` ``II = \begin{bmatrix} \nu(x)\cdot\frac{\partial^2\gamma(x)}{\partial x_1^2} & \nu(x)\cdot\frac{\partial^2\gamma(x)}{\partial x_1\partial x_2} \\ \nu(x)\cdot\frac{\partial^2\gamma(x)}{\partial x_1\partial x_2} & \nu(x)\cdot\frac{\partial^2\gamma(x)}{\partial x_2^2} \end{bmatrix}`` 2nd fundamental
* `thirdform` ``III`` is the composition map `firstform` ``\circ`` `unitnormal`
* `shape` is the geometry shape operator ``I(x)^{-1} II(x)`` of a surface ``\gamma(x)``
* `principals` (curvatures) are the composition `eigvals` ``\circ`` `shape`
* `principalaxes` (curvatures) are the composition `eigvecs` ``\circ`` `shape`
* `curvatures` (polynomials) are the composition `eigpolys` ``\circ`` `shape`
* `curvatures(::TensorField,i)` selects ``i``-th curvature polynomial
* `meancurvature` is the mean curvature (first curvature) of the `shape`
* `gaussintrinsic` is the (intrinsic) Gauss curvature (last curvature)
* `gaussextrinsic` is the (extrinsic) Gauss curvature in sector form
* `gausssign` is the sign of the Gauss curvature of the `shape`

## Interactive computational sessions

*Example* (Plane curves). Let ``t`` be parameter on interval from 0 to ``4\pi``
```julia
using Grassmann, Cartan, Makie # GLMakie
t = TensorField(0:0.01:4*pi)
lin = Chain.(cos(t)*t,sin(t)*11+t)
lines(lin); scaledarrows!(lin,unitframe(lin),gridsize=50)
lines(arclength(lin))
lines(speed(lin))
lines(curvature(lin))
```
Get `curvature` from plane curve or construct `planecurve` from curvature:
```julia
lines(planecurve(cos(t)*t))
lines(planecurve(cos(t*t)*t))
lines(planecurve(cos(t)-t*sin(t)))
```


*Example* (Lorenz). Observe vector fields by integrating streamlines
```julia
Lorenz(s,r,b) = x -> Chain(
    s*(x[2]-x[1]), x[1]*(r-x[3])-x[2], x[1]*x[2]-b*x[3])
p = TensorField(ProductSpace(-40:0.2:40,-40:0.2:40,10:0.2:90))
vf = Lorenz(10.0,60.0,8/3).(p) # pick Lorenz parameters, apply
streamplot(vf,gridsize=(10,10)) # visualize vector field
```
ODE solvers in the `Adapode` package are built on `Cartan`, providing both Runge-Kutta and multistep methods with optional adaptive time stepping.
```julia
using Grassmann, Cartan, Adapode, Makie # GLMakie
fun,x0 = Lorenz(10.0,60.0,8/3),Chain(10.0,10.0,10.0)
ic = InitialCondition(fun,x0,2pi) # tmax = 2pi
lines(odesolve(ic,MultistepIntegrator{4}(2^-15)))
```


*Example* (Riemann sphere).
Project ``\uparrow : \omega \mapsto (2\omega+v_\infty(\omega^2-1))/(\omega^2+1)`` and then apply rotation before rejecting down ``\downarrow :\omega\mapsto((\omega\wedge b)v_\infty)/(1-v_\infty\cdot\omega)``.

```julia
using Grassmann, Cartan, Makie # GLMakie
pts = TensorField(-2*pi:0.0001:2*pi)
@basis S"∞+++" # Riemann sphere
f(t) = ↓(exp(π*t*((3/7)*v12+v∞3))>>>↑(v1+v2+v3))
f(t) = ↓(exp(t*v∞*(sin(3t)*3v1+cos(2t)*7v2-sin(5t)*4v3)/2)>>>↑(v1+v2-v3))
f(t) = ↓(exp(t*(v12+0.07v∞*(sin(3t)*3v1+cos(2t)*7v2-sin(5t)*4v3)/2))>>>↑(v1+v2-v3))
lines(V(2,3,4).(f.(pts))) # applies to each f(t)
```
```julia
@basis S"∞∅+++" # conformal geometric algebra
f(t) = ↓(exp(π*t*((3/7)*v12+v∞3))>>>↑(v1+v2+v3))
lines(V(3,4,5).(vector.(f.(pts))))
```

*Example* (Bivector). `using Grassmann, Cartan, Makie # GLMakie`
```julia
basis"2" # Euclidean geometric algebra in 2 dimensions
vdom = TensorField(ProductSpace{V}(-1.5:0.1:1.5,-1.5:0.1:1.5))
streamplot(tensorfield(exp(pi*v12/2)).(vdom))
streamplot(tensorfield(exp((pi/2)*v12/2)).(vdom))
streamplot(tensorfield(exp((pi/4)*v12/2)).(vdom))
streamplot(tensorfield(v1*exp((pi/4)*v12/2)).(vdom))
```
```julia
@basis S"+-" # Geometric algebra with Lobachevskian plane
vdom = TensorField(ProductSpace{V}(-1.5:0.1:1.5,-1.5:0.1:1.5))
streamplot(tensorfield(exp((pi/8)*v12/2)).(vdom))
streamplot(tensorfield(v1*exp((pi/4)*v12/2)).(vdom))
```

*Example*. `using Grassmann, Cartan, Makie; @basis S"∞+++"`
```julia
vdom1 = TensorField(ProductSpace{V(1,2,3)}(
    -1.5:0.1:1.5,-1.5:0.1:1.5,-1.5:0.1:1.5));
tf1 = tensorfield(exp((pi/4)*(v12+v∞3)),V(2,3,4)).(vdom1)
streamplot(tf1,gridsize=(10,10))
```
```julia
vdom2 = TensorField(ProductSpace{V(2,3,4)}(
    -1.5:0.1:1.5,-1.5:0.1:1.5,-1.5:0.1:1.5));
tf2 = tensorfield(exp((pi/4)*(v12+v∞3)),V(2,3,4)).(vdom2)
streamplot(tf2,gridsize=(10,10))
```

**Definition**. Let `` [\cdot,\dots,\cdot] : \mathfrak g^n \rightarrow \mathfrak g `` define the ``n``-linear Lie bracket with
```math
[X_1,\dots,X_i,\dots,X_n] = \sum_{\sigma\in S_n} \varepsilon(\sigma) X_{\sigma(1)}(\dots(X_{\sigma(i)}(\dots(X_{\sigma(n)})\dots))\dots).
```
In `Grassmann` and `Cartan` this definition can be accessed with `Lie[Xi...]`.

In 2024, the author proved a new [multilinear Lie bracket recursion formula](https://vixra.org/abs/2412.0034).

**Theorem** (Lie bracket recursion). ``n``-bracket is sum of ``(n-1)``-brackets:
```math
[X_1,\dots,X_n] = \sum_{i=1}^n (-1)^{i-1}X_i([X_1,\dots,X_{i-1},X_{i+1},\dots,X_n])
```
This recursion can be explicitly expanded from the unary rule ``[X] = X``,
```math
[X,Y] = X([Y]) - Y([X]),
```
```math
[X,Y,Z] = X([Y,Z]) - Y([X,Z]) + Z([X,Y]),
```
```math
[W,X,Y,Z] = W([X,Y,Z]) - X([W,Y,Z]) + Y([W,X,Z]) - Z([W,X,Y]),
```
```math
{\scriptstyle [V,W,X,Y,Z]\, =\, V([W,X,Y,Z]) - W([V,X,Y,Z]) + X([V,W,Y,Z]) - Y([V,W,X,Z]) + Z([V,W,X,Y])}.
```
The multilinear Lie bracket recursion properly generalizes the bilinear Lie bracket to the ``n``-linear case and is analogous to the Koszul complex of the Grassmann algebra; but is fundamentally different due to multilinear Lie bracket being non-associative, unlike the analogous exterior product.

*Example* (Bracket). `using Grassmann, Cartan, Makie # GLMakie`
```julia
f1(x) = Chain(cos(3x[1]),sin(2x[1]))
f2(x) = sin(x[1]/2)*sin(x[2])
f3(x) = Chain(cos(x[1])*cos(x[2]),sin(x[2])*sin(x[1]))
vf1 = f1.(TorusParameter(100,100))
vf2 = gradient(f2.(TorusParameter(100,100)))
vf3 = f3.(TorusParameter(100,100))
lie1 = Lie[vf1,vf2] # Lie[vf1,vf2] == -Lie[vf2,vf1]
lie2 = Lie[vf1,vf2,vf3] # ternary Lie bracket
streamplot(lie1); streamplot(lie2)
```

*Example* (``\int_\Omega 1``). Length of line, area of disk, and volume of ball
```julia
linspace = ProductSpace(-2:0.03:2) # using Grassmann, Cartan
diameter = TensorField(linspace, x->abs(x)<1) # radius = 1
(integrate(diameter),2.0) # grid doesn't exactly align on 1.0
```
```math
(1.98v_1, 2.0v_1)
```
```julia
square = ProductSpace(-2:0.003:2,-2:0.003:2)
disk = TensorField(square, x->abs(x)<1) # radius = 1
(integrate(disk),1pi)
```
```math
(3.141414000000001v_{12}, 3.141592653589793v_{12})
```
```julia
cube = ProductSpace(-2:0.07:2,-2:0.07:2,-2:0.07:2)
ball = TensorField(cube, x->abs(x)<1) # radius = 1
(integrate(ball),4pi/3)
```
```math
(4.180680595387064v_{123}, 4.1887902047863905v_{123})
```
However, this is inefficient numerical integration, for example the ``58\times58\times58`` cube has ``195,112`` terms and the ``1334\times1334`` square has ``1,779,556`` terms.

**Theorem** (Hyper-area of hypersurface).
Let ``\gamma:X\subset\mathbb R^n\rightarrow\mathbb R^{n+1}`` be parameterized hypersurface ``\partial(\Omega) = \gamma(X)``.
Since the pullback ``\gamma^*(1)`` is ``\det d\gamma``,
```math
	\int_{\partial(\Omega)} 1 = \int_{\gamma(X)} 1 = \int_X|\det d\gamma| = \int_X \left|\frac{\partial\gamma}{\partial x_1} \wedge \dots \wedge \frac{\partial\gamma}{\partial x_n}\right|
```
```math
			 = \int_{a_1}^{b_1}\dots\int_{a_n}^{b_n} |\det d\gamma| = \int_{a_1}^{b_1}\dots\int_{a_n}^{b_n} \left|\frac{\partial \gamma}{\partial x_1} \wedge \dots \wedge \frac{\partial\gamma}{\partial x_n} \right|.
```

*Example*. Disk circumference, sphere spat `using Grassmann,Cartan`
```julia
t = TensorField(0:0.001:2pi)
circ = Chain.(cos(t),sin(t))
spher(x) = Chain(
    cos(x[2])*sin(x[1]), sin(x[2])*sin(x[1]),
    cos(x[1])) # GeographicParameter is swapped convention
sph = spher.(SphereParameter(60,60))
[surfacearea(circ), 2pi] # or totalarclength for AbstractCurve
[surfacearea(sph), 4pi]
lines(circ); wireframe(sph) # using Makie # GLMakie
```
```math
	\begin{matrix}
		\begin{bmatrix}
			6.283000000652752 \\ 6.283185307179586
		\end{bmatrix} \\ \quad \\
		\begin{bmatrix}
			12.533742943601457 \\ 12.566370614359172
	\end{bmatrix}
	\end{matrix}
```

**Theorem** (Sector integral).
Let ``X\subset\mathbb R^n`` and ``\gamma : X \rightarrow \partial(\Omega)\subset\mathbb R^{n+1} `` is parameterized hypersurface ``\partial(\Omega)=\gamma(X)`` with ``\gamma(x) = \gamma(x_1,\dots,x_n) ``, then
```math
\int_\Omega 1 = \frac{\rho^n}{n+1}\int_{X}\gamma\wedge\frac{\partial\gamma}{\partial x_1}\wedge\cdots\wedge\frac{\partial\gamma}{\partial x_n}
```
so there exists ``\Omega`` defining the sector of hypersurface ``\gamma(X)`` with scale ``\rho = 1``.

*Proof*. Theorem proved by Michael Reed in Grassmann.jl and Cartan.jl research papers.

*Example* (``\int_\Omega1``). Recall `circ,sph` to evaluate 
```julia
(sectorintegrate(circ),sectorintegrate(sph)) # more efficient
```
```math
(3.1415v_{12}, 4.17791v_{123})
```

*Example* (Link number). Define the `linkmap` of two `SpaceCurve` instances ``f(t)`` and ``g(s)`` as parameterized hypersurface ``\ell(s,t) = g(s)-f(t)``.
As a corollary of the *sector integral theorem* combined with unit linkmap:
```math
\frac{1}{4\pi}\int_X \gamma \wedge \frac{\partial\gamma}{\partial t}\wedge\frac{\partial\gamma}{\partial s} = \frac{1}{4\pi}\int_X \frac{g(s)-f(t)}{|g(s)-f(t)|^3}\wedge \frac{df(t)}{dt}\wedge\frac{dg(s)}{ds},
```
therefore Gauss `linknumber` is ``\frac{3}{4\pi}`` times `sectorintegrate` ``\circ`` `unit` ``\circ`` `linkmap` when evaluated with parameterized hypersurface ``\gamma(s,t) = \ell(s,t)/|\ell(s,t)|``.
So the `linknumber` divides `sectorintegral` of ``\gamma(X)`` by the volume of ball ``\Omega``.
```julia
t = TensorField(0:0.01:2pi) # using Grassmann, Cartan, Makie
f(t) = Chain(cos(t[1]),sin(t[1]),0)
g(t) = Chain(0,1+cos(t[1]),sin(t[1]))
lines(f.(t)); lines!(g.(t)); (linknumber(f.(t),g.(t)), 1.0)
mesh(linkmap(f.(t),g.(t)),normalnorm)
mesh(unit(linkmap(f.(t),g.(t))),normalnorm)
```

**Theorem** (Gauss curvature). New alternative formulas for (extrinsic) Gauss curvature ``K_e`` and for (intrinsic) Gauss curvature ``K_i`` with normal ``N``,
```math
K_e(x) = \nu(x)\wedge\frac{\partial\nu(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \nu(x)}{\partial x_n}, \quad 
	K_i(x) = \frac{K_e(x)}{|N(x)|},
```
```math
	|K_e(x)| = \left|\star\left(\frac{\partial \nu(x)}{\partial x_1}\wedge\cdots\wedge\frac{\partial \nu(x)}{\partial x_n}\right)\right|, \quad
	|K_i(x)| = \frac{|K_e(x)|}{|N(x)|}.
```
With this formula, the Gauss-Bonnet integral is a `sectorintegral` theorem.

*Example* (Torus).
```julia
using Grassmann, Cartan, Makie # GLMakie
torus(x) = Chain(
    (2+0.5cos(x[1]))*cos(x[2]),
    (2+0.5cos(x[1]))*sin(x[2]),
    0.5sin(x[1]))
tor = torus.(TorusParameter(60,60))
mesh(tor,normalnorm)
mesh(tor,meancurvature)
mesh(tor,gausssign)
```

*Example* (Wiggle).
```julia
using Grassmann, Cartan, Makie # GLMakie
wobble(x) = (1+0.3sin(3x[1])+0.1cos(7x[2]))
wumble(x) = (3+0.5cos(x[2]))
wiggle(x) = Chain(
    (wumble(x)+wobble(x)*cos(x[1]))*cos(x[2]),
    (wumble(x)+wobble(x)*cos(x[1]))*sin(x[2]),
    wobble(x)*sin(x[1]))
wig = wiggle.(TorusParameter(60,60))
mesh(wig,normalnorm)
mesh(wig,gaussextrinsic)
mesh(wig,gaussintrinsic)
```


**Definition**. When there is a Levi-Civita connection with zero torsion related to a `metrictensor`, then ``\nabla_X Y - \nabla_Y X = [X,Y]`` and there exist Christoffel symbols of the `secondkind` ``\Gamma_{ij}^k = \Gamma_{ji}^k`` with
`` \nabla_{\partial_i}\partial_j = \sum_k \Gamma_{ij}^k \partial_k``.
In particular, these can be expressed in terms of the `metrictensor` ``g`` as
```math
\Gamma^k_{ij} = \frac{1}{2} \sum_m g^{km}\set{\frac{\partial g_{mj}}{\partial x_i} + \frac{\partial g_{im}}{\partial x_j} - \frac{\partial g_{ij}}{\partial x_m} }.
```
Local geodesic differential equations for Riemannian geometry are then
```math
\frac{d^2 x_k}{dt^2} + \sum_{ij} \Gamma_{ij}^k \frac{dx_i}{dt}\frac{dx_j}{dt} = 0.
```

*Example*. `using Grassmann, Cartan, Adapode, Makie # GLMakie`
```julia
torus(x) = Chain(
    (2+0.5cos(x[1]))*cos(x[2]),
    (2+0.5cos(x[1]))*sin(x[2]),
    0.5sin(x[1]))
tor = torus.(TorusParameter(60,60))
tormet = surfacemetric(tor) # intrinsic metric
torcoef = secondkind(tormet) # Christoffel symbols
ic = geodesic(torcoef,Chain(1.0,1.0),Chain(1.0,sqrt(2)),10pi)
sol = geosolve(ic,ExplicitIntegrator{4}(2^-7)) # Runge-Kutta
lines(torus.(sol))
```
```julia
totalarclength(sol) # apparent length of parameter path
@basis MetricTensor([1 1; 1 1]) # abstract non-Euclidean V
solm = TensorField(tormet(sol),Chain{V}.(value.(fiber(sol))))
totalarclength(solm) # 2D estimate totalarclength(torus.(sol))
totalarclength(torus.(sol)) # compute in 3D Euclidean metric
lines(solm) # parametric curve can have non-Euclidean metric
lines(arclength(solm)); lines!(arclength(sol))
```

*Example* (Klein geodesic). General `ImmersedTopology` are supported
```julia
klein(x) = klein(x[1],x[2]/2)
function klein(v,u)
    x = cos(u)*(-2/15)*(3cos(v)-30sin(u)+90sin(u)*cos(u)^4-
        60sin(u)*cos(u)^6+5cos(u)*cos(v)*sin(u))
    y = sin(u)*(-1/15)*(3cos(v)-3cos(v)*cos(u)^2-
        48cos(v)*cos(u)^4+48cos(v)*cos(u)^6-
        60sin(u)+5cos(u)*cos(v)*sin(u)-
        5cos(v)*sin(u)*cos(u)^3-80cos(v)*sin(u)*cos(u)^5+
        80cos(v)*sin(u)*cos(u)^7)
    z = sin(v)*(2/15)*(3+5cos(u)*sin(u))
    Chain(x,y,z)
end # too long paths over QuotientTopology can stack overflow
kle = klein.(KleinParameter(100,100))
klecoef = secondkind(surfacemetric(kle))
ic = geodesic(klecoef,Chain(1.0,1.0),Chain(1.0,2.0),2pi)
lines(geosolve(ic,ExplicitIntegrator{4}(2^-7)));wireframe(kle)
```

*Example* (Upper half plane). Intrinsic hyperbolic Lobachevsky metric
```julia
halfplane(x) = TensorOperator(Chain(
    Chain(Chain(0.0,inv(x[2])),Chain(-inv(x[2]),0.0)),
    Chain(Chain(-inv(x[2]),0.0),Chain(0.0,-inv(x[2])))))
z1 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,2.0),10pi,7)
z2 = geosolve(halfplane,Chain(1.0,0.1),Chain(1.0,2.0),10pi,7)
z3 = geosolve(halfplane,Chain(1.0,0.5),Chain(1.0,2.0),10pi,7)
z4 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,1.0),10pi,7)
z5 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,1.5),10pi,7)
lines(z1); lines!(z2); lines!(z3); lines!(z4); lines!(z5)
```

*Example*. Calculus over Hopf fibration is enabled by `HopfTopology`,
```julia
stereohopf(x) = stereohopf(x[1],x[2],x[3])
function stereohopf(theta,phi,psi)
    a = cos(theta)*exp((im/2)*(psi-phi))
    b = sin(theta)*exp((im/2)*(psi+phi))
    Chain(imag(a),real(b),imag(b))/(1-real(a))
end
hs = stereohopf.(HopfParameter());
alteration!(hs,wireframe,wireframe!)
```


*Example*. Streamplots on tangent spaces enabled by `Cartan` methods,
```julia
spher(x) = Chain(
    cos(x[2])*sin(x[1]), sin(x[2])*sin(x[1]),
    cos(x[1])) # GeographicParameter is swapped convention
sph = spher.(SphereParameter(60,60))
f2(x) = sin(x[1]/2)*sin(x[2])
vf2 = gradient(f2.(TorusParameter(100,100)))
streamplot(sph,vf2)
```
```julia
torus(x) = Chain(
    (2+0.5cos(x[1]))*cos(x[2]),
    (2+0.5cos(x[1]))*sin(x[2]),
    0.5sin(x[1]))
tor = torus.(TorusParameter(60,60))
f3(x) = Chain(cos(x[1])*cos(x[2]),sin(x[2])*sin(x[1]))
vf3 = f3.(TorusParameter(100,100))
streamplot(tor,vf3)
```


*Example* (da Rios). The `Cartan` abstractions enable easily integrating
```math
\frac{\partial\gamma(x)}{\partial x_2} = \star\left(\frac{\partial\gamma(x)}{\partial x_1} \wedge \frac{\partial^2\gamma(x)}{\partial x_1^2} \right)
```
```julia
using Grassmann, Cartan, Adapode, Makie # GLMakie
start(x) = Chain(cos(x),sin(x),cos(1.5x)*sin(1.5x)/5)
x1 = start.(TorusParameter(180));
darios(t,dt=tangent(fiber(t))) = hodge(wedge(dt,tangent(dt)))
sol = odesolve(darios,x1,1.0,2^-11)
mesh(sol,normalnorm)
```

*Example* (Bishop frame). As an alternative to the standard Frenet style `unitframe`, the `bishopunitframe(::SpaceCurve,angle::Real)` has an optional angle (integration constant) modulo rotation of tangent axis.
```julia
using Grassmann, Cartan, Makie # GLMakie
start(x) = Chain(cos(x),sin(x),cos(1.5x)*sin(1.5x)/5)
x1 = start.(TorusParameter(180));
scaledarrows(x1,bishopunitframe(x1),gridsize=25)
lines!(x1,linestyle=:dash) # angle is optional
```


*Example* (Eigenmodes of disk).
	``-\Delta u = \lambda u`` with boundary ``n\cdot\nabla u = 0`` is enabled with `assemble` for stiffness and mass matrix from `Adapode`:
```julia
using Grassmann, Cartan, Adapode, MATLAB, Makie # GLMakie
pt,pe = initmesh("circleg","hmax"=>0.1) # MATLAB circleg mesh
A,M = assemble(pt,1,1,0) # stiffness & mass matrix assembly
using KrylovKit # provides general eigsolve
yi,xi = geneigsolve((A,M),10,:SR;krylovdim=100) # maxiter=100
amp = TensorField.(Ref(pt),xi./3) # solutions amplitude
mode = TensorField.(graphbundle.(amp),xi) # make 3D surface
mesh(mode[7]); wireframe!(pt) # figure modes are 4,5,7,8,6,9
```


To build on the `FiberBundle` functionality of `Cartan`, the numerical analysis package `Adapode` is being developed to provide extra utilities for finite element method assemblies.
Poisson equation (``-\nabla\cdot(c\nabla u) = f``) syntax or transport (``-\epsilon\nabla^2u+c\cdot\nabla u = f``) equations with finite element methods can be expressed in terms of methods like `volumes` using exterior products or `gradienthat` by applying the exterior algebra principles discussed.
Global `Grassmann` element assembly problems involve applying geometric algebra locally per element basis and combining it with a global manifold topology.

```julia
function solvepoisson(t,e,c,f,k,gD=0,gN=0)
    m = volumes(t)
    b = assembleload(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,k,gD,gN)
    return TensorField(t,(A+R)\(b+r))
end
function solvetransport(t,e,c,f=1,eps=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,eps,m,g)
    b = assembleload(t,f,m)
    C = assembleconvection(t,c,m,g)
    TensorField(t,solvedirichlet(A+C,b,e))
end
function solvetransportdiffusion(tf,ek,c,d,gD=0,gN=0)
    t,f,e,k = base(tf),fiber(tf),base(ek),fiber(ek)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,c,m,g)
    b = means(immersion(t),f)
    C = assembleconvection(t,b,m,g)
    Sd = assembleSD(t,sqrt(d)*b,m,g)
    R,r = assemblerobin(e,k,gD,gN)
    return TensorField(t,(A+R-C'+Sd)\r)
end
```
More general problems for finite element boundary value problems are also enabled by mesh representations imported into `Cartan` from external sources and computationally operated on in terms of `Grassmann` algebra.
Many of these methods automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
Further advanced features such as `DiscontinuousTopology` have been implemented and the `LagrangeTopology` variant of `SimplexTopology` is being used in research.


*Example* (Heatflow around airfoil).
`FlowGeometry` builds on `Cartan` to provide NACA airfoil shapes, and `Adapode` can solve transport diffusion.
```julia
using Grassmann, Cartan, Adapode, FlowGeometry, MATLAB, Makie
pt,pe = initmesh(decsg(NACA"6511"),"hmax"=>0.1)
tf = solvepoisson(pt,pe,1,0,
    x->(x[2]>3.49 ? 1e6 : 0.0),0,x->(x[2]<-1.49 ? 1.0 : 0.0))
function kappa(z); x = base(z)
    if x[2]<-1.49 || sqrt((x[2]-0.5)^2+x[3]^2)<0.51
        1e6
    else
        x[2]>3.49 ? fiber(z)[1] : 0.0
    end
end
gtf = -gradient(tf)
kf = kappa.(gtf(immersion(pe)))
tf2 = solvetransportdiffusion(gtf,kf,0.01,1/50,
    x->(sqrt((x[2]-0.5)^2+x[3]^2)<0.7 ? 1.0 : 0.0))
wireframe(pt)
streamplot(gtf,-0.3..1.3,-0.2..0.2)
mesh(tf2)
```


*Example*. Most finite element methods `using Grassmann, Cartan` automatically generalize to higher dimension manifolds with e.g. tetrahedra,
and the author has contributed to packages such as *Triangulate.jl*, *TetGen.jl*.
```julia
using Grassmann, Cartan, Adapode, FlowGeometry, MiniQhull, TetGen
ps = sphere(sphere(∂(delaunay(PointCloud(sphere())))))
pt,pe = tetrahedralize(cubesphere(),"vpq1.414a0.1";
    holes=[TetGen.Point(0.0,0.0,0.0)])
tf = solvepoisson(pt,pe,1,0,
    x->(x[2]>1.99 ? 1e6 : 0.0),0,x->(x[2]<-1.99 ? 1.0 : 0.0))
streamplot(-gradient(tf),-1.1..1.1,-1.1..1.1,-1.1..1.1,
    gridsize=(10,10,10))
wireframe!(ps)
```

*Example* (Maxwell's equations rewritten).
`Cartan` has Nedelec edge interpolation useful for solving the time harmonic wave equation.
Form Maxwell's equations using the Faraday bivector ``dA`` with ``ddA = 0``,
```math
Ev_t + \star(Bv_t) = (\nabla V - \partial_t A)v_t + \star((\star dA)v_t) = dA,
```
where ``E`` is electric field, ``B`` magnetic field, ``A`` is vector potential.
```math
ddA = 0 \Longleftrightarrow \begin{cases} \partial B = 0 & \text{Gauss's law} \\ \star dE = -\partial_t B & \text{Faraday's law} \end{cases}
```
```math
\star d\star dA = J \Longleftrightarrow \begin{cases} \partial E = \rho & \text{Gauss's law} \\ \star dB = J + \partial_t E & \text{Ampere's law} \end{cases}
```
Maxwell's equations simplify to a single spacetime wave equation.
```math
\nabla(E v_t + \star(B v_t)) = \nabla dA = \star d\star dA = \nabla^2A = J
```



*Example* (Stokes theorem).
Paraboloid ``S=\gamma(X)`` bound by compact support `disk` of radius 3, with circle ``f([0,2\pi])=\partial(S)``, and vector field ``F``:
```julia
using Grassmann, Cartan, Makie # GLMakie
square = TensorField(ProductSpace(-3:0.003:3,-3:0.003:3))
cube = TensorField(ProductSpace(-4:0.1:4,-4:0.1:4,-1:0.2:10))
disk = (x->float(abs(x)<3)).(square) # compact support
paraboloid(x) = 9-x[1]*x[1]-x[2]*x[2]
S = graph(disk*paraboloid.(square))
F(x) = Chain(2x[3]-x[2],x[1]+x[3],3x[1]-2x[2])
mesh(S,normalnorm)
scaledarrows!(S,disk*unitnormal(S),gridsize=(22,22))
streamplot!(F.(cube),gridsize=(11,11,11))
```
```math
\int_S \nabla\times F\cdot dS = \int_X (\nabla\times F) \cdot \left(\frac{\partial\gamma(x)}{\partial x_1}\times\frac{\partial\gamma(x)}{\partial x_2}\right)dx_1dx_2
```
```julia
integrate(disk*(curl(F.(cube)).(S) ⋅ normal(S)))
```
```math
\int_{\partial(S)} F\cdot ds = \int_0^{2\pi} F(f(t))\cdot f'(t) dt
```
```julia
t = TensorField(0:0.001:2pi)
f(t) = Chain(3cos(t[1]),3sin(t[1]),0.0)
integrate(F.(f.(t)) ⋅ tangent(f.(t)))
```
```math
\int_S \nabla\times F\cdot dS = \int_{\partial(S)} F\cdot ds
```
```math
56.5474 \approx 56.547 \approx 56.548667764616276 \approx 18\pi
```
Both integration techniques come out to the same answer, this is called Stokes theorem, a special case of the more general Stokes-Cartan theorem.


## Summary of Grassmann.jl and Cartan.jl

*Grassmann.jl* and *Cartan.jl* pioneered many computational language design aspects for fully generalized high performance computing with differential geometric algebra.
All of the mathematical types and operations in this program were implemented from scratch with fundamental mathematical principles merged to Julia’s type polymorphism code generation, which has been refined and is being optimized for scientific computing over time.

This leads to the capability for multiple dispatch polymorphisms with type aliases such as `Scalar`, `GradedVector`, `Bivector`, `Trivector`, or also `Quaternion`, `Simplex`, etc.
There are aliases such as `Positions`, `Interval`, `IntervalRange`, `Rectangle`, `Hyperrectangle`, `RealRegion`, `RealSpace`, or
the many aliases of the type `TensorField`, such as `ElementMap`, `SimplexMap`, `FaceMap`, `IntervalMap`, `RectangleMap`, `HyperrectangleMap`,
`Variation`, `ParametricMap`, `RealFunction`, `PlaneCurve`, `SpaceCurve`, `AbstractCurve`, `SurfaceGrid`, `VolumeGrid`, `ScalarGrid`, `CliffordField`, `DiagonalField`,
`EndomorphismField`, `OutermorphismField`, `ComplexMap`, `PhasorField`, or `QuaternionField`, `SpinorField`, `GradedField`, `ScalarField` `VectorField` `BivectorField`, `TrivectorField`.
Versatility of the `Grassmann` and `Cartan` type system opens up many possibilities for computational language design.

This is a new paradigm of geometric algebra, anti-symmetric tensor products, rotational algebras, bivector groups, and multilinear Lie brackets.
Algebra based on Leibniz differentials and Grassmann's exterior calculus extended with `TensorField` sections over a `FrameBundle`
yields differential geometric algebra based on the `ImmersedTopology` of a `FiberBundle`. 
The *sector integral theorem* is a new alternative specialization to the Stokes-Cartan theorem for general integrals in differential geometry, relating an integral on a manifold and an integral on its boundary.
Sector integral theory is a new alternative formalism enabling `Cartan` style calculations.

By Grassmann's exterior & interior products, the Hodge-DeRahm chain complex from cohomology theory is
```math
0 \,\underset{\partial}{\overset{d}{\rightleftarrows}}\, \Omega^0(M) \,\underset{\partial}{\overset{d}{\rightleftarrows}}\, \Omega^1(M) \,\underset{\partial}{\overset{d}{\rightleftarrows}}\, \cdots \,\underset{\partial}{\overset{d}{\rightleftarrows}}\, \Omega^n(M) \,\underset{\partial}{\overset{d}{\rightleftarrows}}\, 0,
```
having dimensional equivalence brought by the Grassmann-Hodge complement,
```math
\mathcal H^{n-p}M \cong \frac{\text{ker}(d\Omega^{n-p}M)}{\text{im}(d\Omega^{n-p+1}M)}, \qquad \dim\mathcal H^pM = \dim\frac{\text{ker}(\partial\Omega^pM)}{\text{im}(\partial\Omega^{p+1}M)}.
```
The rank of the grade ``p`` boundary incidence operator is
```math
\text{rank}\langle\partial\langle M\rangle_{p+1}\rangle_p = \min\{\dim\langle\partial\langle M\rangle_{p+1}\rangle_p,\dim\langle M\rangle_{p+1}\}.
```
Invariant topological information can be computed using the rank of homology,
```math
b_p(M) = \dim\langle M\rangle_{p+1} - \text{rank}\langle\partial\langle M\rangle_{p+1}\rangle_p - \text{rank}\langle\partial\langle M\rangle_{p+2}\rangle_{p+1}
```
are the Betti numbers with Euler characteristic ``\chi(M) = \sum_p (-1)^pb_p``.

Grassmann algebra is a unifying mathematical foundation.
Improving efficiency of multi-disciplinary research using differential geometric algebra by relying on universal mathematical principles is possible.
Transforming superficial knowledge into deeper understanding is then achieved with the unified foundations widely applicable to the many different sub-disciplines related to geometry and mathematical physics.
During the early stages when *Grassmann.jl* and *Cartan.jl* were being developed, many new computational language design principles were pioneered for differential geometric algebra research and development with a modern interactive scientific programming language.
With the interest surrounding the project increasing, there have been some other similar projects taking inspiration from the *Grassmann.jl* computational language design and thus validating the concepts.

While some of the computational language designs in *Grassmann.jl* and *Cartan.jl* may seem like obvious choices for people seeing the completed idea, please be aware that it has taken an enormous amount of creativity and effort to make the many different decisions for these projects.
The style of computational language the author wanted to use didn't exist yet before, so if it really was such an obvious design--then why didn't it exist before?
It took a lot of deep thinking and trying out previously overlooked ideas.

[![JuliaCon 2019](https://img.shields.io/badge/JuliaCon-2019-red)](https://www.youtube.com/watch?v=eQjDN0JQ6-s)
[![Grassmann.jl YouTube](https://img.shields.io/badge/Grassmann.jl-YouTube-red)](https://youtu.be/worMICG1MaI)
[![PDF 2019](https://img.shields.io/badge/PDF-2019-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-juliacon-2019.pdf)
[![PDF 2021](https://img.shields.io/badge/PDF-2021-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=differential-geometric-algebra-2021.pdf)
[![PDF 2025](https://img.shields.io/badge/PDF-2025-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cartan.crucialflow.com)

* Michael Reed, [Differential geometric algebra with Leibniz and Grassmann](https://crucialflow.com/grassmann-juliacon-2019.pdf) (2019)
* Michael Reed, [Foundatons of differential geometric algebra](https://vixra.org/abs/2304.0228) (2021)
* Michael Reed, [Multilinear Lie bracket recursion formula](https://vixra.org/abs/2412.0034) (2024)
* Michael Reed, [Differential geometric algebra: compute using Grassmann.jl and Cartan.jl](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf) (2025)
* Emil Artin, [Geometric Algebra](https://archive.org/details/geometricalgebra033556mbp) (1957)
* John Browne, [Grassmann Algebra, Volume 1: Foundations](https://www.grassmannalgebra.com/) (2011)
* C. Doran, D. Hestenes, F. Sommen, and N. Van Acker, [Lie groups as spin groups](http://geocalc.clas.asu.edu/pdf/LGasSG.pdf), J. Math Phys. (1993)
* David Hestenes, [Universal Geometric Algebra](http://lomont.org/math/geometric-algebra/Universal%20Geometric%20Algebra%20-%20Hestenes%20-%201988.pdf), Pure and Applied (1988)
* David Hestenes, Renatus Ziegler, [Projective Geometry with Clifford Algebra](http://geocalc.clas.asu.edu/pdf/PGwithCA.pdf), Acta Appl. Math. (1991)
* David Hestenes, [Tutorial on geometric calculus](http://geocalc.clas.asu.edu/pdf/Tutorial%20on%20Geometric%20Calculus.pdf). Advances in Applied Clifford Algebra (2013)
* Lachlan Gunn, Derek Abbott, James Chappell, Ashar Iqbal, [Functions of multivector variables](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4361175/pdf/pone.0116943.pdf) (2011)
* Aaron D. Schutte, [A nilpotent algebra approach to Lagrangian mechanics and constrained motion](https://www-robotics.jpl.nasa.gov/publications/Aaron_Schutte/schutte_nonlinear_dynamics_1.pdf) (2016)
* Vladimir and Tijana Ivancevic, [Undergraduate lecture notes in DeRahm-Hodge theory](https://arxiv.org/abs/0807.4991). arXiv (2011)
* Peter Woit, [Clifford algebras and spin groups](http://www.math.columbia.edu/~woit/LieGroups-2012/cliffalgsandspingroups.pdf), Lecture Notes (2012)

```
       _           _                         _
      | |         | |                       | |
   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
 | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|

   https://github.com/chakravala
   https://crucialflow.com
   ____  ____    ____   _____  _____ ___ ___   ____  ____   ____
  /    T|    \  /    T / ___/ / ___/|   T   T /    T|    \ |    \
 Y   __j|  D  )Y  o  |(   \_ (   \_ | _   _ |Y  o  ||  _  Y|  _  Y
 |  T  ||    / |     | \__  T \__  T|  \_/  ||     ||  |  ||  |  |
 |  l_ ||    \ |  _  | /  \ | /  \ ||   |   ||  _  ||  |  ||  |  |
 |     ||  .  Y|  |  | \    | \    ||   |   ||  |  ||  |  ||  |  |
 l___,_jl__j\_jl__j__j  \___j  \___jl___j___jl__j__jl__j__jl__j__j
 _________                __                  __________
 \_   ___ \_____ ________/  |______    ____   \\       /
 /    \  \/\__  \\_  __ \   __\__  \  /    \   \\     /
 \     \____/ __ \|  | \/|  |  / __ \|   |  \   \\   /
  \______  (____  /__|   |__| (____  /___|  /    \\ /
         \/     \/                 \/     \/      \/
```
