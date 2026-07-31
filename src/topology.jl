
#   This file is part of Cartan.jl
#   It is licensed under the GPL license
#   Cartan Copyright (C) 2023 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

using MeshTopology

export ProductSpace, RealRegion, NumberLine, Rectangle, Hyperrectangle, ⧺, ⊕, resample
export affinemanifold, isfull

resize_lastdim!(x::Vector,i) = resize!(x,i)

# ProductSpace

affinemanifold(N::Int) = Submanifold(N+2)(list(2,N+1)...)
const affmanifold = affinemanifold
@generated function affinepoint(p::Chain{V,1,T}) where {V,T}
    :(Chain{$(V(list(1,mdims(V)+1)...))}(Values(one(T),$([:(@inbounds p[$i]) for i ∈ list(1,mdims(V))]...))))
end

"""
    ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}

Can be constructed with `\\oplus` operation `⊕` and `AbstractRange`,
```julia
julia> (0:0.1:1)⊕(0:0.1:1)
11×11 ProductSpace{⟨_11_⟩, Float64, 2, 2, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}:
...
```
generating a lazy array of `Chain{V,1}` point vectors from the input ranges.
```Julia
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
```
"""
struct ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}
    v::Values{M,S} # how to deal with T???
    ProductSpace{V,T,N}(v::Values{M,S}) where {V,T,N,M,S} = new{Grassmann.DirectSum.submanifold(V),T,N,M,S}(v)
    ProductSpace{V,T}(v::Values{M,S}) where {V,T,M,S} = new{Grassmann.DirectSum.submanifold(V),T,mdims(V),M,S}(v)
end

const RealRegion{V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N,N,S}
const NumberLine{V,T,S} = RealRegion{V,T,1,S}
const Rectangle{V,T,S} = RealRegion{V,T,2,S}
const Hyperrectangle{V,T,S} = RealRegion{V,T,3,S}

RealRegion{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
RealRegion(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{affmanifold(N),T,N}(v)
ProductSpace{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
ProductSpace(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{affmanifold(N),T,N}(v)

Base.split(t::ProductSpace) = t.v
Base.show(io::IO,t::RealRegion{V,T,N,<:AbstractRange} where {V,T,N}) = print(io,'(',Chain(getindex.(split(t),1)),"):(",Chain(Number.(getproperty.(split(t),:step))),"):(",Chain(getindex.(split(t),length.(split(t)))),')')

(::Base.Colon)(min::Chain{V,1,T},step::Chain{V,1,T},max::Chain{V,1,T}) where {V,T} = ProductSpace{V,T}(Colon().(value(min),value(step),value(max)))

Base.iterate(t::RealRegion) = (getindex(t,1),1)
Base.iterate(t::RealRegion,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

resize_lastdim!(m::ProductSpace,i) = (resize!(m.v[end],i); m)

isrange(m::AbstractRange) = true
isrange(m::AbstractVector) = false
isrange(m::ProductSpace) = prod(isrange.(split(m)))

@generated Base.size(m::RealRegion{V}) where V = :(($([:(size(@inbounds m.v[$i])...) for i ∈ 1:mdims(V)]...),))
@generated Base.getindex(m::RealRegion{V,T,N},i::Vararg{Int}) where {V,T,N} = :(Chain{V,1,T}(Values{N,T}($([:((@inbounds m.v[$j])[@inbounds i[$j]]) for j ∈ 1:N]...))))
Base.getindex(m::NumberLine{V,T},i::Int) where {V,T} = Chain{V,1,T}(Values(((@inbounds m.v[1])[i],)))
@pure Base.getindex(t::RealRegion,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.eltype(::Type{<:ProductSpace{V,T,N}}) where {V,T,N} = Chain{V,1,T,N}

Base.IndexStyle(::RealRegion) = IndexCartesian()
function Base.getindex(A::RealRegion, I::Int)
    Base.@_inline_meta
    @inbounds getindex(A, Base._to_subscript_indices(A, I)...)
end
function Base._to_subscript_indices(A::RealRegion, i::Int)
    Base.@_inline_meta
    Base._unsafe_ind2sub(A, i)
end
function Base._ind2sub(A::RealRegion, ind)
    Base.@_inline_meta
    Base._ind2sub(axes(A), ind)
end

"""
    ⊕(v::AbstractVector{<:Real}...)

Constructs a direct sum basis space using the Cartesian `ProductSpace` implementation.
"""
⊕(a::AbstractVector{<:Real}...) = RealRegion(Values(a))
⊕(a::ProductSpace,b::AbstractVector{<:Real}) = RealRegion(Values(a.v...,b))
⊕(a::AbstractVector{<:Real},b::ProductSpace) = RealRegion(Values(a,b.v...))
⊕(a::ProductSpace,b::ProductSpace) = RealRegion(Values(a.v...,b.v...))
cross(a::ProductSpace,b::AbstractVector{<:Real}) = a⊕b
cross(a::ProductSpace,b::ProductSpace) = a⊕b

RealRegion(a::AbstractVector{<:Real}...) = RealRegion(Values(a))
ProductSpace(a::AbstractVector{<:Real}...) = ProductSpace(Values(a))
RealRegion{V}(a::AbstractVector{<:Real}...) where V = RealRegion{V}(Values(a))
ProductSpace{V}(a::AbstractVector{<:Real}...) where V = ProductSpace{V}(Values(a))

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

widths(t::AbstractVector) = t[end]-t[1]
widths(t::ProductSpace) = widths.(t.v)

remove(t::ProductSpace{V,T,2} where {V,T},::Val{1}) = (@inbounds t.v[2])
remove(t::ProductSpace{V,T,2} where {V,T},::Val{2}) = (@inbounds t.v[1])
@generated remove(t::ProductSpace{V,T,N} where {V,T},::Val{J}) where {N,J} = :(ProductSpace(t.v[$(Values([i for i ∈ 1:N if i≠J]...))]))

# 1
(m::ProductSpace)(c::Colon,i::Int...) = (@inbounds m.v[1])
(m::ProductSpace)(i::Int,c::Colon,j::Int...) = m.v[2]
(m::ProductSpace)(i::Int,j::Int,c::Colon,k::Int...) = m.v[3]
(m::ProductSpace)(i::Int,j::Int,k::Int,c::Colon,l::Int...) = m.v[4]
(m::ProductSpace)(i::Int,j::Int,k::Int,l::Int,c::Colon,o::Int...) = m.v[5]

#= 2 - 0
(m::ProductSpace)(c::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3)])
(m::ProductSpace)(c::Colon,i::Int,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,4)])
(m::ProductSpace)(c::Colon,i::Int,j::Int,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(1,5)])
# 2 - 1
(m::ProductSpace)(i::Int,c::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(2,4)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(2,5)])
# 2 - 2
(m::ProductSpace)(i::Int,j::Int,c::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(3,4)])
(m::ProductSpace)(i::Int,j::Int,c::Colon,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(3,5)])
# 2 - 3
(m::ProductSpace)(i::Int,j::Int,k::Int,c::Colon,::Colon,l::Int...) = ProductSpace(m.v[Values(4,5)])

# 3 - 0 - 0
(m::ProductSpace)(c::Colon,::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2,3)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,4)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,2,5)])
# 3 - 0 - 1
(m::ProductSpace)(c::Colon,i::Int,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3,4)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,3,5)])
# 3 - 0 - 2
(m::ProductSpace)(c::Colon,i::Int,j::Int,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(1,4,5)])
# 3 - 1
(m::ProductSpace)(i::Int,c::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3,4)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(2,4,5)])
(m::ProductSpace)(i::Int,c::Colon,::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(2,3,5)])
# 3 - 2
(m::ProductSpace)(i::Int,j::Int,c::Colon,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(3,4,5)])

# 4
(m::ProductSpace)(c::Colon,::Colon,::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2,3,4)])
(m::ProductSpace)(c::Colon,::Colon,::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,3,5)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,4,5)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3,4,5)])
(m::ProductSpace)(i::Int,c::Colon,::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3,4,5)])=#

iscolon(a::Type{Colon},b::T) where T = Values{1,T}(b)
iscolon(a::Type{Int},b::T) where T = Values{0,T}()
iscolon(a::Colon,b::T) where T = Values{1,T}(b)
iscolon(a::Int,b::T) where T = Values{0,T}()
colon_permutation(args...) = vcat(iscolon.(Values(args),Cartan.list(1,length(args)))...)
@generated function (m::ProductSpace)(args::Union{Int,Colon}...)
    perm = colon_permutation(args...)
    if isone(length(perm))
        isone(perm[1]) ? :(@inbounds m.v[$(perm[1])]) : :(m.v[$(perm[1])])
    else
        :(ProductSpace(m.v[$perm]))
    end
end

import MeshTopology: resample

resample(m::ProductSpace,i::NTuple=size(m)) = ProductSpace(resample.(split(m),i))

import MeshTopology: CrossRange, crossrange

export CrossRange

# ImmersedTopology

export ImmersedTopology, ProductTopology, SimplexTopology, SimplexManifold
export QuotientTopology, OpenTopology, CompactTopology
export topology, immersion, vertices, iscover

import MeshTopology: ImmersedTopology, immersion, sdims, immersiontype, fullimmersion
import MeshTopology: topology, subelements, ProductTopology, resize, exclude, elements
import MeshTopology: refval, refnodes, RefInt, SimplexTopology, bundle, fulltopology
import MeshTopology: totalelements, totalnodes!, totalnodes, nodes, fullvertices, vertices
import MeshTopology: verticesinv, getimage, getfacet, istotal, isfull, iscover, untotal
import MeshTopology: fullimmersion_vertices, subtopology, getelement, subimmersion, _axes

# DiscontinuousTopology

export DiscontinuousTopology, discontinuous, disconnect, continuous

import MeshTopology: refine, DiscontinuousTopology, discontinuousvertices, isdiscontinuous
import MeshTopology: isdisconnected, continuous, discontinuous, disconnect
#import MeshTopology: VectorTopology

# LagrangeTopology

export LagrangeTopology, LagrangeTriangles, LagrangeTetrahedra, cornertopology
export totalcornernodes, totaledgesnodes, totalcenternodes
export cornernodes, edgesnodes, centernodes

import MeshTopology: simplexnumber, trinum, tetnum, LagrangeTopology, totaledges
import MeshTopology: totalfacets, totalcornernodes, totaledgesnodes, totalfacetsnodes
import MeshTopology: totalcenternodes, cornernodes, edgesnodes, facetsnodes, centernodes
import MeshTopology: lagrangesimplex, centersimplex, facetsimplex, edgesimplex
import MeshTopology: LagrangeEdges, LagrangeTriangles, LagrangeTetrahedra
import MeshTopology: cornertopology, edges, edgesindices, facets, facetsindices
import MeshTopology: lagrangevertices2, lagrangevertices3, lagrangevertices4
import MeshTopology: edgesindex, facetsindex, centerindex, getedge, getlagrange1
import MeshTopology: getlagrange2, getlagrange3, getlagrange4, _getelement
import MeshTopology: getelement1, getelement2, getelement3, getelement4, getelement

# Global

export Global

"""
    Global{N,T} <: AbstractArray{T,N}

Represents an `AbstractArray` where every local value is globally the same.
```julia
julia> Global{1}(InducedMetric())
Global{1}(InducedMetric())

julia> ans[1]
InducedMetric()
```
For example, `Global{N,InducedMetric}` is commonly used for a globally induced metric.
"""
struct Global{N,T} <: AbstractArray{T,N}
    v::T
    #n::NTuple{N,Int}
    #Global{N}(v::T,n=(1,)) where {T,N} = new{N,T}(v,n)
    Global{N}(v::T) where {T,N} = new{N,T}(v)
end

#Base.size(t::Global) = t.n
Base.getindex(t::Global,i::Vararg{Int}) = t.v
Base.getindex(t::Global,i::CartesianIndex) = t.v
Base.setindex!(t::Global{N,InducedMetric} where N,v::InducedMetric,i::Vararg{Int}) = v
Base.resize!(t::Global,i) = t
@pure Base.eltype(::Type{<:Global{T}}) where T = T

Base.vec(t::Global{1}) = t
Base.vec(t::Global) = Global{1}(t.v)

Base.IndexStyle(::Global) = IndexCartesian()
function Base.getindex(A::Global, I::Int)
    Base.@_inline_meta
    A.v
end

Base.show(io::IO,t::Global{N}) where N = print(io,"Global{$N}($(t.v))")
Base.show(io::IO, ::MIME"text/plain", t::Global) = show(io,t)

#metricextensor(c::AbstractArray{T,N} where T) where N = Global{N}(InducedMetric(),size(c))
ref(itr::InducedMetric) = Ref(itr)
ref(itr::Global) = Ref(itr.v)
ref(itr) = itr
refmetric(x) = ref(metricextensor(x))

# LocalFiber

"""
    LocalFiber{B,F} <: Number

Defines abstract local trivial bundle with `basetype` of `B` and `fibertype` of `F`.
```Julia
base(s) # ::B
fiber(s) # ::F
basetype(s) # B
fibertype(s) # F
```
A `LocalFiber{B,F}` consists of two components: `B`, which represents the `base` manifold, and `F`, which represents the `fiber` bundle over `B`.
"""
abstract type LocalFiber{B,F} <: Number end
Base.@pure isfiber(::LocalFiber) = true
Base.@pure isfiber(::Any) = false

fiber(s) = s
fibertype(s) = typeof(s)
fibertype(::Type{T}) where T = T
base(s::LocalFiber) = s.v.first
fiber(s::LocalFiber) = s.v.second
basepoint(s::LocalFiber) = point(base(s))
basetype(::LocalFiber{B}) where B = B
pointtype(::LocalFiber{B}) where B = basetype(B)
fibertype(::LocalFiber{B,F} where B) where F = F
metrictype(::LocalFiber{B,F} where B) where F = fibertype(B)
basetype(::Type{<:LocalFiber{B}}) where B = B
pointtype(::Type{<:LocalFiber{B}}) where B = basetype(B)
fibertype(::Type{<:LocalFiber{B,F} where B}) where F = F
metrictype(::Type{<:LocalFiber{B,F} where B}) where F = fibertype(B)
base(s::Real) = s

Base.getindex(s::LocalFiber) = s.v.first
Base.getindex(s::LocalFiber,i::Int...) = getindex(s.v.second,i...)
Base.getindex(s::LocalFiber,i::Integer...) = getindex(s.v.second,i...)

function Base.show(io::IO, s::LocalFiber)
    fibertype(s) <: InducedMetric && (return show(io, base(s)))
    p = s.v
    Base.isdelimited(io, p) && return show_pairtyped(io, s)
    typeinfos = Base.gettypeinfos(io, p)
    for i = (1, 2)
        io_i = IOContext(io, :typeinfo => typeinfos[i])
        Base.isdelimited(io_i, p[i]) || print(io, "(")
        show(io_i, p[i])
        Base.isdelimited(io_i, p[i]) || print(io, ")")
        i == 1 && print(io, get(io, :compact, false)::Bool ? "↦" : " ↦ ")
    end
end

function show_pairtyped(io::IO, s::LocalFiber{B,F}) where {B,F}
    show(io, typeof(s))
    show(io, (base(s), fiber(s)))
end

# Coordinate

export Coordinate, point

"""
    Coordinate{P,G} <: LocalFiber{P,G} <: Number

Defines a `Coordinate` bundled type with `pointtype` of `P` and `metrictype` of `G`.
```Julia
point(s) # ::P
metricextensor(s) # ::G
pointtype(s) # P
metrictype(s) # G
```
A `Coordinate{P,G}` consists of two components: `P`, which represents the `point` manifold, and `G`, which represents the `metricextensor` bundle over `P`.
"""
struct Coordinate{P,G} <: LocalFiber{P,G}
    v::Pair{P,G}
    Coordinate(v::Pair{P,G}) where {P,G} = new{P,G}(v)
    Coordinate(p::P,g::G=InducedMetric()) where {P,G} = new{P,G}(p=>g)
end

point(c) = c
point(c::Coordinate) = base(c)
point(c::LocalFiber) = point(base(c))
metricextensor(c) = InducedMetric()
metricextensor(c::Coordinate) = fiber(c)
metrictensor(c) = InducedMetric()
metrictensor(c::Coordinate) = TensorOperator(fiber(c)[1])
pointtype(::Coordinate{P}) where P = P
pointtype(::Type{<:Coordinate{P}}) where P = P
metrictype(::Coordinate{P,G} where P) where G = G
metrictype(::Type{<:Coordinate{P,G} where P}) where G = G

Base.getindex(s::Coordinate,i::Int...) = getindex(s.v.first,i...)
Base.getindex(s::Coordinate,i::Integer...) = getindex(s.v.first,i...)

graph(s::LocalFiber{<:AbstractReal,<:AbstractReal}) = Chain(Real(base(s)),Real(fiber(s)))
graph(s::LocalFiber{<:AbstractReal,<:Chain}) = Chain(Real(base(s)),value(fiber(s))...)
graph(s::LocalFiber{<:Chain,<:AbstractReal}) = Chain(value(base(s))...,Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:AbstractReal},<:AbstractReal}) = Chain(Real(basepoint(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:AbstractReal},<:Chain}) = Chain(Real(basepoint(s)),value(fiber(s))...)
graph(s::LocalFiber{<:Coordinate{<:Chain},<:AbstractReal}) = Chain(value(basepoint(s))...,Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:Chain},<:Chain}) = Chain(value(basepoint(s))...,value(fiber(s))...)

export Positions, Interval, RealSpace, ComplexSpace
const Positions{P<:Chain,G} = AbstractVector{<:Coordinate{P,G}}
const Interval{P<:AbstractReal,G} = AbstractVector{<:Coordinate{P,G}}
#const RectanglePatch{P,G} = AbstractMatrix{<:Coordinate{P,G}}
#const HyperrectanglePatch{P,G} = AbstractArray{<:Coordinate{P,G},3}
const RealSpace{N,P<:Chain{V,1,<:Real} where V,G} = AbstractArray{<:Coordinate{P,G},N}
const ComplexSpace{N,P<:Chain{V,1,<:Complex} where V,G} = AbstractArray{<:Coordinate{P,G},N}
#const RectanglePatch{P,G} = RealSpace{2,P,G}
#const HyperrectanglePatch{P,G} = RealSpace{3,P,G}

# LocalPrincipal

"""
    LocalPrincipal{M,G} <: LocalFiber{M,G} <: Number

A `LocalPrincipal` bundled with `principalbasetype` of `M` and `principalfibertype` of `G`.
```Julia
principalbase(s) # ::M
principalfiber(s) # ::G
principalbasetype(s) # M
principalfibertype(s) # G
```
A `LocalPrincipal{M,G}` consists of two components: `M`, which represents the `principalbase` manifold, and `G`, which represents the `principalfiber` bundle over `M`.
"""
struct LocalPrincipal{M,G} <: LocalFiber{M,G}
    v::Pair{M,G}
    LocalPrincipal(v::Pair{M,G}) where {M,G} = new{M,G}(v)
    LocalPrincipal(x::M,g::G) where {M,G} = new{M,G}(x=>g)
end

# LocalTensor

"""
    LocalTensor{B,F} <: LocalFiber{B,F} <: Number

Defines a local bundled type with `basetype` of `B` and `fibertype` of `F`.
```Julia
base(s) # ::B
fiber(s) # ::F
basetype(s) # B
fibertype(s) # F
```
A `LocalTensor{B,F}` consists of two components: `B`, which represents the `base` manifold, and `F`, which represents the `fiber` bundle over `B`.
"""
struct LocalTensor{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalTensor(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalTensor(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalTensor(b::B,f::LocalTensor{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalTensor(b::LocalTensor{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

export Section, LocalTensor
const Section = LocalTensor
const ↦, domain, codomain = LocalTensor, base, fiber
↤(F,B) = B ↦ F

localfiber(x) = x
localfiber(x::LocalTensor) = fiber(x)

(m::TensorNested)(x::LocalTensor) = LocalTensor(base(x),m(fiber(x)))
@inline Base.:<<(a::LocalFiber,b::LocalFiber) = contraction(b,~a)
@inline Base.:>>(a::LocalFiber,b::LocalFiber) = contraction(~a,b)
@inline Base.:<(a::LocalFiber,b::LocalFiber) = contraction(b,a)
Base.sign(s::LocalTensor) = LocalTensor(base(s),sign(Real(fiber(s))))
Base.inv(a::LocalTensor{B,<:Real} where B) = LocalTensor(base(a), inv(fiber(a)))
Base.inv(a::LocalTensor{B,<:Complex} where B) = LocalTensor(base(a), inv(fiber(a)))
Base.:/(a::LocalTensor,b::LocalTensor{B,<:Real} where B) = LocalTensor(base(a), fiber(a)/fiber(b))
Base.:/(a::LocalTensor,b::LocalTensor{B,<:Complex} where B) = LocalTensor(base(a), fiber(a)/fiber(b))
LinearAlgebra.:×(a::LocalTensor{R},b::LocalTensor{R}) where R = TensorField(base(a), ⋆(fiber(a)∧fiber(b),metricextensor(a)))
Grassmann.compound(t::LocalTensor,i::Val) = LocalTensor(base(t), compound(fiber(t),i))
Grassmann.compound(t::LocalTensor,i::Int) = LocalTensor(base(t), compound(fiber(t),i))
Grassmann.eigen(t::LocalTensor,i::Val) = LocalTensor(base(t), eigen(fiber(t),i))
Grassmann.eigen(t::LocalTensor,i::Int) = LocalTensor(base(t), eigen(fiber(t),i))
Grassmann.eigvals(t::LocalTensor,i::Val) = LocalTensor(base(t), eigvals(fiber(t),i))
Grassmann.eigvals(t::LocalTensor,i::Int) = LocalTensor(base(t), eigvals(fiber(t),i))
Grassmann.eigvecs(t::LocalTensor,i::Val) = LocalTensor(base(t), eigvecs(fiber(t),i))
Grassmann.eigvecs(t::LocalTensor,i::Int) = LocalTensor(base(t), eigvecs(fiber(t),i))
Grassmann.eigpolys(t::LocalTensor,G::Val) = LocalTensor(base(t), eigpolys(fiber(t),G))
Base.:<(a::LocalTensor{R},b::LocalTensor{R}) where R = Base.:>(b,a)
Base.:<(a::Number,b::LocalTensor) = Base.:>(b,a)
Base.:<(a::LocalTensor,b::Number) = Base.:>(b,a)
Base.log(s::LocalTensor) = LocalTensor(base(s), Grassmann.log_metric(fiber(s),metricextensor(s)))
for fun ∈ (:inv,:exp,:exp2,:exp10,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:acsch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2)
    @eval Base.$fun(s::LocalTensor) = LocalTensor(base(s), $fun(fiber(s),metricextensor(s)))
end
for type ∈ (:Coordinate,:LocalTensor)
    for tensor ∈ (:Single,:Couple,:PseudoCouple,:Chain,:Spinor,:AntiSpinor,:Multivector,:DiagonalOperator,:TensorOperator,:Outermorphism)
        @eval (T::Type{<:$tensor})(s::$type) = $type(base(s), T(fiber(s)))
    end
    for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad,:transpose,:iszero,:isone,:isnan,:isinf,:isfinite,:floor,:ceil,:round)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:trivector,:pseudoscalar,:value,:curl,:∂,:d,:complementleft,:realvalue,:imagvalue,:outermorphism,:Outermorphism,:DiagonalOperator,:TensorOperator,:eigen,:eigvecs,:eigvals,:eigvalsreal,:eigvalscomplex,:eigvecsreal,:eigvecscomplex,:eigpolys,:pfaffian,:∧,:↑,:↓,:vectorize,:discriminant,:discriminantreal,:discriminantcomplex,:vandermonde,:vandermondereal,:vandermondecomplex,:adjugate,:cofactor)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric,:unit,:complexify,:polarize,:amplitude,:phase)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s),metricextensor(s)))
    end
    for op ∈ (:+,:-,:&,:∧,:∨,:max,:min,:div,:rem,:mod,:mod1,:ldexp)
        let bop = op ∈ (:∧,:∨) ? :(Grassmann.$op) : :(Base.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),$op(fiber(a),fiber(b)))
            $bop(a::Number,b::$type) = $type(base(b), $op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), $op(fiber(a),b))
        end end
    end
    for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
        let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop(fiber(a),fiber(b),metricextensor(a)))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op(fiber(a),b,$((op≠:^ ? () : (:(metricextensor(a)),))...)))
        end end
    end
    @eval begin
        $type(b,f::Function) = $type(b,f(b))
        Grassmann.contraction(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.contraction(fiber(a),fiber(b)))
        LinearAlgebra.norm(s::$type) = $type(base(s), norm(fiber(s)))
        LinearAlgebra.det(s::$type) = $type(base(s), det(fiber(s)))
        LinearAlgebra.tr(s::$type) = $type(base(s), tr(fiber(s)))
        Base.:^(s::$type,n::Int) = $type(base(s), fiber(s)^n)
        (V::Submanifold)(s::$type) = $type(base(s), V(fiber(s)))
        (::Type{T})(s::$type) where T<:Real = $type(base(s), T(fiber(s)))
        (::Type{Complex})(s::$type) = $type(base(s), Complex(fiber(s)))
        (::Type{Complex{T}})(s::$type) where T = $type(base(s), Complex{T}(fiber(s)))
        Grassmann.Phasor(s::$type) = $type(base(s), Phasor(fiber(s)))
        Grassmann.Couple(s::$type) = $type(base(s), Couple(fiber(s)))
        (X::GradedVector)(s::$type) = $type(base(s),X(fiber(s)))
        (::Type{T})(s::$type...) where T<:Chain = @inbounds $type(base(s[1]), Chain(Values(fiber.(s)...)))
    end
    if VERSION≥v"1.9"
        @eval (::Type{T})(s::Union{<:$type,<:Real,<:Complex,<:TensorAlgebra}...) where T<:Chain = @inbounds $type(base(s[1]), Chain(Values(fiber.(s)...)))
    end
end
if VERSION≥v"1.9" && Base.pkgversion(Grassmann)≤v"0.8.42"
    @inline (::Type{T})(x::Union{<:Real,<:Complex,<:TensorAlgebra}...) where T<:Chain = T(x)
end

