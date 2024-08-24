module Cartan

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
# _________                __                  __________
# \_   ___ \_____ ________/  |______    ____   \\       /
# /    \  \/\__  \\_  __ \   __\__  \  /    \   \\     /
# \     \____/ __ \|  | \/|  |  / __ \|   |  \   \\   /
#  \______  (____  /__|   |__| (____  /___|  /    \\ /
#         \/     \/                 \/     \/      \/
#  _____                           ___ _      _     _
# /__   \___ _ __  ___  ___  _ __ / __(_) ___| | __| |___
#   / /\/ _ \ '_ \/ __|/ _ \| '__/ _\ | |/ _ \ |/ _` / __|
#  / / |  __/ | | \__ \ (_) | | / /   | |  __/ | (_| \__ \
#  \/   \___|_| |_|___/\___/|_| \/    |_|\___|_|\__,_|___/

using SparseArrays, LinearAlgebra, Base.Threads, Grassmann, Requires
import Grassmann: value, vector, valuetype, tangent, istangent, Derivation, radius, ⊕
import Grassmann: realvalue, imagvalue, points, metrictensor
import Grassmann: Values, Variables, FixedVector, list
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Base: @pure, OneTo, getindex

export Values, Derivation
export initmesh, pdegrad, det

export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid
export RealFunction, ComplexMap, SpinorField, CliffordField
export ScalarMap, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap, AbstractCurve
export LocalSection, GlobalFiber, LocalFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype, graph
export ProductSpace, RealRegion, NumberLine, Rectangle, Hyperrectangle, ⧺, ⊕

# ProductSpace

struct ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}
    v::Values{M,S} # how to deal with T???
    ProductSpace{V,T,N}(v::Values{M,S}) where {V,T,N,M,S} = new{V,T,N,M,S}(v)
    ProductSpace{V,T}(v::Values{M,S}) where {V,T,M,S} = new{V,T,mdims(V),M,S}(v)
end

const RealRegion{V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N,N,S}
const NumberLine{V,T,S} = RealRegion{V,T,1,S}
const Rectangle{V,T,S} = RealRegion{V,T,2,S}
const Hyperrectangle{V,T,S} = RealRegion{V,T,3,S}

RealRegion{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
RealRegion(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{Submanifold(N),T,N}(v)
ProductSpace{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
ProductSpace(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{Submanifold(N),T,N}(v)

Base.show(io::IO,t::RealRegion{V,T,N,<:AbstractRange} where {V,T,N}) = print(io,'(',Chain(getindex.(t.v,1)),"):(",Chain(Number.(getproperty.(t.v,:step))),"):(",Chain(getindex.(t.v,length.(t.v))),')')

(::Base.Colon)(min::Chain{V,1,T},step::Chain{V,1,T},max::Chain{V,1,T}) where {V,T} = ProductSpace{V,T}(Colon().(value(min),value(step),value(max)))

Base.iterate(t::RealRegion) = (getindex(t,1),1)
Base.iterate(t::RealRegion,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

@generated Base.size(m::RealRegion{V}) where V = :(($([:(size(m.v[$i])...) for i ∈ 1:mdims(V)]...),))
@generated Base.getindex(m::RealRegion{V,T,N},i::Vararg{Int}) where {V,T,N} = :(Chain{V,1,T}($([:(m.v[$j][i[$j]]) for j ∈ 1:N]...)))
@pure Base.getindex(t::RealRegion,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.eltype(::Type{ProductSpace{V,T,N}}) where {V,T,N} = Chain{V,1,T,N}

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

⊕(a::AbstractVector{A},b::AbstractVector{B}) where {A<:Real,B<:Real} = RealRegion(Values(a,b))

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

remove(t::ProductSpace{V,T,2} where {V,T},::Val{1}) = t.v[2]
remove(t::ProductSpace{V,T,2} where {V,T},::Val{2}) = t.v[1]
@generated remove(t::ProductSpace{V,T,N} where {V,T},::Val{J}) where {N,J} = :(ProductSpace(domain(t).v[$(Values([i for i ∈ 1:N if i≠J]...))]))

# ImmersedTopology

export ImmersedTopology, SimplexManifold, topology, immersion, vertices, iscover

abstract type ImmersedTopology{N,P} <: AbstractVector{Values{N,Int}} end
const immersion = ImmersedTopology

top_id = 0

struct SimplexManifold{N,P<:AbstractVector{Int}} <: ImmersedTopology{N,P}
    id::Int
    t::Vector{Values{N,Int}}
    i::P
    p::Int
    SimplexManifold(t::Vector{Values{N,Int}},i::P=vertices(t),p::Int=length(i)) where {N,P} = new{N,P}((global top_id+=1),t,i,p)
end

SimplexManifold(t::Vector{Values{N,Int}},p::Int) where N = SimplexManifold(t,vertices(t),p)

bundle(m::SimplexManifold) = m.id
topology(m::SimplexManifold) = m.t
vertices(m::SimplexManifold) = m.i

Base.size(m::SimplexManifold) = size(topology(m))
Base.length(m::SimplexManifold) = length(topology(m))
Base.axes(m::SimplexManifold) = axes(topology(m))
Base.getindex(m::SimplexManifold,i::Int) = getindex(topology(m),i)
@pure Base.eltype(::Type{ImmersedTopology{N}}) where N = Values{N,Int}
Grassmann.mdims(m::SimplexManifold{N}) where N = N

_axes(t::SimplexManifold{N}) where N = (Base.OneTo(length(t)),Base.OneTo(N))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::SimplexManifold) = Base.array_summary(io, a, _axes(a))

iscover(x::ImmersedTopology) = length(vertices(x)) == x.p
subsym(x) = iscover(x) ? "⊆" : "⊂"

function Base.array_summary(io::IO, a::SimplexManifold, inds::Tuple{Vararg{OneTo}})
    print(io, Base.dims2string(length.(inds)))
    print(io, subsym(a), length(vertices(a)), " ")
    Base.showarg(io, a, true)
end

# Global

export Global

struct Global{N,T} <: AbstractArray{T,N}
    v::T
    #n::NTuple{N,Int}
    #Global{N}(v::T,n=(1,)) where {T,N} = new{N,T}(v,n)
    Global{N}(v::T) where {T,N} = new{N,T}(v)
end

#Base.size(m::Global) = m.n
Base.getindex(m::Global,i::Vararg{Int}) = m.v
Base.getindex(t::Global,i::CartesianIndex) = m.v
@pure Base.eltype(::Type{<:Global{T}}) where T = T

Base.IndexStyle(::Global) = IndexCartesian()
function Base.getindex(A::Global, I::Int)
    Base.@_inline_meta
    A.v
end

Base.show(io::IO,t::Global{N}) where N = print(io,"Global{$N}($(t.v))")
Base.show(io::IO, ::MIME"text/plain", t::Global) = show(io,t)

#metrictensor(c::AbstractArray{T,N} where T) where N = Global{N}(InducedMetric(),size(c))
ref(itr::InducedMetric) = Ref(itr)
ref(itr::Global) = Ref(itr.v)
ref(itr) = itr

# LocalFiber

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
basepointtype(::LocalFiber{B}) where B = pointtype(B)
fibertype(::LocalFiber{B,F} where B) where F = F
basetype(::Type{<:LocalFiber{B}}) where B = B
fibertype(::Type{<:LocalFiber{B,F} where B}) where F = F

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

struct Coordinate{P,G} <: LocalFiber{P,G}
    v::Pair{P,G}
    Coordinate(v::Pair{P,G}) where {P,G} = new{P,G}(v)
    Coordinate(p::P,g::G) where {P,G} = new{P,G}(p=>g)
end

point(c) = c
point(c::Coordinate) = base(c)
point(c::LocalFiber) = point(base(c))
metrictensor(c) = InducedMetric()
metrictensor(c::Coordinate) = fiber(c)

Base.getindex(s::Coordinate,i::Int...) = getindex(s.v.first,i...)
Base.getindex(s::Coordinate,i::Integer...) = getindex(s.v.first,i...)

graph(s::LocalFiber{<:AbstractReal,<:AbstractReal}) = Chain(Real(base(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Chain,<:AbstractReal}) = Chain(value(base(s))...,Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:AbstractReal},<:AbstractReal}) = Chain(Real(basepoint(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:Chain},<:AbstractReal}) = Chain(value(basepoint(s))...,Real(fiber(s)))

export Positions, Interval, RealSpace, ComplexSpace
const Positions{P<:Chain,G} = AbstractVector{<:Coordinate{P,G}}
const Interval{P<:AbstractReal,G} = AbstractVector{<:Coordinate{P,G}}
#const RectanglePatch{P,G} = AbstractMatrix{<:Coordinate{P,G}}
#const HyperrectanglePatch{P,G} = AbstractArray{<:Coordinate{P,G},3}
const RealSpace{N,P<:Chain{V,1,<:Real} where V,G} = AbstractArray{<:Coordinate{P,G},N}
const ComplexSpace{N,P<:Chain{V,1,<:Complex} where V,G} = AbstractArray{<:Coordinate{P,G},N}
#const RectanglePatch{P,G} = RealSpace{2,P,G}
#const HyperrectanglePatch{P,G} = RealSpace{3,P,G}

# LocalSection

struct LocalSection{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalSection(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalSection(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalSection(b::B,f::LocalSection{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalSection(b::LocalSection{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

# LocalTensor

struct LocalTensor{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalTensor(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalTensor(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalTensor(b::B,f::LocalTensor{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalTensor(b::LocalTensor{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

export Section
const Section = LocalTensor
const ↦, domain, codomain = LocalTensor, base, fiber
↤(F,B) = B ↦ F

@inline Base.:<<(a::LocalFiber,b::LocalFiber) = contraction(b,~a)
@inline Base.:>>(a::LocalFiber,b::LocalFiber) = contraction(~a,b)
@inline Base.:<(a::LocalFiber,b::LocalFiber) = contraction(b,a)
for type ∈ (:Coordinate,:LocalSection,:LocalTensor)
    for tensor ∈ (:Single,:Couple,:PseudoCouple,:Chain,:Spinor,:AntiSpinor,:Multivector,:DiagonalOperator,:TensorOperator,:Outermorphism)
        @eval (T::Type{<:$tensor})(s::$type) = $type(base(s), T(fiber(s)))
    end
    for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:inv,:exp,:log,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s),metrictensor(base(s))))
    end
    for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:curl,:∂,:d,:complementleft,:realvalue,:imagvalue,:outermorphism)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s),metrictensor(base(s))))
    end
    for op ∈ (:+,:-,:&,:∧,:∨)
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
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop(fiber(a),fiber(b),metrictensor(base(a))))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op(fiber(a),b,$((op≠:^ ? () : (:(metrictensor(base(a))),))...)))
        end end
    end
    @eval begin
        $type(b,f::Function) = $type(b,f(b))
        Grassmann.contraction(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.contraction(fiber(a),fiber(b)))
        LinearAlgebra.norm(s::$type) = $type(base(s), norm(fiber(s)))
        LinearAlgebra.det(s::$type) = $type(base(s), det(fiber(s)))
        (V::Submanifold)(s::$type) = $type(base(a), V(fiber(s)))
        (::Type{T})(s::$type) where T<:Real = $type(base(s), T(fiber(s)))
        (::Type{Complex})(s::$type) = $type(base(s), Complex(fiber(s)))
        (::Type{Complex{T}})(s::$type) where T = $type(base(s), Complex{T}(fiber(s)))
        Grassmann.Phasor(s::$type) = $type(base(s), Phasor(fiber(s)))
        Grassmann.Couple(s::$type) = $type(base(s), Couple(fiber(s)))
        (::Type{T})(s::$type...) where T<:Chain = @inbounds $type(base(s[1]), Chain(fiber.(s)...))
    end
end

# PointCloud

export PointCloud

point_id = 0

struct PointCloud{P,G,PA<:AbstractVector{P},GA<:AbstractVector{G}} <: AbstractVector{Coordinate{P,G}}
    id::Int
    dom::PA
    cod::GA
    PointCloud(id::Int,p::PA,g::GA) where {P,G,PA<:AbstractVector{P},GA<:AbstractVector{G}} = new{P,G,PA,GA}(id,p,g)
end

PointCloud(id::Int,dom) = PointCloud(id,dom,Global{1}(InducedMetric()))
PointCloud(dom) = PointCloud(dom,Global{1}(InducedMetric()))

points(m::PointCloud) = m.dom
metrictensor(m::PointCloud) = m.cod
pointtype(m::PointCloud{P}) where P = P
pointtype(m::Type{<:PointCloud{P}}) where P = P
metrictype(::PointCloud{P,G} where P) where G = G
metrictype(::Type{<:PointCloud{P,G} where P}) where G = G

Base.size(m::PointCloud) = size(m.dom)
Base.resize!(m::PointCloud,i) = (resize!(points(m),i),resize!(metrictensor(m),i))
Base.broadcast(f,t::PointCloud) = PointCloud(f.(points(t)),f.(metrictensor(t)))

@pure Grassmann.Manifold(m::PointCloud) = Manifold(points(m))
@pure LinearAlgebra.rank(m::PointCloud) = rank(points(m))
@pure Grassmann.grade(m::PointCloud) = grade(points(m))
@pure Grassmann.antigrade(m::PointCloud) = antigrade(points(m))
@pure Grassmann.mdims(m::PointCloud) = mdims(points(m))

const point_cache = (Vector{Chain{V,G,T,X}} where {V,G,T,X})[]
const point_metric_cache = (AbstractVector{T} where T)[]

PointCloud(m::Int) = PointCloud(m,point_cache[m],point_metric_cache[m])
function PointCloud(p::P,g::G) where {P<:AbstractVector,G<:AbstractVector}
    push!(point_cache,p)
    push!(point_metric_cache,g)
    PointCloud(length(point_cache),p,g)
end
function clearpointcache!()
    for P ∈ 1:length(point_cache)
        deletebundle!(P)
    end
end
bundle(m::PointCloud) = m.id
deletebundle!(m::PointCloud) = deletepointcloud!(bundle(m))
function deletepointcloud!(P::Int)
    point_cache[P] = [Chain{ℝ^0,0,Int}(Values(0))]
    point_metric_cache[P] = [Chain{ℝ^0,0,Int}(Values(0))]
    nothing
end

Base.firstindex(m::PointCloud) = 1
Base.lastindex(m::PointCloud) = length(points(m))
Base.length(m::PointCloud) = length(points(m))
Base.resize!(m::PointCloud,n::Int) = (resize!(points(m),n),resize!(metrictensor(m),n))

# GlobalFiber

abstract type GlobalFiber{E,N} <: AbstractArray{E,N} end
Base.@pure isfiberbundle(::GlobalFiber) = true
Base.@pure isfiberbundle(::Any) = false

base(t::GlobalFiber) = t.dom
fiber(t::GlobalFiber) = t.cod
base(t::Array) = ProductSpace(Values(axes(t)))
fiber(t::Array) = t
basetype(::Array{T}) where T = T
fibertype(::Array{T}) where T = T

topology(m::GlobalFiber) = topology(immersion(m))
vertices(m::GlobalFiber) = vertices(immersion(m))
iscover(m::GlobalFiber) = iscover(immersion(m))
imagepoints(m::GlobalFiber) = iscover(m) ? points(m) : points(m)[vertices(m)]

unitdomain(t::GlobalFiber) = base(t)*inv(base(t)[end])
arcdomain(t::GlobalFiber) = unitdomain(t)*arclength(codomain(t))
graph(t::GlobalFiber) = graph.(t)

Base.size(m::GlobalFiber) = size(m.cod)
Base.resize!(m::GlobalFiber,i) = (resize!(domain(m),i),resize!(codomain(m),i))

# AbstractFrameBundle

export AbstractFrameBundle, GridFrameBundle, SimplexFrameBundle, FacetFrameBundle
export IntervalRange, AlignedRegion, AlignedSpace

abstract type AbstractFrameBundle{M,N} <: GlobalFiber{M,N} end

Base.size(m::AbstractFrameBundle) = size(points(m))

@pure Grassmann.Manifold(m::AbstractFrameBundle) = Manifold(points(m))
@pure LinearAlgebra.rank(m::AbstractFrameBundle) = rank(points(m))
@pure Grassmann.grade(m::AbstractFrameBundle) = grade(points(m))
@pure Grassmann.antigrade(m::AbstractFrameBundle) = antigrade(points(m))
@pure Grassmann.mdims(m::AbstractFrameBundle) = mdims(points(m))

# GridFrameBundle

grid_id = 0

struct GridFrameBundle{P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} <: AbstractFrameBundle{Coordinate{P,G},N}
    id::Int
    dom::PA
    cod::GA
    GridFrameBundle(id::Int,p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{P,G,N,PA,GA}(id,p,g)
    GridFrameBundle(p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{P,G,N,PA,GA}((global grid_id+=1),p,g)
end

const IntervalRange{P<:Real,G,PA<:AbstractRange,GA} = GridFrameBundle{P,G,1,PA,GA}
const AlignedRegion{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA<:Global} = GridFrameBundle{P,G,N,PA,GA}
const AlignedSpace{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA} = GridFrameBundle{P,G,N,PA,GA}

GridFrameBundle(id::Int,p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(id,p,Global{N}(InducedMetric()))
GridFrameBundle(p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(p,Global{N}(InducedMetric()))
GridFrameBundle(dom::GridFrameBundle,fun) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Array) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Function) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::AbstractArray,fun::Function) = GridFrameBundle(dom, fun.(dom))

points(m::GridFrameBundle) = m.dom
metrictensor(m::GridFrameBundle) = m.cod
coordinates(t::GridFrameBundle) = t
pointtype(m::GridFrameBundle) = basetype(m)
pointtype(m::Type{<:GridFrameBundle}) = basetype(m)
metrictype(m::GridFrameBundle) = fibertype(m)
metrictype(m::Type{<:GridFrameBundle}) = fibertype(m)
basetype(::GridFrameBundle{B}) where B = B
basetype(::Type{<:GridFrameBundle{B}}) where B = B
fibertype(::GridFrameBundle{B,F} where B) where F = F
fibertype(::Type{<:GridFrameBundle{B,F} where B}) where F = F

Base.resize!(m::GridFrameBundle,i) = (resize!(points(m),i),resize!(metrictensor(m),i))
Base.broadcast(f,t::GridFrameBundle) = GridFrameBundle(f.(points(t)),f.(metrictensor(t)))

# SimplexFrameBundle

struct SimplexFrameBundle{P,G,PA<:AbstractVector{P},GA<:AbstractVector{G},TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,G},1}
    p::PointCloud{P,G,PA,GA}
    t::TA
    SimplexFrameBundle(p::PointCloud{P,G,PA,GA},t::T) where {P,G,PA,GA,T} = new{P,G,PA,GA,T}(p,t)
end

SimplexFrameBundle(id::Int,p,t,g) = SimplexFrameBundle(PointCloud(id,p,g),t)
SimplexFrameBundle(id::Int,p,t) = SimplexFrameBundle(PointCloud(id,p),t)
#SimplexFrameBundle(p::AbstractVector,t) = SimplexFrameBundle(PointCloud(p),t)

(p::PointCloud)(t::ImmersedTopology) = length(p)≠length(t) ? SimplexFrameBundle(p,t) : FacetFrameBundle(p,t)
PointCloud(m::SimplexFrameBundle) = m.p
ImmersedTopology(m::SimplexFrameBundle) = m.t
coordinates(t::SimplexFrameBundle) = t
points(m::SimplexFrameBundle) = points(PointCloud(m))
metrictensor(m::SimplexFrameBundle) = metrictensor(PointCloud(m))
pointtype(m::SimplexFrameBundle) = basetype(m)
pointtype(m::Type{<:SimplexFrameBundle}) = basetype(m)
metrictype(m::SimplexFrameBundle) = fibertype(m)
metrictype(m::Type{<:SimplexFrameBundle}) = fibertype(m)
basetype(::SimplexFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:SimplexFrameBundle{P}}) where P = pointtype(P)
fibertype(::SimplexFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:SimplexFrameBundle{P}}) where P = metrictype(P)
Base.size(m::SimplexFrameBundle) = size(vertices(m))

Base.broadcast(f,t::SimplexFrameBundle) = SimplexFrameBundle(f.(PointCloud(t)),ImmersedTopology(t))

struct FacetFrameBundle{P,G,PA,GA,TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,G},1}
    id::Int
    p::PointCloud{P,G,PA,GA}
    t::TA
    FacetFrameBundle(id::Int,p::PointCloud{P,G,PA,GA},t::T) where {P,G,PA,GA,T} = new{P,G,PA,GA,T}(id,p,t)
end

#FacetFrameBundle(id::Int,p,t,g) = FacetFrameBundle(PointCloud(id,p,g),t)
#FacetFrameBundle(id::Int,p,t) = FacetFrameBundle(PointCloud(id,p),t)
#FacetFrameBundle(p::AbstractVector,t) = FacetFrameBundle(PointCloud(p),t)

PointCloud(m::FacetFrameBundle) = m.p
ImmersedTopology(m::FacetFrameBundle) = m.t
coordinates(t::FacetFrameBundle) = t
points(m::FacetFrameBundle) = points(PointCloud(m))
metrictensor(m::FacetFrameBundle) = metrictensor(PointCloud(m))
pointtype(m::FacetFrameBundle) = basetype(m)
pointtype(m::Type{<:FacetFrameBundle}) = basetype(m)
metrictype(m::FacetFrameBundle) = fibertype(m)
metrictype(m::Type{<:FacetFrameBundle}) = fibertype(m)
basetype(::FacetFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:FacetFrameBundle{P}}) where P = pointtype(P)
fibertype(::FacetFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:FacetFrameBundle{P}}) where P = metrictype(P)

Base.broadcast(f,t::FacetFrameBundle) = FacetFrameBundle(0,f.(PointCloud(t)),ImmersedTopology(t))

function SimplexFrameBundle(p::P,t,g::G) where {P<:AbstractVector,G<:AbstractVector}
    SimplexFrameBundle(PointCloud(p,g),t)
end
function SimplexFrameBundle(m::FacetFrameBundle)
    SimplexFrameBundle(PointCloud(m.id,point_cache[m.id],point_metric_cache[m.id]),ImmersedTopology(m))
end
function FacetFrameBundle(m::SimplexFrameBundle)
    et = topology(ImmersedTopology(m))
    FacetFrameBundle(m.id,PointCloud(0,barycenter.(m[et]),barycenter.(getindex.(Ref(metrictensor(m)),et))),ImmersedTopology(m))
end

bundle(m::SimplexFrameBundle) = m.id
bundle(m::FacetFrameBundle) = m.id
deletebundle!(m::SimplexFrameBundle) = deletepointcloud!(bundle(m))
deletebundle!(m::FacetFrameBundle) = deletepointcloud!(bundle(m))
@pure isbundle(::AbstractFrameBundle) = true
@pure isbundle(t) = false
#@pure ispoints(t::Submanifold{V}) where V = isbundle(V) && rank(V) == 1 && !isbundle(Manifold(V))
#@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
#@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
#@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

Base.firstindex(m::SimplexFrameBundle) = 1
Base.lastindex(m::SimplexFrameBundle) = length(vertices(m))
Base.length(m::SimplexFrameBundle) = length(vertices(m))
#Base.resize!(m::SimplexFrameBundle,n::Int) = resize!(value(m),n)

# GlobalSection

struct GlobalSection{B,F,N,BA<:AbstractFrameBundle{B,N},FA<:AbstractFrameBundle{F,N}} <: GlobalFiber{LocalSection{B,F},N}
    dom::BA
    cod::FA
    GlobalSection(dom::BA,cod::FA) where {B,F,N,BA<:AbstractFrameBundle{B,N},FA<:AbstractFrameBundle{F,N}} = new{B,F,N,BA,FA}(dom,cod)
end

# TensorField

struct TensorField{B,F,N,M<:AbstractFrameBundle{B,N}} <: GlobalFiber{LocalTensor{B,F},N}
    dom::M
    cod::Array{F,N}
    function TensorField(dom::M,cod::Array{F,N}) where {B,F,N,M<:AbstractFrameBundle{B,N}}
        new{B,F,N,M}(dom,cod)
    end
end

function TensorField(id::Int,dom::PA,cod::Array{F,N},met::GA=Global{N}(InducedMetric())) where {N,P,F,PA<:AbstractArray{P,N},GA<:AbstractArray}
    TensorField(GridFrameBundle(id,dom,met),cod)
end
function TensorField(id::Int,dom::P,cod::Vector{F},met::G=Global{N}(InducedMetric())) where {F,P<:PointCloud,G<:AbstractVector}
    TensorField(SimplexFrameBundle(id,dom,met),cod)
end
TensorField(id::Int,dom,cod::Array,met::GlobalFiber) = TensorField(id,dom,cod,fiber(met))
TensorField(dom::AbstractFrameBundle,cod::AbstractFrameBundle) = TensorField(dom,points(cod))
TensorField(dom::AbstractArray{B,N} where B,cod::Array{F,N} where F,met::AbstractArray=Global{N}(InducedMetric())) where N = TensorField((global grid_id+=1),dom,cod,fiber(met))
TensorField(dom::ChainBundle,cod::Vector,met::AbstractVector=Global{1}(InducedMetric())) = TensorField((global grid_id+=1),dom,cod,met)

#const ParametricMesh{B,F,P<:AbstractVector{<:Chain}} = TensorField{B,F,1,P}
const ScalarMap{B,F<:AbstractReal,P<:SimplexFrameBundle} = TensorField{B,F,1,P}
#const ElementFunction{B,F<:AbstractReal,P<:AbstractVector} = TensorField{B,F,1,P}
const IntervalMap{B,F,P<:Interval} = TensorField{B,F,1,P}
const RectangleMap{B,F,P<:RealSpace{2}} = TensorField{B,F,2,P}
const HyperrectangleMap{B,F,P<:RealSpace{3}} = TensorField{B,F,3,P}
const ParametricMap{B,F,N,P<:RealSpace} = TensorField{B,F,N,P}
const RealFunction{B,F<:AbstractReal,P<:Interval} = TensorField{B,F,1,P}
const PlaneCurve{B,F<:Chain{V,G,Q,2} where {V,G,Q},P<:Interval} = TensorField{B,F,1,P}
const SpaceCurve{B,F<:Chain{V,G,Q,3} where {V,G,Q},P<:Interval} = TensorField{B,F,1,P}
const AbstractCurve{B,F<:Chain,P<:Interval} = TensorField{B,F,1,P}
const SurfaceGrid{B,F<:AbstractReal,P<:RealSpace{2}} = TensorField{B,F,2,P}
const VolumeGrid{B,F<:AbstractReal,P<:RealSpace{3}} = TensorField{B,F,3,P}
const ScalarGrid{B,F<:AbstractReal,N,P<:RealSpace{N}} = TensorField{B,F,N,P}
const GlobalFrame{B<:LocalFiber{P,<:TensorNested} where P,N} = GlobalSection{B,N}
const DiagonalField{B,F<:DiagonalOperator,N,P} = TensorField{B,F,N,P}
const EndomorphismField{B,F<:Endomorphism,N,P} = TensorField{B,F,N,P}
const OutermorphismField{B,F<:Outermorphism,N,P} = TensorField{B,F,N,P}
const CliffordField{B,F<:Multivector,N,P} = TensorField{B,F,N,P}
const QuaternionField{B,F<:Quaternion,N,P} = TensorField{B,F,N,P}
const ComplexMap{B,F<:AbstractComplex,N,P} = TensorField{B,F,N,P}
const PhasorField{B,F<:Phasor,N,P} = TensorField{B,F,N,P}
const SpinorField{B,F<:AbstractSpinor,N,P} = TensorField{B,F,N,P}
const GradedField{G,B,F<:Chain{V,G} where V,N,P} = TensorField{B,F,N,P}
const ScalarField{B,F<:AbstractReal,N,P} = TensorField{B,F,N,P}
const VectorField = GradedField{1}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

struct Connection{T}
    ∇::T
    Connection(∇::T) where T = new{T}(∇)
end

(∇::Connection)(x::VectorField) = CovariantDerivative(∇.∇⋅x,x)
(∇::Connection)(x::VectorField,y::VectorField) = (x⋅gradient(y))+((∇.∇⋅x)⋅y)

struct CovariantDerivative{T,X}
    ∇x::T
    x::X
    CovariantDerivative(∇x::T,x::X) where {T,X} = new{T,X}(∇x,x)
end

CovariantDerivative(∇::Connection,x) = ∇(x)
(∇x::CovariantDerivative)(y::VectorField) = (∇x.x⋅gradient(y))+(∇x.∇x⋅y)

for bundle ∈ (:TensorField,:GlobalSection)
    @eval begin
        $bundle(dom,fun::BitArray) = $bundle(dom, Float64.(fun))
        $bundle(dom,fun::$bundle) = $bundle(dom, fiber(fun))
        $bundle(dom::$bundle,fun) = $bundle(base(dom), fun)
        $bundle(dom::$bundle,fun::Array) = $bundle(base(dom), fun)
        $bundle(dom::$bundle,fun::Function) = $bundle(base(dom), fun)
        $bundle(dom::AbstractArray,fun::AbstractRange) = $bundle(dom, collect(fun))
        $bundle(dom::AbstractArray,fun::RealRegion) = $bundle(dom, collect(fun))
        $bundle(dom::AbstractArray,fun::Function) = $bundle(dom, fun.(dom))
        $bundle(dom::ChainBundle,fun::Function) = $bundle(dom, fun.(value(points(dom))))
        basetype(::$bundle{B}) where B = B
        basetype(::Type{<:$bundle{B}}) where B = B
        Base.broadcast(f,t::$bundle) = $bundle(domain(t), f.(codomain(t)))
    end
end

←(F,B) = B → F
const → = TensorField
base(t::GlobalSection) = t.dom
fiber(t::GlobalSection) = t.cod
base(t::TensorField) = t.dom
fiber(t::TensorField) = t.cod
coordinates(t::TensorField) = base(t)
points(t::TensorField) = points(base(t))
metrictensor(t::TensorField) = metrictensor(base(t))
immersion(t::TensorField) = immersion(base(t))
pointtype(m::TensorField) = pointtype(base(m))
pointtype(m::Type{<:TensorField}) = pointtype(basetype(m))
metrictype(m::TensorField) = metrictype(base(m))
metrictype(m::Type{<:TensorField}) = metrictype(basetype(m))
fibertype(::GlobalSection{B,F} where B) where F = F
fibertype(::Type{<:GlobalSection{B,F} where B}) where F = F
fibertype(::TensorField{B,F} where B) where F = F
fibertype(::Type{<:TensorField{B,F} where B}) where F = F

@pure Base.eltype(::Type{<:GridFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
Base.getindex(m::GridFrameBundle,i::Vararg{Int}) = Coordinate(getindex(points(m),i...), getindex(metrictensor(m),i...))
Base.setindex!(m::GridFrameBundle{P},s::P,i::Vararg{Int}) where P = setindex!(points(m),s,i...)
Base.setindex!(m::GridFrameBundle{P,G} where P,s::G,i::Vararg{Int}) where G = setindex!(metrictensor(m),s,i...)
function Base.setindex!(m::GridFrameBundle,s::Coordinate,i::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metrictensor(m),metrictensor(s),i...)
    return s
end

(m::SimplexFrameBundle)(i::ImmersedTopology) = SimplexFrameBundle(bundle(m),points(m),i,metrictensor(m))
Base.getindex(m::SimplexFrameBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(points(m)[value(i)])
Base.getindex(m::SimplexFrameBundle,i::Values{N,Int}) where N = points(m)[value(i)]
getindex(m::AbstractVector,i::ImmersedTopology) = getindex.(Ref(m),topology(i))
getindex(m::AbstractVector,i::SimplexFrameBundle) = m[immersion(i)]
getindex(m::SimplexFrameBundle,i::ImmersedTopology) = points(m)[i]
getindex(m::SimplexFrameBundle,i::SimplexFrameBundle) = points(m)[immersion(i)]

getimage(m,i) = iscover(m) ? i : getindex(vertices(m),i)

@pure Base.eltype(::Type{<:PointCloud{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::PointCloud,i::Int)
    Coordinate(getindex(points(m),i), getindex(metrictensor(m),i))
end
Base.setindex!(m::PointCloud{P},s::P,i::Int) where P = setindex!(points(m),s,i)
Base.setindex!(m::PointCloud{P,G} where P,s::G,i::Int) where G = setindex!(metrictensor(m),s,i)
function Base.setindex!(m::PointCloud,s::Coordinate,i::Int)
    setindex!(points(m),point(s),i)
    setindex!(metrictensor(m),metrictensor(s),i)
    return s
end

@pure Base.eltype(::Type{<:SimplexFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::SimplexFrameBundle,i::Int)
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind), getindex(metrictensor(m),ind))
end
Base.setindex!(m::SimplexFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::SimplexFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metrictensor(m),s,getimage(m,i))
function Base.setindex!(m::SimplexFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metrictensor(m),metrictensor(s),ind)
    return s
end

@pure Base.eltype(::Type{<:FacetFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::FacetFrameBundle,i::Int)
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind), getindex(metrictensor(m),ind))
end
Base.setindex!(m::FacetFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::FacetFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metrictensor(m),s,getimage(m,i))
function Base.setindex!(m::FacetFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metrictensor(m),metrictensor(s),ind)
    return s
end

@pure Base.eltype(::Type{<:TensorField{B,F}}) where {B,F} = LocalTensor{B,F}
Base.getindex(m::TensorField,i::Vararg{Int}) = LocalTensor(getindex(domain(m),i...), getindex(codomain(m),i...))
#Base.setindex!(m::TensorField{B,F,1,<:Interval},s::LocalTensor,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::TensorField,s::LocalTensor,i::Vararg{Int})
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

@pure Base.eltype(::Type{<:GlobalSection{B,F}}) where {B,F} = LocalSection{B,F}
Base.getindex(m::GlobalSection,i::Vararg{Int}) = LocalSection(getindex(domain(m),i...), getindex(codomain(m),i...))
#Base.setindex!(m::GlobalSection{B,F,1,<:Interval},s::LocalSection,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
#Base.setindex!(m::GlobalSection{B,F,N,<:RealRegion{V,T,N,<:AbstractRange} where {V,T}},s::LocalSection,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::GlobalSection{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::GlobalSection,s::LocalSection,i::Vararg{Int})
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

Base.BroadcastStyle(::Type{<:GridFrameBundle{P,G,N,PA,GA}}) where {P,G,N,PA,GA} = Broadcast.ArrayStyle{GridFrameBundle{P,G,N,PA,GA}}()
Base.BroadcastStyle(::Type{<:TensorField{B,F,N,P}}) where {B,F,N,P} = Broadcast.ArrayStyle{TensorField{B,F,N,P}}()
Base.BroadcastStyle(::Type{<:GlobalSection{B,F,N,BA,FA}}) where {B,F,N,BA,FA} = Broadcast.ArrayStyle{TensorField{B,F,N,BA,FA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{P,G,N,PA,GA}}}, ::Type{ElType}) where {P,G,N,PA,GA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{P,G,N,PA,GA}}}, ::Type{ElType}) where {P,G,N,PA,GA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{ElType,N}, axes(bc)), metrictensor(t))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,F,N,P}}}, ::Type{ElType}) where {B,F,N,P,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    TensorField(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlobalSection{B,F,N,BA,FA}}}, ::Type{ElType}) where {B,F,N,BA,FA,ElType}
    # Scan the inputs for the TensorField:
    t = find_gs(bc)
    # Use the domain field of t to create the output
    GlobalSection(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end

"`A = find_gf(As)` returns the first GridFrameBundle among the arguments."
find_gf(bc::Base.Broadcast.Broadcasted) = find_gf(bc.args)
find_gf(bc::Base.Broadcast.Extruded) = find_gf(bc.x)
find_gf(args::Tuple) = find_gf(find_gf(args[1]), Base.tail(args))
find_gf(x) = x
find_gf(::Tuple{}) = nothing
find_gf(a::GridFrameBundle, rest) = a
find_gf(::Any, rest) = find_gf(rest)

"`A = find_tf(As)` returns the first TensorField among the arguments."
find_tf(bc::Base.Broadcast.Broadcasted) = find_tf(bc.args)
find_tf(bc::Base.Broadcast.Extruded) = find_tf(bc.x)
find_tf(args::Tuple) = find_tf(find_tf(args[1]), Base.tail(args))
find_tf(x) = x
find_tf(::Tuple{}) = nothing
find_tf(a::TensorField, rest) = a
find_tf(::Any, rest) = find_tf(rest)

"`A = find_gs(As)` returns the first GlobalSection among the arguments."
find_gs(bc::Base.Broadcast.Broadcasted) = find_gs(bc.args)
find_gs(bc::Base.Broadcast.Extruded) = find_gs(bc.x)
find_gs(args::Tuple) = find_gs(find_gs(args[1]), Base.tail(args))
find_gs(x) = x
find_gs(::Tuple{}) = nothing
find_gs(a::GlobalSection, rest) = a
find_gs(::Any, rest) = find_gs(rest)

linterp(x,x1,x2,f1,f2) = f1 + (f2-f1)*(x-x1)/(x2-x1)
function bilinterp(x,y,x1,x2,y1,y2,f11,f21,f12,f22)
    f1 = linterp(x,x1,x2,f11,f21)
    f2 = linterp(x,x1,x2,f12,f22)
    linterp(y,y1,y2,f1,f2)
end
function trilinterp(x,y,z,x1,x2,y1,y2,z1,z2,f111,f211,f121,f221,f112,f212,f122,f222)
    f1 = bilinterp(x,y,x1,x2,y1,y2,f111,f211,f121,f221)
    f2 = bilinterp(x,y,x1,x2,y1,y2,f112,f212,f122,f222)
    linterp(z,z1,z2,f1,f2)
end

(m::IntervalMap)(s::LocalTensor) = LocalTensor(base(s), m(fiber(s)))
function (m::IntervalMap)(t)
    i = searchsortedfirst(domain(m),t[1])-1
    linterp(t[1],m.dom[i],m.dom[i+1],m.cod[i],m.cod[i+1])
end
function (m::IntervalMap)(t::Vector,d=diff(m.cod)./diff(m.dom))
    [parametric(i,m,d) for i ∈ t]
end
function parametric(t,m,d=diff(codomain(m))./diff(domain(m)))
    i = searchsortedfirst(domain(m),t)-1
    codomain(m)[i]+(t-domain(m)[i])*d[i]
end

function (m::TensorField{B,F,N,<:SimplexFrameBundle} where {B,F,N})(t)
    i = immersion(m)[findfirst(t,domain(m))]
    Chain(codomain(m)[i])⋅(Chain(points(domain(m))[i])/t)
end

(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(x,y) = m(Chain(x,y))
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(s::LocalSection) = LocalTensor(base(s), m(fiber(s)))
function (m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(t::Chain)
    x,y = points(m).v[1],points(m).v[2]
    i,j = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1
    #f1 = linterp(t[1],x[i],x[i+1],m.cod[i,j],m.cod[i+1,j])
    #f2 = linterp(t[1],x[i],x[i+1],m.cod[i,j+1],m.cod[i+1,j+1])
    #linterp(t[2],y[j],y[j+1],f1,f2)
    bilinterp(t[1],t[2],x[i],x[i+1],y[j],y[j+1],
        m.cod[i,j],m.cod[i+1,j],m.cod[i,j+1],m.cod[i+1,j+1])
end

(m::TensorField{B,F,N,<:RealSpace{3}} where {B,F,N})(x,y,z) = m(Chain(x,y,z))
(m::TensorField{B,F,N,<:RealSpace{3}} where {B,F,N})(s::LocalTensor) = LocalTensor(base(s), m(fiber(s)))
function (m::TensorField{B,F,N,<:RealSpace{3}} where {B,F,N})(t::Chain)
    x,y,z = points(m).v[1],points(m).v[2],points(m).v[3]
    i,j,k = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1,searchsortedfirst(z,t[3])-1
    #f1 = linterp(t[1],x[i],x[i+1],m.cod[i,j,k],m.cod[i+1,j,k])
    #f2 = linterp(t[1],x[i],x[i+1],m.cod[i,j+1,k],m.cod[i+1,j+1,k])
    #g1 = linterp(t[2],y[j],y[j+1],f1,f2)
    #f3 = linterp(t[1],x[i],x[i+1],m.cod[i,j,k+1],m.cod[i+1,j,k+1])
    #f4 = linterp(t[1],x[i],x[i+1],m.cod[i,j+1,k+1],m.cod[i+1,j+1,k+1])
    #g2 = linterp(t[2],y[j],y[j+1],f3,f4)
    #linterp(t[3],z[k],z[k+1],g1,g2)
    trilinterp(t[1],t[2],t[3],x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],
        m.cod[i,j,k],m.cod[i+1,j,k],m.cod[i,j+1,k],m.cod[i+1,j+1,k],
        m.cod[i,j,k+1],m.cod[i+1,j,k+1],m.cod[i,j+1,k+1],m.cod[i+1,j+1,k+1])
end

valmat(t::Values{N,<:Vector},s=size(t[1])) where N = [Values((t[q][i] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1])]
valmat(t::Values{N,<:Matrix},s=size(t[1])) where N = [Values((t[q][i,j] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2])]
valmat(t::Values{N,<:Array{T,3} where T},s=size(t[1])) where N = [Values((t[q][i,j,k] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2]), k ∈ OneTo(s[3])]

fromany(t::Chain{V,G,Any}) where {V,G} = Chain{V,G}(value(t)...)
fromany(t::Chain) = t

TensorField(t::Chain{V,G}) where {V,G} = TensorField(base(t[1]), Chain{V,G}.(valmat(fiber.(value(t)))))
Grassmann.Chain(t::TensorField{B,<:Union{Real,Complex}} where B) = Chain{Submanifold(ndims(t)),0}(t)
function Grassmann.Chain(t::TensorField{B,<:Chain{V,G}} where B) where {V,G}
    Chain{V,G}((TensorField(base(t), getindex.(fiber(t),j)) for j ∈ 1:binomial(mdims(V),G))...)
end
Base.:^(t::TensorField,n::Int) = TensorField(domain(t), .^(codomain(t),n,ref(metrictensor(base(t)))))
for op ∈ (:+,:-,:&,:∧,:∨)
    let bop = op ∈ (:∧,:∨) ? :(Grassmann.$op) : :(Base.$op)
        @eval begin
            $bop(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(a), $op.(codomain(a),codomain(b)))
            $bop(a::TensorField,b::Number) = TensorField(domain(a), $op.(codomain(a),Ref(b)))
            $bop(a::Number,b::TensorField) = TensorField(domain(b), $op.(Ref(a),codomain(b)))
        end
    end
end
@inline Base.:<<(a::GlobalFiber,b::GlobalFiber) = contraction(b,~a)
@inline Base.:>>(a::GlobalFiber,b::GlobalFiber) = contraction(~a,b)
@inline Base.:<(a::GlobalFiber,b::GlobalFiber) = contraction(b,a)
for type ∈ (:TensorField,)
    for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:contraction,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
        let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop.(fiber(a),fiber(b),ref(metrictensor(base(a)))))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op.(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op.(fiber(a),b,$((op≠:^ ? () : (:(ref(metrictensor(base(a)))),))...)))
        end end
    end
end
for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad)
    @eval Base.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t)))
end
for fun ∈ (:exp,:log,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2)#:inv
    @eval Base.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t),ref(metrictensor(t))))
end
for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:complementleft)
    @eval Grassmann.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t)))
end
for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric)
    @eval Grassmann.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t),ref(metrictensor(t))))
end
for fun ∈ (:sum,:prod)
    @eval Base.$fun(t::TensorField) = LocalTensor(domain(t)[end], $fun(codomain(t)))
end
for fun ∈ (:cumsum,:cumprod)
    @eval function Base.$fun(t::TensorField)
         out = $fun(codomain(t))
         pushfirst!(out,zero(eltype(out)))
         TensorField(domain(t), out)
    end
end

Grassmann.signbit(::TensorField) = false
Base.inv(t::TensorField) = TensorField(codomain(t), domain(t))
Base.diff(t::TensorField) = TensorField(diff(domain(t)), diff(codomain(t)))
absvalue(t::TensorField) = TensorField(domain(t), value.(abs.(codomain(t))))
LinearAlgebra.det(t::TensorField) = TensorField(domain(t), det.(codomain(t)))
LinearAlgebra.norm(t::TensorField) = TensorField(domain(t), norm.(codomain(t)))
(V::Submanifold)(t::TensorField) = TensorField(domain(t), V.(codomain(t)))
(::Type{T})(t::TensorField) where T<:Real = TensorField(domain(t), T.(codomain(t)))
(::Type{Complex})(t::TensorField) = TensorField(domain(t), Complex.(codomain(t)))
(::Type{Complex{T}})(t::TensorField) where T = TensorField(domain(t), Complex{T}.(codomain(t)))
Grassmann.Phasor(s::TensorField) = TensorField(domain(s), Phasor(codomain(s)))
Grassmann.Couple(s::TensorField) = TensorField(domain(s), Couple(codomain(s)))

checkdomain(a::GlobalFiber,b::GlobalFiber) = domain(a)≠domain(b) ? error("GlobalFiber base not equal") : true

include("diffgeo.jl")
include("constants.jl")
include("element.jl")

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        export linegraph, linegraph!
        funsym(sym) = String(sym)[end] == '!' ? sym : Symbol(sym,:!)
        for lines ∈ (:lines,:lines!,:linesegments,:linesegments!)
            @eval begin
                Makie.$lines(t::ScalarMap;args...) = Makie.$lines(getindex.(domain(t),2),codomain(t);args...)
                Makie.$lines(t::SpaceCurve;args...) = Makie.$lines(codomain(t);color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::PlaneCurve;args...) = Makie.$lines(codomain(t);color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::RealFunction;args...) = Makie.$lines(Real.(points(t)),Real.(codomain(t));color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::ComplexMap{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F} = Makie.$lines(realvalue.(codomain(t)),imagvalue.(codomain(t));color=Real.(codomain(speed(t))),args...)
            end
        end
        #Makie.lines(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
        #Makie.lines!(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
        function linegraph(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
            x,y = Real.(domain(t)),value.(codomain(t))
            display(Makie.lines(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),G)
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function linegraph!(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
            x,y = Real.(domain(t)),value.(codomain(t))
            display(Makie.lines!(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),G)
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        Makie.volume(t::VolumeGrid;args...) = Makie.volume(domain(t).v...,Real.(codomain(t));args...)
        Makie.volume!(t::VolumeGrid;args...) = Makie.volume!(domain(t).v...,Real.(codomain(t));args...)
        Makie.volumeslices(t::VolumeGrid;args...) = Makie.volumeslices(domain(t).v...,Real.(codomain(t));args...)
        for fun ∈ (:surface,:surface!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(points(t).v...,Real.(codomain(t));color=Real.(abs.(codomain(gradient_fast(Real(t))))),args...)
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F<:AbstractComplex};args...) = Makie.$fun(points(t).v...,Real.(radius.(codomain(t)));color=Real.(angle.(codomain(t))),colormap=:twilight,args...)
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G;args...) where {B,F<:Chain}
                    x,y = points(t),value.(codomain(t))
                    yi = Real.(getindex.(y,1))
                    display(Makie.$fun(x.v...,yi;color=Real.(abs.(codomain(gradient_fast(x→yi)))),args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),G)
                        yi = Real.(getindex.(y,i))
                        Makie.$(funsym(fun))(x.v...,yi;color=Real.(abs.(codomain(gradient_fast(x→yi)))),args...)
                    end
                end
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = Makie.$fun(domain(t).v...,Real.(radius.(codomain(t)));args...)
            end
        end
        for fun ∈ (:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = Makie.$fun(domain(t).v...,Real.(angle.(codomain(t)));colormap=:twilight,args...)
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(points(t).v...,Real.(codomain(t));args...)
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G;args...) where {B,F}
                    x,y = points(t),value.(codomain(t))
                    display(Makie.$fun(x.v...,Real.(getindex.(y,1));args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),G)
                        Makie.$(funsym(fun))(x.v...,Real.(getindex.(y,i));args...)
                    end
                end
            end
        end
        for fun ∈ (:streamplot,:streamplot!)
            @eval begin
                Makie.$fun(f::Function,t::Rectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(f::Function,t::Hyperrectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(m::ScalarField{<:Coordinate{<:Chain},<:AbstractReal,N,<:RealSpace} where N;args...) = Makie.$fun(gradient_fast(m);args...)
                Makie.$fun(m::ScalarMap,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
                Makie.$fun(m::VectorField{R,F,1,<:SimplexFrameBundle} where {R,F},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
                Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealSpace} where {F,N};args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),points(m).v...;args...)
            end
        end
        for fun ∈ (:arrows,:arrows!)
            @eval begin
                Makie.$fun(t::ScalarField{<:Coordinate{<:Chain},F,2,<:RealSpace{2}} where F;args...) = Makie.$fun(Makie.Point.(fiber(graph(Real(t))))[:],Makie.Point.(fiber(normal(Real(t))))[:];args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},<:Chain{V,G,T,2} where {V,G,T},2,<:AlignedRegion{2}};args...) = Makie.$fun(domain(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,N,<:GridFrameBundle} where {F,N};args...) = Makie.$fun(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::ScalarMap;args...) = Makie.$fun(points(points(t)),Real.(codomain(t));args...)
            end
        end
        Makie.mesh(t::ScalarMap;args...) = Makie.mesh(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh!(t::ScalarMap;args...) = Makie.mesh!(domain(t);color=Real.(codomain(t)),args...)
        #Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=Real.(codomain(t)),args...)
        #Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=Real.(codomain(t)),args...)
        Makie.convert_arguments(P::Makie.PointBased, a::SimplexFrameBundle) = Makie.convert_arguments(P, points(a))
        Makie.convert_single_argument(a::LocalFiber) = convert_arguments(P,Point(a))
        Makie.arrows(p::SimplexFrameBundle,v;args...) = Makie.arrows(GeometryBasics.Point.(↓(V).(points(p))),GeometryBasics.Point.(value(v));args...)
        Makie.arrows!(p::SimplexFrameBundle,v;args...) = Makie.arrows!(GeometryBasics.Point.(↓(V).(points(p))),GeometryBasics.Point.(value(v));args...)
        Makie.scatter(p::SimplexFrameBundle,x;args...) = Makie.scatter(submesh(p)[:,1],x;args...)
        Makie.scatter!(p::SimplexFrameBundle,x;args...) = Makie.scatter!(submesh(p)[:,1],x;args...)
        Makie.scatter(p::SimplexFrameBundle;args...) = Makie.scatter(submesh(p);args...)
        Makie.scatter!(p::SimplexFrameBundle;args...) = Makie.scatter!(submesh(p);args...)
        Makie.lines(p::SimplexFrameBundle;args...) = Makie.lines(points(p);args...)
        Makie.lines!(p::SimplexFrameBundle;args...) = Makie.lines!(points(p);args...)
        #Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(GeometryBasics.Point.(p);args...)
        #Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(GeometryBasics.Point.(p);args...)
        #Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
        #Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
        #Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
        #Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)
        Makie.linesegments(e::SimplexFrameBundle;args...) = (p=points(e); Makie.linesegments(Grassmann.pointpair.(e[ImmersedTopology(e)],↓(Manifold(p)));args...))
        Makie.linesegments!(e::SimplexFrameBundle;args...) = (p=points(e); Makie.linesegments!(Grassmann.pointpair.(e[ImmersedTopology(e)],↓(Manifold(p)));args...))
        Makie.wireframe(t::SimplexFrameBundle;args...) = Makie.linesegments(t(edges(t));args...)
        Makie.wireframe!(t::SimplexFrameBundle;args...) = Makie.linesegments!(t(edges(t));args...)
        Makie.mesh(M::SimplexFrameBundle;args...) = Makie.mesh(points(M),ImmersedTopology(M);args...)
        Makie.mesh!(M::SimplexFrameBundle;args...) = Makie.mesh!(points(M),ImmersedTopology(M);args...)
        for fun ∈ (:mesh,:mesh!,:wireframe,:wireframe!)
            @eval Makie.$fun(M::GridFrameBundle;args...) = Makie.$fun(GeometryBasics.Mesh(M);args...)
        end
        Makie.mesh(M::TensorField{B,F,2,<:GridFrameBundle} where {B,F};args...) = Makie.mesh(GeometryBasics.Mesh(base(M));color=fiber(M)[:],args...)
        Makie.mesh!(M::TensorField{B,F,2,<:GridFrameBundle} where {B,F};args...) = Makie.mesh!(GeometryBasics.Mesh(base(M));color=fiber(M)[:],args...)
        function Makie.mesh(M::SimplexFrameBundle;args...)
            if mdims(points(M)) == 2
                sm = submesh(M)[:,1]
                Makie.lines(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh(submesh(M),array(ImmersedTopology(M));args...)
            end
        end
        function Makie.mesh!(M::SimplexFrameBundle,t;args...)
            if mdims(points(M)) == 2
                sm = submesh(M)[:,1]
                Makie.lines!(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh!(submesh(M),array(ImmersedTopology(M));args...)
            end
        end
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        UnicodePlots.lineplot(t::ScalarMap;args...) = UnicodePlots.lineplot(getindex.(domain(t),2),codomain(t);args...)
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(Real.(domain(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot(t::ComplexMap{B,F,1};args...) where {B<:AbstractReal,F} = UnicodePlots.lineplot(real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,F,1};args...) where {G,B<:Coordinate{<:AbstractReal},F} = UnicodePlots.lineplot(Real.(domain(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
        UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
        UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(codomain(t));args...)
        UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(codomain(t));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],args...)
        UnicodePlots.heatmap(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.heatmap(Real.(angle.(codomain(t)));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],colormap=:twilight,args...)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,F,1,<:Interval}) where {B,F} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F}) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
        Base.display(t::GradedField{G,B,F,1,<:Interval}) where {G,B,F} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::SurfaceGrid) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:LocalFiber = GeometryBasics.Point(base(t))
        #GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
        function GeometryBasics.Mesh(m::GridFrameBundle)
            nm = size(points(m))
            faces = GeometryBasics.Tesselation(GeometryBasics.Rect(0, 0, 1, 1), nm)
            uv = Chain(0.0,0.0):map(inv,Chain((nm.-1)...)):Chain(1.0,1.0)
            GeometryBasics.Mesh(GeometryBasics.meta(GeometryBasics.Point.(points(m)[:]); uv=GeometryBasics.Vec{2}.(value.(uv[:]))), GeometryBasics.decompose(GeometryBasics.QuadFace{GeometryBasics.GLIndex}, faces))
        end
        function initmesh(m::GeometryBasics.Mesh)
            c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
            s = size(eltype(c))[1]+1; V = Submanifold(ℝ^s) # s
            n = size(eltype(f))[1]
            p = [Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c]
            M = s ≠ n ? p(list(s-n+1,s)) : p
            t = SimplexManifold([Values{n,Int}(k) for k ∈ f])
            return (p,∂(t),t)
        end
    end
    @require MATLAB="10e44e05-a98a-55b3-a45b-ba969058deb6" begin
        const matlab_cache = (Array{T,2} where T)[]
        const matlab_top_cache = (Array{T,2} where T)[]
        function matlab(p::Array{T,2} where T,B)
            for k ∈ length(matlab_cache):B
                push!(matlab_cache,Array{Any,2}(undef,0,0))
            end
            matlab_cache[B] = p
        end
        function matlab_top(p::Array{T,2} where T,B)
                for k ∈ length(matlab_top_cache):B
                push!(matlab_top_cache,Array{Any,2}(undef,0,0))
            end
            matlab_top_cache[B] = p
        end
        function matlab(p::SimplexFrameBundle)
            B = bundle(p)
            if length(matlab_cache)<B || isempty(matlab_cache[B])
                ap = array(p)'
                matlab(islocal(p) ? vcat(ap,ones(length(p))') : ap[2:end,:],B)
            else
                return matlab_cache[B]
            end
        end
        function matlab(p::SimplexManifold)
            B = bundle(p)
            if length(matlab_top_cache)<B || isempty(matlab_top_cache[B])
                ap = array(p)'
                matlab_top(islocal(p) ? vcat(ap,ones(length(p))') : ap[2:end,:],B)
            else
                return matlab_top_cache[B]
            end
        end
        initmesh(g,args...) = initmeshall(g,args...)[list(1,3)]
        initmeshall(g::Matrix{Int},args...) = initmeshall(Matrix{Float64}(g),args...)
        function initmeshall(g,args...)
            P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
            p,e,t = initmeshdata(P,E,T,Val(2))
            return (p,e,t,T,E,P)
        end
        function initmeshes(g,args...)
            p,e,t,T = initmeshall(g,args...)
            p,e,t,[Int(T[end,k]) for k ∈ 1:size(T,2)]
        end
        export initmeshes
        function refinemesh(g,args...)
            p,e,t,T,E,P = initmeshall(g,args...)
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            return (g,p,e,t)
        end
        #=refinemesh3(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,3,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh4(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,4,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh(g,p::ChainBundle,e,t) = refinemesh3(g,p,e,t)
        refinemesh(g,p::ChainBundle,e,t,s::String) = refinemesh3(g,p,e,t,s)
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int}) = refinemesh3(g,p,e,t,float.(η))
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int},s::String) = refinemesh3(g,p,e,t,float.(η),s)
        refinemes(g,p::ChainBundle,e,t,u) = refinemesh4(g,p,e,t,u)
        refinemesh(g,p::ChainBundle,e,t,u,s::String) = refinemesh4(g,p,e,t,u,s)
        refinemesh(g,p::ChainBundle,e,t,u,η) = refinemesh4(g,p,e,t,u,float.(η))
        refinemesh(g,p::ChainBundle,e,t,u,η,s) = refinemesh4(g,p,e,t,u,float.(η),s)
        refinemesh!(g::Matrix{Int},p::ChainBundle,args...) = refinemesh!(Matrix{Float64}(g),p,args...)
        function refinemesh!(g,p::ChainBundle{V},e,t,s...) where V
            P,E,T = refinemesh(g,p,e,t,s...); l = size(P,1)+1
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            submesh!(p); array!(t); el,tl = list(1,l-1),list(1,l)
            bundle_cache[bundle(p)] = [Chain{V,1,Float64}(vcat(1,P[:,k])) for k ∈ 1:size(P,2)]
            bundle_cache[bundle(e)] = [Chain{↓(p),1,Int}(Int.(E[el,k])) for k ∈ 1:size(E,2)]
            bundle_cache[bundle(t)] = [Chain{p,1,Int}(Int.(T[tl,k])) for k ∈ 1:size(T,2)]
            return (p,e,t)
        end=#
    end
end

end # module Cartan
