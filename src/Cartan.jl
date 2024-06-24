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
# _________                __                   ________
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
import Base: @pure, OneTo

export Values, Derivation
export initmesh, pdegrad, det

export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid
export RealFunction, ComplexMap, SpinorField, CliffordField
export MeshFunction, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap
export LocalSection, GlobalFiber, LocalFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype, graph
export ProductSpace, RealRegion, Interval, Rectangle, Hyperrectangle, ⧺, ⊕

# ProductSpace

struct ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}
    v::Values{M,S} # how to deal with T???
    ProductSpace{V,T,N}(v::Values{M,S}) where {V,T,N,M,S} = new{V,T,N,M,S}(v)
    ProductSpace{V,T}(v::Values{M,S}) where {V,T,M,S} = new{V,T,mdims(V),M,S}(v)
end

const RealRegion{V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N,N,S}
const Interval{V,T,S} = RealRegion{V,T,1,S}
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

export Coordinate

struct Coordinate{P,G} <: LocalFiber{P,G}
    v::Pair{P,G}
    Coordinate(v::Pair{P,G}) where {P,G} = new{P,G}(v)
    Coordinate(p::P,g::G) where {P,G} = new{P,G}(p=>g)
end

point(c) = c
point(c::Coordinate) = base(c)
metrictensor(c) = InducedMetric()
metrictensor(c::Coordinate) = fiber(c)

graph(s::LocalFiber{<:AbstractReal,<:AbstractReal}) = Chain(Real(base(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Chain,<:AbstractReal}) = Chain(value(base(s))...,Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:AbstractReal},<:AbstractReal}) = Chain(Real(basepoint(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:Chain},<:AbstractReal}) = Chain(value(basepoint(s))...,Real(fiber(s)))

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

# GlobalFiber

abstract type GlobalFiber{E,N} <: AbstractArray{E,N} end
Base.@pure isfiberbundle(::GlobalFiber) = true
Base.@pure isfiberbundle(::Any) = false

base(t::GlobalFiber) = t.dom
fiber(t::GlobalFiber) = t.cod
base(t::Array) = ProductSpace(Values(axes(t)))
fiber(t::Array) = t
basepoints(t::GlobalFiber) = points(base(t))
basetype(::Array{T}) where T = T
basepointstype(t::GlobalFiber) = pointstype(base(t))
fibertype(::Array{T}) where T = T

unitdomain(t::GlobalFiber) = base(t)*inv(base(t)[end])
arcdomain(t::GlobalFiber) = unitdomain(t)*arclength(codomain(t))
graph(t::GlobalFiber) = graph.(t)

Base.size(m::GlobalFiber) = size(m.cod)
Base.resize!(m::GlobalFiber,i) = (resize!(domain(m),i),resize!(codomain(m),i))

# GridManifold

export AbstractManifold, GridManifold

abstract type AbstractManifold{M,N} <: GlobalFiber{M,N} end

grid_id = 0

struct GridManifold{P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} <: AbstractManifold{Coordinate{P,G},N}
    id::Int
    dom::PA
    cod::GA
    GridManifold(id::Int,p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{P,G,N,PA,GA}(id,p,g)
    GridManifold(p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{P,G,N,PA,GA}((global grid_id+=1),p,g)
end

points(m::GridManifold) = m.dom
basepoints(m::GridManifold) = points(m)
metrictensor(m::GridManifold) = m.cod

pointtype(m::GridManifold{P}) where P = P
metrictype(m::GridManifold{P,G} where P) where G = G

GridManifold(dom::GridManifold,fun) = GridManifold(base(dom), fun)
GridManifold(dom::GridManifold,fun::Array) = GridManifold(base(dom), fun)
GridManifold(dom::GridManifold,fun::Function) = GridManifold(base(dom), fun)
GridManifold(dom::AbstractArray,fun::Function) = GridManifold(dom, fun.(dom))
basetype(::GridManifold{B}) where B = B
basetype(::Type{<:GridManifold{B}}) where B = B

Base.size(m::GridManifold) = size(points(m))
Base.resize!(m::GridManifold,i) = (resize!(points(m),i),resize!(metrictensor(m),i))
Base.broadcast(f,t::GridManifold) = GridManifold(f.(points(t)),f.(metrictensor(t)))

# GlobalSection

struct GlobalSection{B,F,N,BA,FA<:AbstractArray{F,N}} <: GlobalFiber{LocalSection{B,F},N}
    dom::BA
    cod::FA
    GlobalSection{B}(dom::BA,cod::FA) where {B,F,N,BA,FA<:AbstractArray{F,N}} = new{B,F,N,BA,FA}(dom,cod)
    GlobalSection(dom::BA,cod::FA) where {N,B,F,BA<:AbstractArray{B,N},FA<:AbstractArray{F,N}} = new{B,F,N,BA,FA}(dom,cod)
    GlobalSection(dom::BA,cod::FA) where {BA<:ChainBundle,F,FA<:AbstractVector{F}} = new{eltype(value(points(dom))),F,1,BA,FA}(dom,cod)
end

# TensorField

struct TensorField{B,F,N,PA,GA} <: GlobalFiber{LocalTensor{B,F},N}
    id::Int
    dom::PA
    cod::Array{F,N}
    met::GA
    TensorField(id::Int,dom::PA,cod::Array{F,N},met::GA=Global{N}(InducedMetric())) where {N,P,F,G,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{Coordinate{P,G},F,N,PA,GA}(id,dom,cod,met)
    TensorField(id::Int,dom::PA,cod::Vector{F},met::GA=Global{N}(InducedMetric())) where {F,G,PA<:ChainBundle,GA<:AbstractVector{G}} = new{Coordinate{eltype(value(points(dom))),G},F,1,PA,GA}(id,dom,cod,met)
    TensorField(dom::PA,cod::Array{F,N},met::GA=Global{N}(InducedMetric())) where {N,P,F,G,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{Coordinate{P,G},F,N,PA,GA}((global grid_id+=1),dom,cod,met)
    TensorField(dom::MA,cod::Array{F,N}) where {F,N,P,G,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N},MA<:GridManifold{P,G,N,PA,GA}} = new{Coordinate{P,G},F,N,PA,GA}(dom.id,dom.dom,cod,dom.cod)
    TensorField(dom::PA,cod::Vector{F},met::GA=Global{N}(InducedMetric())) where {F,G,PA<:ChainBundle,GA<:AbstractVector{G}} = new{Coordinate{eltype(value(points(dom))),G},F,1,PA,GA}((global grid_id+=1),dom,cod,met)
end

#const ParametricMesh{B,F,PA<:AbstractVector{<:Chain},GA} = TensorField{B,F,1,PA,GA}
const MeshFunction{B,F<:AbstractReal,BA<:ChainBundle,GA} = TensorField{B,F,1,BA,GA}
const ElementFunction{B,F<:AbstractReal,PA<:AbstractVector,GA} = TensorField{B,F,1,PA,GA}
const IntervalMap{B,F,PA<:AbstractVector{<:AbstractReal},GA} = TensorField{B,F,1,PA,GA}
const RectangleMap{B,F,BA<:Rectangle,GA} = TensorField{B,F,2,BA,GA}
const HyperrectangleMap{B,F,PA<:Hyperrectangle,GA} = TensorField{B,F,3,PA,GA}
const ParametricMap{B,F,N,PA<:RealRegion,GA} = TensorField{B,F,N,PA,GA}
const RealFunction{B,F<:AbstractReal,PA<:AbstractVector{<:AbstractReal},GA} = TensorField{B,F,1,PA,GA}
const PlaneCurve{B,F<:Chain{V,G,Q,2} where {V,G,Q},PA<:AbstractVector{<:AbstractReal},GA} = TensorField{B,F,1,PA,GA}
const SpaceCurve{B,F<:Chain{V,G,Q,3} where {V,G,Q},PA<:AbstractVector{<:AbstractReal},GA} = TensorField{B,F,1,PA,GA}
const SurfaceGrid{B,F<:AbstractReal,PA<:AbstractMatrix,GA} = TensorField{B,F,2,PA,GA}
const VolumeGrid{B,F<:AbstractReal,PA<:AbstractArray{P,3} where P,GA} = TensorField{B,F,3,PA,GA}
const ScalarGrid{B,F<:AbstractReal,N,PA<:AbstractArray,GA} = TensorField{B,F,N,PA,GA}
#const ParametricGrid{B,F,N,PA<:AbstractArray,GA} = TensorField{B,F,N,PA,GA}
const GlobalFrame{B<:LocalFiber{P,<:TensorNested} where P,N} = GlobalSection{B,N}
const DiagonalField{B,F<:DiagonalOperator,N,PA,GA} = TensorField{B,F,N,PA,GA}
const EndomorphismField{B,F<:Endomorphism,N,PA,GA} = TensorField{B,F,N,PA,GA}
const OutermorphismField{B,F<:Outermorphism,N,PA,GA} = TensorField{B,F,N,PA,GA}
const CliffordField{B,F<:Multivector,N,PA,GA} = TensorField{B,F,N,PA,GA}
const QuaternionField{B,F<:Quaternion,N,PA,GA} = TensorField{B,F,N,PA,GA}
const ComplexMap{B,F<:AbstractComplex,N,PA,GA} = TensorField{B,F,N,PA,GA}
const PhasorField{B,F<:Phasor,N,PA,GA} = TensorField{B,F,N,PA,GA}
const SpinorField{B,F<:AbstractSpinor,N,PA,GA} = TensorField{B,F,N,PA,GA}
const GradedField{G,B,F<:Chain{V,G} where V,N,PA,GA} = TensorField{B,F,N,PA,GA}
const ScalarField{B,F<:AbstractReal,N,PA,GA} = TensorField{B,F,N,PA,GA}
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
fibertype(::GridManifold{B,F} where B) where F = F
fibertype(::Type{<:GridManifold{B,F} where B}) where F = F
fibertype(::GlobalSection{B,F} where B) where F = F
fibertype(::Type{<:GlobalSection{B,F} where B}) where F = F
fibertype(::TensorField{B,F} where B) where F = F
fibertype(::Type{<:TensorField{B,F} where B}) where F = F
base(t::TensorField) = GridManifold(t.id,basepoints(t),metrictensor(t))
basepoints(t::TensorField) = t.dom
metrictensor(t::TensorField) = t.met

@pure Base.eltype(::Type{<:GridManifold{P,G}}) where {P,G} = Coordinate{P,G}
Base.getindex(m::GridManifold,i::Vararg{Int}) = Coordinate(getindex(points(m),i...), getindex(metrictensor(m),i...))
Base.setindex!(m::GridManifold{P},s::P,i::Vararg{Int}) where P = setindex!(points(m),s,i...)
Base.setindex!(m::GridManifold{P,G} where P,s::G,i::Vararg{Int}) where G = setindex!(metrictensor(m),s,i...)
function Base.setindex!(m::GridManifold,s::Coordinate,i::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metrictensor(m),metrictensor(s),i...)
    return s
end

@pure Base.eltype(::Type{<:TensorField{B,F}}) where {B,F} = LocalTensor{B,F}
Base.getindex(m::TensorField,i::Vararg{Int}) = LocalTensor(getindex(domain(m),i...), getindex(codomain(m),i...))
Base.getindex(m::ElementFunction{R,F,<:ChainBundle} where {R,F},i::Vararg{Int}) = LocalTensor(getindex(value(points(domain(m))),i...), getindex(codomain(m),i...))
Base.setindex!(m::TensorField{B,F,1,<:AbstractRange},s::LocalTensor,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,F,N,<:RealRegion{V,T,N,<:AbstractRange} where {V,T}},s::LocalTensor,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,F,N,<:AbstractArray},s::F,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::TensorField{B,F,N,<:Array} where {F,N},s::LocalTensor,i::Vararg{Int}) where B
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

@pure Base.eltype(::Type{<:GlobalSection{B,F}}) where {B,F} = LocalSection{B,F}
Base.getindex(m::GlobalSection,i::Vararg{Int}) = LocalSection(getindex(domain(m),i...), getindex(codomain(m),i...))
#Base.getindex(m::ElementFunction{R,F,<:ChainBundle} where {R,F},i::Vararg{Int}) = LocalSection(getindex(value(points(domain(m))),i...), getindex(codomain(m),i...))
Base.setindex!(m::GlobalSection{B,F,1,<:AbstractRange},s::LocalSection,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::GlobalSection{B,F,N,<:RealRegion{V,T,N,<:AbstractRange} where {V,T}},s::LocalSection,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::GlobalSection{B,Fm,N,<:AbstractArray} where Fm,s::F,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::GlobalSection{B,F,N,<:Array},s::LocalSection,i::Vararg{Int}) where {B,F,N}
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

Base.BroadcastStyle(::Type{<:GridManifold{P,G,N,PA,GA}}) where {P,G,N,PA,GA} = Broadcast.ArrayStyle{GridManifold{P,G,N,PA,GA}}()
Base.BroadcastStyle(::Type{<:TensorField{B,F,N,PA,GA}}) where {B,F,N,PA,GA} = Broadcast.ArrayStyle{TensorField{B,F,N,PA,GA}}()
Base.BroadcastStyle(::Type{<:GlobalSection{B,F,N,BA,FA}}) where {B,F,N,BA,FA} = Broadcast.ArrayStyle{TensorField{B,F,N,BA,FA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridManifold{P,G,N,PA,GA}}}, ::Type{ElType}) where {P,G,N,PA,GA,ElType}
    ax = axes(bc)
    # Use the data type to create the output
    GridManifold(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,F,N,PA,GA}}}, ::Type{ElType}) where {B,F,N,PA,GA,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    TensorField(t.id, t.dom, similar(Array{fibertype(ElType),N}, axes(bc)), t.met)
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlobalSection{B,F,N,BA,FA}}}, ::Type{ElType}) where {B,F,N,BA,FA,ElType}
    # Scan the inputs for the TensorField:
    t = find_gs(bc)
    # Use the domain field of t to create the output
    GlobalSection(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end

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

(m::IntervalMap{B,F,<:AbstractVector{B}})(s::LocalTensor) where {B<:AbstractReal,F} = LocalTensor(base(s), m(fiber(s)))
function (m::IntervalMap{B,F,<:AbstractVector{B}})(t) where {B<:AbstractReal,F}
    i = searchsortedfirst(domain(m),t[1])-1
    linterp(t[1],m.dom[i],m.dom[i+1],m.cod[i],m.cod[i+1])
end
function (m::IntervalMap{B,F,<:AbstractVector{B}})(t::Vector,d=diff(m.cod)./diff(m.dom)) where {B<:AbstractReal,F}
    [parametric(i,m,d) for i ∈ t]
end
function parametric(t,m,d=diff(codomain(m))./diff(domain(m)))
    i = searchsortedfirst(domain(m),t)-1
    codomain(m)[i]+(t-domain(m)[i])*d[i]
end

function (m::TensorField{B,F,N,<:ChainBundle} where {B,F,N})(t)
    i = domain(m)[findfirst(t,domain(m))]
    (codomain(m)[i])⋅(points(domain(m))[i]/t)
end

(m::TensorField{B,F,N,<:Rectangle} where {B,F,N})(x,y) = m(Chain(x,y))
(m::TensorField{B,F,N,<:Rectangle} where {B,F,N})(s::LocalSection) = LocalTensor(base(s), m(fiber(s)))
function (m::TensorField{B,F,N,<:Rectangle} where {B,F,N})(t::Chain)
    x,y = basepoints(m).v[1],basepoints(m).v[2]
    i,j = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1
    #f1 = linterp(t[1],x[i],x[i+1],m.cod[i,j],m.cod[i+1,j])
    #f2 = linterp(t[1],x[i],x[i+1],m.cod[i,j+1],m.cod[i+1,j+1])
    #linterp(t[2],y[j],y[j+1],f1,f2)
    bilinterp(t[1],t[2],x[i],x[i+1],y[j],y[j+1],
        m.cod[i,j],m.cod[i+1,j],m.cod[i,j+1],m.cod[i+1,j+1])
end

(m::TensorField{B,F,N,<:Hyperrectangle} where {B,F,N})(x,y,z) = m(Chain(x,y,z))
(m::TensorField{B,F,N,<:Hyperrectangle} where {B,F,N})(s::LocalTensor) = LocalTensor(base(s), m(fiber(s)))
function (m::TensorField{B,F,N,<:Hyperrectangle} where {B,F,N})(t::Chain)
    x,y,z = basepoints(m).v[1],basepoints(m).v[2],basepoints(m).v[3]
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
for fun ∈ (:exp,:log,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2)
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
                Makie.$lines(t::SpaceCurve;args...) = Makie.$lines(codomain(t);color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::PlaneCurve;args...) = Makie.$lines(codomain(t);color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::RealFunction;args...) = Makie.$lines(Real.(basepoints(t)),Real.(codomain(t));color=Real.(codomain(speed(t))),args...)
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
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(basepoints(t).v...,Real.(codomain(t));color=Real.(abs.(codomain(gradient_fast(Real(t))))),args...)
                Makie.$fun(t::ComplexMap{B,F,2,<:Rectangle} where {B,F<:AbstractComplex};args...) = Makie.$fun(basepoints(t).v...,Real.(radius.(codomain(t)));color=Real.(angle.(codomain(t))),colormap=:twilight,args...)
                function Makie.$fun(t::GradedField{G,B,F,2,<:Rectangle} where G;args...) where {B,F<:Chain}
                    x,y = basepoints(t),value.(codomain(t))
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
                Makie.$fun(t::ComplexMap{B,F,2,<:Rectangle} where {B,F};args...) = Makie.$fun(domain(t).v...,Real.(radius.(codomain(t)));args...)
            end
        end
        for fun ∈ (:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,F,2,<:Rectangle} where {B,F};args...) = Makie.$fun(domain(t).v...,Real.(angle.(codomain(t)));colormap=:twilight,args...)
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(basepoints(t).v...,Real.(codomain(t));args...)
                function Makie.$fun(t::GradedField{G,B,F,2,<:Rectangle} where G;args...) where {B,F}
                    x,y = basepoints(t),value.(codomain(t))
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
                Makie.$fun(m::ScalarField{<:Coordinate{<:Chain},<:AbstractReal,N,<:RealRegion} where N;args...) = Makie.$fun(gradient_fast(m);args...)
                Makie.$fun(m::ScalarField{R,<:AbstractReal,1,<:ChainBundle} where R,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
                Makie.$fun(m::VectorField{R,F,1,<:ChainBundle} where {R,F},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
                Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealRegion} where {F,N};args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),basepoints(m).v...;args...)
            end
        end
        for fun ∈ (:arrows,:arrows!)
            @eval begin
                Makie.$fun(t::ScalarField{<:Coordinate{<:Chain},F,N,<:Rectangle} where {F,N};args...) = Makie.$fun(Makie.Point.(fiber(graph(Real(int))))[:],Makie.Point.(fiber(normal(Real(int))))[:];args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,2,<:Rectangle} where F;args...) = Makie.$fun(domain(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,3,<:Hyperrectangle} where F;args...) = Makie.$fun(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::MeshFunction;args...) = Makie.$fun(points(basepoints(t)),Real.(codomain(t));args...)
            end
        end
        Makie.mesh(t::ElementFunction;args...) = Makie.mesh(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh!(t::ElementFunction;args...) = Makie.mesh!(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh(t::MeshFunction;args...) = Makie.mesh(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh!(t::MeshFunction;args...) = Makie.mesh!(domain(t);color=Real.(codomain(t)),args...)
        Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=Real.(codomain(t)),args...)
        Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=Real.(codomain(t)),args...)
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(Real.(domain(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot(t::ComplexMap{B,F,1};args...) where {B<:AbstractReal,F} = UnicodePlots.lineplot(real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,F,1};args...) where {G,B<:Coordinate{<:AbstractReal},F} = UnicodePlots.lineplot(Real.(domain(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::ComplexMap{B,F,2,<:Rectangle} where {B,F};args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
        UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::ComplexMap{B,F,2,<:Rectangle} where {B,F};args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
        UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(codomain(t));args...)
        UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(codomain(t));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],args...)
        UnicodePlots.heatmap(t::ComplexMap{B,F,2,<:Rectangle} where {B,F};args...) = UnicodePlots.heatmap(Real.(angle.(codomain(t)));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],colormap=:twilight,args...)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,F,1,<:AbstractVector{<:AbstractReal}}) where {B,F} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,F,2,<:Rectangle} where {B,F}) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
        Base.display(t::GradedField{G,B,F,1,<:AbstractVector{<:AbstractReal}}) where {G,B,F} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::SurfaceGrid) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
    end
end

end # module Cartan
