module TensorFields

#   This file is part of TensorFields.jl
#   It is licensed under the GPL license
#   TensorFields Copyright (C) 2023 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com
#  _____                           ___ _      _     _
# /__   \___ _ __  ___  ___  _ __ / __(_) ___| | __| |___
#   / /\/ _ \ '_ \/ __|/ _ \| '__/ _\ | |/ _ \ |/ _` / __|
#  / / |  __/ | | \__ \ (_) | | / /   | |  __/ | (_| \__ \
#  \/   \___|_| |_|___/\___/|_| \/    |_|\___|_|\__,_|___/

using SparseArrays, LinearAlgebra, Base.Threads
using AbstractTensors, DirectSum, Grassmann, Requires
import Grassmann: value, vector, valuetype, tangent, Derivation, radius
import Base: @pure, OneTo
import AbstractTensors: Values, Variables, FixedVector
import AbstractTensors: Scalar, GradedVector, Bivector, Trivector
import DirectSum: ⊕

export Values, Derivation
export initmesh, pdegrad, det

export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid
export RealFunction, ComplexMap, SpinorField, CliffordField
export MeshFunction, GradedField, QuaternionField, PhasorField
export ParametricMap, RectangleMap, HyperrectangleMap
export Section, FiberBundle, AbstractFiber
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

# AbstractFiber

abstract type AbstractFiber <: Number end
Base.@pure isfiber(::AbstractFiber) = true
Base.@pure isfiber(::Any) = false

# Section

struct Section{B,F} <: AbstractFiber
    v::Pair{B,F}
    Section(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    Section(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    Section(b::B,f::Section{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    Section(b::Section{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

Section(b,f::Function) = b ↦ f(b)

fiber(s) = s
fibertype(s) = typeof(s)
fibertype(::Type{T}) where T = T
base(s::Section) = s.v.first
fiber(s::Section) = s.v.second
basetype(::Section{B}) where B = B
fibertype(::Section{B,F} where B) where F = F
basetype(::Type{<:Section{B}}) where B = B
fibertype(::Type{<:Section{B,F} where B}) where F = F
graph(s::Section{<:AbstractReal,<:AbstractReal}) = Chain(Real(base(s)),Real(fiber(s)))
graph(s::Section{<:Chain,<:AbstractReal}) = Chain(value(base(s))...,Real(fiber(s)))
const ↦, domain, codomain = Section, base, fiber
↤(F,B) = B ↦ F

Base.getindex(s::Section) = s.v.first
Base.getindex(s::Section,i::Int) = getindex(s.v.second,i)
Base.getindex(s::Section,i::Integer) = getindex(s.v.second,i)

function Base.show(io::IO, s::Section)
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

function show_pairtyped(io::IO, s::Section{B,F}) where {B,F}
    show(io, typeof(s))
    show(io, (base(s), fiber(s)))
end

for fun ∈ (:-,:!,:~,:inv,:exp,:log,:sinh,:cosh,:abs,:sqrt,:real,:imag,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2,:conj)
    @eval Base.$fun(s::Section) = base(s) ↦ $fun(fiber(s))
end
for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:curl,:∂,:d,:⋆,:angle,:radius)
    @eval Grassmann.$fun(s::Section) = base(s) ↦ $fun(fiber(s))
end
for op ∈ (:+,:-,:*,:/,:<,:>,:<<,:>>,:&)
    @eval Base.$op(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),$op(fiber(a),fiber(b))) : error("Section $(base(a)) ≠ $(base(b))")
end
for op ∈ (:∧,:∨)
    @eval Grassmann.$op(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),$op(fiber(a),fiber(b))) : error("Section $(base(a)) ≠ $(base(b))")
end

Grassmann.contraction(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),Grassmann.contraction(fiber(a),fiber(b))) : error("Section $(base(a)) ≠ $(base(b))")

Base.:*(a::Number,b::Section) = base(b) ↦ (a*fiber(b))
Base.:*(a::Section,b::Number) = base(a) ↦ (fiber(a)*b)
Base.:/(a::Section,b::Number) = base(a) ↦ (fiber(a)/b)
LinearAlgebra.norm(s::Section) = base(s) ↦ norm(fiber(s))
LinearAlgebra.det(s::Section) = base(s) ↦ det(fiber(s))
(V::Submanifold)(s::Section) = base(a) ↦ V(fiber(s))
(::Type{T})(s::Section) where T<:Real = base(s) ↦ T(fiber(s))
(::Type{Complex})(s::Section) = base(s) ↦ Complex(fiber(s))
(::Type{Complex{T}})(s::Section) where T = base(s) ↦ Complex{T}(fiber(s))
Grassmann.Phasor(s::Section) = base(s) ↦ Phasor(fiber(s))
Grassmann.Couple(s::Section) = base(s) ↦ Couple(fiber(s))

# FiberBundle

abstract type FiberBundle{E,N} <: AbstractArray{E,N} end
Base.@pure isfiberbundle(::FiberBundle) = true
Base.@pure isfiberbundle(::Any) = false

# TensorField

struct TensorField{B,T,F,N} <: FiberBundle{Section{B,F},N}
    dom::T
    cod::Array{F,N}
    TensorField{B}(dom::T,cod::Array{F,N}) where {B,T,F,N} = new{B,T,F,N}(dom,cod)
    TensorField(dom::T,cod::Array{F,N}) where {N,B,T<:AbstractArray{B,N},F} = new{B,T,F,N}(dom,cod)
    TensorField(dom::T,cod::Vector{F}) where {T<:ChainBundle,F} = new{eltype(value(points(dom))),T,F,1}(dom,cod)
end

#const ParametricMesh{B<:Chain,T<:AbstractVector{B},F} = TensorField{B,T,F,1}
const MeshFunction{B,T<:ChainBundle,F<:AbstractReal} = TensorField{B,T,F,1}
const ElementFunction{B,T<:AbstractVector{B},F<:AbstractReal} = TensorField{B,T,F,1}
const IntervalMap{B<:AbstractReal,T<:AbstractVector{B},F} = TensorField{B,T,F,1}
const RectangleMap{B,T<:Rectangle,F} = TensorField{B,T,F,2}
const HyperrectangleMap{B,T<:Hyperrectangle,F} = TensorField{B,T,F,3}
const ParametricMap{B,T<:RealRegion,F,N} = TensorField{B,T,F,N}
const RealFunction{B<:AbstractReal,T<:AbstractVector{B},F<:AbstractReal} = TensorField{B,T,F,1}
const PlaneCurve{B<:AbstractReal,T<:AbstractVector{B},F<:Chain{V,G,Q,2} where {V,G,Q}} = TensorField{B,T,F,1}
const SpaceCurve{B<:AbstractReal,T<:AbstractVector{B},F<:Chain{V,G,Q,3} where {V,G,Q}} = TensorField{B,T,F,1}
const SurfaceGrid{B,T<:AbstractMatrix{B},F<:AbstractReal} = TensorField{B,T,F,2}
const VolumeGrid{B,T<:AbstractArray{B,3},F<:AbstractReal} = TensorField{B,T,F,3}
const ScalarGrid{B,T<:AbstractArray{B},F<:AbstractReal,N} = TensorField{B,T,F,N}
#const ParametricGrid{B,T<:AbstractArray{B},F,N} = TensorField{B,T,F,N}
const CliffordField{B,T,F<:Multivector,N} = TensorField{B,T,F,N}
const QuaternionField{B,T,F<:Quaternion,N} = TensorField{B,T,F,N}
const ComplexMap{B,T,F<:AbstractComplex,N} = TensorField{B,T,F,N}
const PhasorField{B,T,F<:Phasor,N} = TensorField{B,T,F,N}
const SpinorField{B,T,F<:AbstractSpinor,N} = TensorField{B,T,F,N}
const GradedField{G,B,T,F<:Chain{V,G} where V,N} = TensorField{B,T,F,N}
const ScalarField{B,T,F<:AbstractReal,N} = TensorField{B,T,F,N}
const VectorField = GradedField{1}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

TensorField(dom,fun::BitArray) = dom → Float64.(fun)
TensorField(dom,fun::TensorField) = dom → fiber(fun)
TensorField(dom::TensorField,fun) = base(dom) → fun
TensorField(dom::TensorField,fun::Array) = base(dom) → fun
TensorField(dom::TensorField,fun::Function) = base(dom) → fun
TensorField(dom::AbstractArray,fun::AbstractRange) = dom → collect(fun)
TensorField(dom::AbstractArray,fun::RealRegion) = dom → collect(fun)
TensorField(dom::AbstractArray,fun::Function) = dom → fun.(dom)
TensorField(dom::ChainBundle,fun::Function) = dom → fun.(value(points(dom)))

←(F,B) = B → F
const → = TensorField
base(t::Array) = ProductSpace(Values(axes(t)))
fiber(t::Array) = t
base(t::TensorField) = t.dom
fiber(t::TensorField) = t.cod
basetype(::TensorField{B}) where B = B
basetype(::Array{T}) where T = Int
fibertype(::TensorField{B,T,F} where {B,T}) where F = F
fibertype(::Array{T}) where T = T
basetype(::Type{<:TensorField{B}}) where B = B
fibertype(::Type{<:TensorField{B,T,F} where {B,T}}) where F = F

unitdomain(t::TensorField) = base(t)*inv(base(t)[end])
arcdomain(t::TensorField) = unitdomain(t)*arclength(codomain(t))
graph(t::TensorField) = graph.(t)

Base.size(m::TensorField) = size(m.cod)
Base.resize!(m::TensorField,i) = (resize!(domain(m),i),resize!(codomain(m),i))
@pure Base.eltype(::Type{TensorField{B,T,F}}) where {B,T,F} = Section{B,F}
Base.broadcast(f,t::TensorField) = domain(t) → f.(codomain(t))
Base.getindex(m::TensorField,i::Vararg{Int}) = getindex(domain(m),i...) ↦ getindex(codomain(m),i...)
Base.getindex(m::ElementFunction{R,<:ChainBundle} where R,i::Vararg{Int}) = getindex(value(points(domain(m))),i...) ↦ getindex(codomain(m),i...)
Base.setindex!(m::TensorField{B,<:AbstractRange{B}},s::Section,i::Vararg{Int}) where B = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,<:RealRegion{V,T,N,<:AbstractRange} where {V,T,N}},s::Section,i::Vararg{Int}) where B = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,<:AbstractArray{B},F},s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::TensorField{B,<:Array{B}},s::Section,i::Vararg{Int}) where B
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

Base.BroadcastStyle(::Type{<:TensorField{B,T,F,N}}) where {B,T,F,N} = Broadcast.ArrayStyle{TensorField{B,T,F,N}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,T,F,N}}}, ::Type{ElType}) where {B,T,F,N,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    domain(t) → similar(Array{fibertype(ElType),N}, axes(bc))
end

"`A = find_tf(As)` returns the first TensorField among the arguments."
find_tf(bc::Base.Broadcast.Broadcasted) = find_tf(bc.args)
find_tf(args::Tuple) = find_tf(find_tf(args[1]), Base.tail(args))
find_tf(x) = x
find_tf(::Tuple{}) = nothing
find_tf(a::TensorField, rest) = a
find_tf(::Any, rest) = find_tf(rest)

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

(m::IntervalMap{B,<:AbstractVector{B}})(s::Section) where B<:AbstractReal = base(s) ↦ m(fiber(s))
function (m::IntervalMap{B,<:AbstractVector{B}})(t) where B<:AbstractReal
    i = searchsortedfirst(domain(m),t[1])-1
    linterp(t[1],m.dom[i],m.dom[i+1],m.cod[i],m.cod[i+1])
end
function (m::IntervalMap{B,<:AbstractVector{B}})(t::Vector,d=diff(m.cod)./diff(m.dom)) where B<:AbstractReal
    [parametric(i,m,d) for i ∈ t]
end
function parametric(t,m,d=diff(codomain(m))./diff(domain(m)))
    i = searchsortedfirst(domain(m),t)-1
    codomain(m)[i]+(t-domain(m)[i])*d[i]
end

function (m::TensorField{B,<:ChainBundle} where B)(t)
    i = domain(m)[findfirst(t,domain(m))]
    (codomain(m)[i])⋅(points(domain(m))[i]/t)
end

(m::TensorField{R,<:Rectangle} where R)(x,y) = m(Chain(x,y))
(m::TensorField{R,<:Rectangle} where R)(s::Section) = base(s) ↦ m(fiber(s))
function (m::TensorField{R,<:Rectangle} where R)(t::Chain)
    x,y = domain(m).v[1],domain(m).v[2]
    i,j = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1
    #f1 = linterp(t[1],x[i],x[i+1],m.cod[i,j],m.cod[i+1,j])
    #f2 = linterp(t[1],x[i],x[i+1],m.cod[i,j+1],m.cod[i+1,j+1])
    #linterp(t[2],y[j],y[j+1],f1,f2)
    bilinterp(t[1],t[2],x[i],x[i+1],y[j],y[j+1],
        m.cod[i,j],m.cod[i+1,j],m.cod[i,j+1],m.cod[i+1,j+1])
end

(m::TensorField{R,<:Hyperrectangle} where R)(x,y,z) = m(Chain(x,y,z))
(m::TensorField{R,<:Hyperrectangle} where R)(s::Section) = base(s) ↦ m(fiber(s))
function (m::TensorField{R,<:Hyperrectangle} where R)(t::Chain)
    x,y,z = domain(m).v[1],domain(m).v[2],domain(m).v[3]
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

TensorField(t::Chain{V,G}) where {V,G} = base(t[1]) → Chain{V,G}.(valmat(fiber.(value(t))))
Grassmann.Chain(t::TensorField{B,T,<:Union{Real,Complex}} where {B,T}) = Chain{Submanifold(ndims(t)),0}(t)
function Grassmann.Chain(t::TensorField{B,T,<:Chain{V,G}} where {B,T}) where {V,G}
    Chain{V,G}((base(t) → getindex.(fiber(t),j) for j ∈ 1:binomial(mdims(V),G))...)
end
Base.:^(t::TensorField,n::Int) = domain(t) → codomain(t).^n
Base.:^(t::TensorField,n::Number) = domain(t) → codomain(t).^n
for op ∈ (:*,:/,:+,:-,:>,:<,:>>,:<<,:&)
    @eval begin
        Base.$op(a::TensorField,b::TensorField) = checkdomain(a,b) && (domain(a) → $op.(codomain(a),codomain(b)))
        Base.$op(a::TensorField,b::Number) = domain(a) → $op.(codomain(a),Ref(b))
        Base.$op(a::Number,b::TensorField) = domain(b) → $op.(Ref(a),codomain(b))
    end
end
for op ∈ (:∧,:∨,:⋅)
    @eval begin
        Grassmann.$op(a::TensorField,b::TensorField) = checkdomain(a,b) && (domain(a) → $op.(codomain(a),codomain(b)))
        Grassmann.$op(a::TensorField,b::Number) = domain(a) → $op.(codomain(a),Ref(b))
        Grassmann.$op(a::Number,b::TensorField) = domain(b) → $op.(Ref(a),codomain(b))
    end
end
for fun ∈ (:-,:!,:~,:exp,:log,:sinh,:cosh,:abs,:sqrt,:real,:imag,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2,:conj)
    @eval Base.$fun(t::TensorField) = domain(t) → $fun.(codomain(t))
end
for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:⋆,:angle,:radius)
    @eval Grassmann.$fun(t::TensorField) = domain(t) → $fun.(codomain(t))
end
for fun ∈ (:sum,:prod)
    @eval Base.$fun(t::TensorField) = domain(t)[end] ↦ $fun(codomain(t))
end
for fun ∈ (:cumsum,:cumprod)
    @eval function Base.$fun(t::TensorField)
         out = $fun(codomain(t))
         pushfirst!(out,zero(eltype(out)))
         domain(t) → out
    end
end

Grassmann.signbit(::TensorField) = false
Base.inv(t::TensorField) = codomain(t) → domain(t)
Base.diff(t::TensorField) = diff(domain(t)) → diff(codomain(t))
absvalue(t::TensorField) = domain(t) → value.(abs.(codomain(t)))
LinearAlgebra.det(t::TensorField) = domain(t) → det.(codomain(t))
LinearAlgebra.norm(t::TensorField) = domain(t) → norm.(codomain(t))
(V::Submanifold)(t::TensorField) = domain(t) → V.(codomain(t))
(::Type{T})(t::TensorField) where T<:Real = domain(t) → T.(codomain(t))
(::Type{Complex})(t::TensorField) = domain(t) → Complex.(codomain(t))
(::Type{Complex{T}})(t::TensorField) where T = domain(t) → Complex{T}.(codomain(t))
Grassmann.Phasor(s::TensorField) = domain(s) → Phasor(codomain(s))
Grassmann.Couple(s::TensorField) = domain(s) → Couple(codomain(s))

checkdomain(a::TensorField,b::TensorField) = domain(a)≠domain(b) ? error("TensorField domains not equal") : true

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
                Makie.$lines(t::RealFunction;args...) = Makie.$lines(Real.(domain(t)),Real.(codomain(t));color=Real.(codomain(speed(t))),args...)
                Makie.$lines(t::ComplexMap{B,<:AbstractVector{B}};args...) where B<:AbstractReal = Makie.$lines(real.(Complex.(domain(t))),imag.(Complex.(codomain(t)));color=Real.(codomain(speed(t))),args...)
            end
        end
        Makie.lines(t::TensorField{B,<:AbstractVector{B}};args...) where B<:AbstractReal = linegraph(t;args...)
        Makie.lines!(t::TensorField{B,<:AbstractVector{B}};args...) where B<:AbstractReal = linegraph(t;args...)
        function linegraph(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:AbstractReal}
            x,y = Real.(domain(t)),value.(codomain(t))
            display(Makie.lines(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),G)
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function linegraph!(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:AbstractReal}
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
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(domain(t).v...,Real.(codomain(t));color=Real.(abs.(codomain(gradient_fast(Real(t))))),args...)
                Makie.$fun(t::ComplexMap{B,<:Rectangle} where B;args...) = Makie.$fun(domain(t).v...,Real.(radius.(codomain(t)));color=Real.(angle.(codomain(t))),colormap=:twilight,args...)
                function Makie.$fun(t::GradedField{G,B,<:Rectangle};args...) where {G,B}
                    x,y = domain(t),value.(codomain(t))
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
                Makie.$fun(t::ComplexMap{B,<:Rectangle} where B;args...) = Makie.$fun(domain(t).v...,Real.(radius.(codomain(t)));args...)
            end
        end
        for fun ∈ (:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,<:Rectangle} where B;args...) = Makie.$fun(domain(t).v...,Real.(angle.(codomain(t)));colormap=:twilight,args...)
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(domain(t).v...,Real.(codomain(t));args...)
                function Makie.$fun(t::GradedField{G,B,<:Rectangle};args...) where {G,B}
                    x,y = domain(t),value.(codomain(t))
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
                Makie.$fun(m::ScalarField{<:Chain,<:RealRegion,<:AbstractReal};args...) = Makie.$fun(gradient_fast(m);args...)
                Makie.$fun(m::ScalarField{R,<:ChainBundle,<:AbstractReal} where R,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
                Makie.$fun(m::VectorField{R,<:ChainBundle} where R,dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
                Makie.$fun(m::VectorField{<:Chain,<:RealRegion};args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),domain(m).v...;args...)
            end
        end
        for fun ∈ (:arrows,:arrows!)
            @eval begin
                Makie.$fun(t::ScalarField{<:Chain,<:Rectangle};args...) = Makie.$fun(Makie.Point.(fiber(graph(Real(int))))[:],Makie.Point.(fiber(normal(Real(int))))[:];args...)
                Makie.$fun(t::VectorField{<:Chain,<:Rectangle};args...) = Makie.$fun(domain(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Chain,<:Hyperrectangle};args...) = Makie.$fun(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::MeshFunction;args...) = Makie.$fun(points(domain(t)),Real.(codomain(t));args...)
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
        UnicodePlots.lineplot(t::ComplexMap{B,<:AbstractVector{B}};args...) where B<:AbstractReal = UnicodePlots.lineplot(real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:AbstractReal} = UnicodePlots.lineplot(Real.(domain(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::ComplexMap{B,<:Rectangle} where B;args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
        UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::ComplexMap{B,<:Rectangle} where B;args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
        UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(codomain(t));args...)
        UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(codomain(t));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],args...)
        UnicodePlots.heatmap(t::ComplexMap{B,<:Rectangle} where B;args...) = UnicodePlots.heatmap(Real.(angle.(codomain(t)));xfact=step(t.dom.v[1]),yfact=step(t.dom.v[2]),xoffset=t.dom.v[1][1],yoffset=t.dom.v[2][1],colormap=:twilight,args...)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractVector{B}}) where B<:AbstractReal = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:Rectangle} where B) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
        Base.display(t::GradedField{G,B,<:AbstractVector{B}}) where {G,B<:AbstractReal} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::SurfaceGrid) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
    end
end

end # module
