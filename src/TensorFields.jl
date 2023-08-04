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

using SparseArrays, LinearAlgebra
using AbstractTensors, DirectSum, Grassmann, Requires
import Grassmann: value, vector, valuetype, tangent
import Base: @pure
import AbstractTensors: Values, Variables, FixedVector
import AbstractTensors: Scalar, GradedVector, Bivector, Trivector

export Values
export initmesh, pdegrad

export ElementFunction, IntervalMap, PlaneCurve, SpaceCurve, GridSurface, GridParametric
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export RealFunction, ComplexMap, ComplexMapping, SpinorField, CliffordField
export MeshFunction, GradedField, QuaternionField # PhasorField
export Section, FiberBundle, AbstractFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype

abstract type AbstractFiber <: Number end
Base.@pure isfiber(::AbstractFiber) = true
Base.@pure isfiber(::Any) = false

struct Section{B,F} <: AbstractFiber
    v::Pair{B,F}
    Section(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    Section(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    Section(b::B,f::Section{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    Section(b::Section{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

Section(b,f::Function) = b ↦ f(b)

base(s::Section) = s.v.first
fiber(s::Section) = s.v.second
basetype(::Section{B}) where B = B
fibertype(::Section{B,F} where B) where F = F
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

Base.:+(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)+fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:-(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)-fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:<(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)<fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:>(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)>fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:<<(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)<<fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:>>(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)>>fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:&(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)&fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Grassmann.:∧(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)∧fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Grassmann.:∨(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)∨fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Grassmann.contraction(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),Grassmann.contraction(fiber(a),fiber(b))) : error("Section $(base(a)) ≠ $(base(b))")
Base.:*(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)*fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:/(a::Section{R},b::Section{R}) where R = base(a)==base(b) ? Section(base(a),fiber(a)/fiber(b)) : error("Section $(base(a)) ≠ $(base(b))")
Base.:*(a::Number,b::Section) = base(b) ↦ (a*fiber(b))
Base.:*(a::Section,b::Number) = base(a) ↦ (fiber(a)*b)
Base.:/(a::Section,b::Number) = base(a) ↦ (fiber(a)/b)
Base.:-(s::Section) = base(s) ↦ -(fiber(s))
Base.:!(s::Section) = base(s) ↦ !(fiber(s))
Base.:~(s::Section) = base(s) ↦ ~(fiber(s))
for fun ∈ (:inv,:exp,:log,:abs,:sqrt,:real,:imag,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2,:conj)
    @eval Base.$fun(s::Section) = base(s) ↦ $fun(fiber(s))
end
Grassmann.reverse(s::Section) = base(s) ↦ reverse(fiber(s))
Grassmann.involute(s::Section) = base(s) ↦ involute(fiber(s))
Grassmann.clifford(s::Section) = base(s) ↦ clifford(fiber(s))
Grassmann.even(s::Section) = base(s) ↦ even(fiber(s))
Grassmann.odd(s::Section) = base(s) ↦ odd(fiber(s))
Grassmann.scalar(s::Section) = base(s) ↦ scalar(fiber(s))
Grassmann.vector(s::Section) = base(s) ↦ vector(fiber(s))
Grassmann.bivector(s::Section) = base(s) ↦ bivector(fiber(s))
Grassmann.volume(s::Section) = base(s) ↦ volume(fiber(s))
Grassmann.value(s::Section) = base(s) ↦ value(fiber(s))
Grassmann.curl(s::Section) = bas(s) ↦ curl(fiber(s))
Grassmann.∂(s::Section) = base(s) ↦ ∂(fiber(s))
Grassmann.d(s::Section) = base(s) ↦ d(fiber(s))
Grassmann.:⋆(s::Section) = base(s) ↦ ⋆(fiber(s))
LinearAlgebra.norm(s::Section) = base(s) ↦ norm(fiber(s))
(V::Submanifold)(s::Section) = base(a) ↦ V(fiber(s))

abstract type FiberBundle{E,N} <: AbstractArray{E,N} end
Base.@pure isfiberbundle(::FiberBundle) = true
Base.@pure isfiberbundle(::Any) = false

struct TensorField{B,T,F,N} <: FiberBundle{Section{B,F},N}
    dom::T
    cod::Array{F,N}
    TensorField{B}(dom::T,cod::Array{F,N}) where {B,T,F,N} = new{B,T,F,N}(dom,cod)
    TensorField(dom::T,cod::Array{F,N}) where {N,B,T<:AbstractArray{B,N},F} = new{B,T,F,N}(dom,cod)
    TensorField(dom::T,cod::Vector{F}) where {T<:ChainBundle,F} = new{eltype(value(points(dom))),T,F,1}(dom,cod)
end

#const ParametricMesh{B<:Chain,T<:AbstractVector{B},F} = TensorField{B,T,F,1}
const MeshFunction{B,T<:ChainBundle,F<:Real} = TensorField{B,T,F,1}
const ElementFunction{B,T<:AbstractVector{B},F<:Real} = TensorField{B,T,F,1}
const IntervalMap{B<:Real,T<:AbstractVector{B},F} = TensorField{B,T,F,1}
const RealFunction{B<:Real,T<:AbstractVector{B},F<:Union{Real,Single,Chain{V,G,<:Real,1} where {V,G}}} = ElementFunction{B,T,F}
const PlaneCurve{B<:Real,T<:AbstractVector{B},F<:Chain{V,G,Q,2} where {V,G,Q}} = TensorField{B,T,F,1}
const SpaceCurve{B<:Real,T<:AbstractVector{B},F<:Chain{V,G,Q,3} where {V,G,Q}} = TensorField{B,T,F,1}
const GridSurface{B,T<:AbstractMatrix{B},F<:Real} = TensorField{B,T,F,2}
const GridParametric{B,T<:AbstractArray{B},F<:Real,N} = TensorField{B,T,F,N}
const ComplexMapping{B,T,F<:Complex,N} = TensorField{B,T,F,N}
const ComplexMap{B,T,F<:Couple,N} = TensorField{B,T,F,N}
const GradedField{G,B,T,F<:Chain{V,G} where V,N} = TensorField{B,T,F,N}
const VectorField = GradedField{1}
const ScalarField{B,T,F<:Single,N} = TensorField{B,T,F,N}
const SpinorField{B,T,F<:Spinor,N} = TensorField{B,T,F,N}
#const PhasorField{B,T,F<:Phasor,N} = TensorField{B,T,F,N}
const QuaternionField{B,T,F<:Quaternion,N} = TensorField{B,T,F,N}
const CliffordField{B,T,F<:Multivector,N} = TensorField{B,T,F,N}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

TensorField(dom::AbstractArray,fun::Function) = dom → fun.(dom)
TensorField(dom::ChainBundle,fun::Function) = dom → fun.(value(points(dom)))

←(F,B) = B → F
const → = TensorField
base(t::TensorField) = t.dom
fiber(t::TensorField) = t.cod
basetype(::TensorField{B}) where B = B
fibertype(::TensorField{B,T,F} where {B,T}) where F = F

Base.size(m::TensorField) = size(m.cod)
Base.resize!(m::TensorField,i) = (resize!(domain(m),i),resize(codomain(m),i))
@pure Base.eltype(::Type{TensorField{B,T,F}}) where {B,T,F} = Section{B,F}
Base.broadcast(f,t::TensorField) = domain(t) → f.(codomain(t))
Base.getindex(m::TensorField,i::Vararg{Int}) = getindex(domain(m),i...) ↦ getindex(codomain(m),i...)
Base.getindex(m::ElementFunction{R,<:ChainBundle} where R,i::Vararg{Int}) = getindex(value(points(domain(m))),i...) ↦ getindex(codomain(m),i...)
Base.setindex!(m::TensorField{B,<:AbstractRange{B}},s::Section,i::Vararg{Int}) where B = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,<:AbstractVector{B},F},s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::TensorField{B,<:Vector{B}},s::Section,i::Vararg{Int}) where B
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

function (m::IntervalMap{Float64,Vector{Float64}})(t)
    i = searchsortedfirst(domain(m),t)-1
    m.cod[i]+(t-m.dom[i])/(m.dom[i+1]-m.dom[i])*(m.cod[i+1]-m.cod[i])
end
function (m::IntervalMap{Float64,Vector{Float64}})(t::Vector,d=diff(m.cod)./diff(m.dom))
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

Base.:^(t::TensorField,n::Int) = domain(t) → codomain(t).^n
Base.:^(t::TensorField,n::Number) = domain(t) → codomain(t).^n
Base.:*(n::Number,t::TensorField) = domain(t) → n*codomain(t)
Base.:*(t::TensorField,n::Number) = domain(t) → codomain(t)*n
Base.:/(n::Number,t::TensorField) = domain(t) → (n./codomain(t))
Base.:/(t::TensorField,n::Number) = domain(t) → (codomain(t)/n)
Base.:+(n::Number,t::TensorField) = domain(t) → (n.+codomain(t))
Base.:+(t::TensorField,n::Number) = domain(t) → (codomain(t).+n)
Base.:-(n::Number,t::TensorField) = domain(t) → (n.-codomain(t))
Base.:-(t::TensorField,n::Number) = domain(t) → (codomain(t).-n)
Base.:*(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).*codomain(b))
Base.:/(a::TensorField,t::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a)./codomain(b))
Base.:+(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).+codomain(b))
Base.:-(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).-codomain(b))
Base.:>(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).>codomain(b))
Base.:<(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).<codomain(b))
Base.:>>(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).>>codomain(b))
Base.:<<(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).<<codomain(b))
Base.:&(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).&codomain(b))
Grassmann.:∧(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).∧codomain(b))
Grassmann.:∨(a::TensorField,t::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).∨codomain(b))
Grassmann.:⋅(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).⋅codomain(b))
Base.:-(t::TensorField) = base(t) ↦ -(fiber(t))
Base.:!(t::TensorField) = base(t) ↦ !(fiber(t))
Base.:~(t::TensorField) = base(t) ↦ ~(fiber(t))
for fun ∈ (:inv,:exp,:log,:abs,:sqrt,:real,:imag,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2,:conj)
    @eval Base.$fun(t::TensorField) = domain(t) → $fun.(codomain(t))
end
Grassmann.reverse(t::TensorField) = domain(t) → reverse.(codomain(t))
Grassmann.involute(t::TensorField) = domain(t) → involute.(codomain(t))
Grassmann.clifford(t::TensorField) = domain(t) → clifford.(codomain(t))
Grassmann.even(t::TensorField) = domain(t) → even.(codomain(t))
Grassmann.odd(t::TensorField) = domain(t) → odd.(codomain(t))
Grassmann.scalar(t::TensorField) = domain(t) → scalar.(codomain(t))
Grassmann.vector(t::TensorField) = domain(t) → vector.(codomain(t))
Grassmann.bivector(t::TensorField) = domain(t) → bivector.(codomain(t))
Grassmann.volume(t::TensorField) = domain(t) → volume.(codomain(t))
Grassmann.value(t::TensorField) = domain(t) → value.(codomain(t))
absvalue(t::TensorField) = domain(t) → value.(abs.(codomain(t)))
Grassmann.curl(t::TensorField) = domain(t) → curl.(codomain(t))
Grassmann.∂(t::TensorField) = domain(t) → ∂.(codomain(t))
Grassmann.d(t::TensorField) = domain(t) → d.(codomain(t))
Grassmann.:⋆(t::TensorField) = domain(t) → .⋆(codomain(t))
LinearAlgebra.norm(t::TensorField) = domain(t) → norm.(codomain(t))
(V::Submanifold)(t::TensorField) = domain(t) → V.(codomain(t))

checkdomain(a::TensorField,b::TensorField) = domain(a)≠domain(b) ? error("TensorField domains not equal") : true

centraldiff_fast(f::Vector,dt::Float64,l=length(f)) = [centraldiff_fast(i,f,l)/centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(f::Vector,dt::Vector,l=length(f)) = [centraldiff_fast(i,f,l)/dt[i] for i ∈ 1:l]
centraldiff_fast(f::Vector,l=length(f)) = [centraldiff_fast(i,f,l) for i ∈ 1:l]
function centraldiff_fast(i::Int,f::Vector{<:Chain},l=length(f))
    if isone(i) # 4f[2]-f[3]-3f[1]
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l # 3f[end]-4f[end-1]+f[end-2]
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    else
        f[i+1]-f[i-1]
    end
end
centraldiff_fast(f::StepRangeLen,l=length(f)) = [centraldiff_fast(i,Float64(f.step),l) for i ∈ 1:l]
centraldiff_fast(dt::Float64,l::Int) = [centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(i::Int,dt::Float64,l::Int) = i∈(1,l) ? 6dt : 2dt
#centraldiff_fast(i::Int,dt::Float64,l::Int) = 2dt

#=centraldiff(f::Vector,dt::Float64,l=length(f)) = [centraldiff(i,f,l)/centraldiff(i,dt,l) for i ∈ 1:l]
centraldiff(f::Vector,dt::Vector,l=length(f)) = [centraldiff(i,f,l)/dt[i] for i ∈ 1:l]
centraldiff(f::Vector,l=length(f)) = [centraldiff(i,f,l) for i ∈ 1:l]
function centraldiff(i::Int,f::Vector,l::Int=length(f))
    if isone(i)
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    elseif i==2
        6f[3]-f[4]-3f[2]-2f[1]
    elseif i==l-1
        3f[end-1]-6f[end-2]+f[end-3]+2f[end]
    else
        f[i-2]+8f[i+1]-8f[i-1]-f[i+2]
    end
end=#

export Grid

struct Grid{N,T,A<:AbstractArray{T,N}}
    v::A
end

Base.size(m::Grid) = size(m.v)

@generated function Base.getindex(g::Grid{M},j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(g.v,$([k≠N ? :(i[$k]) : :(i[$k]+j) for k ∈ 1:M]...)))
end

centraldiffdiff(f,dt,l) = centraldiff(centraldiff(f,dt,l),dt,l)
centraldiffdiff(f,dt) = centraldiffdiff(f,dt,size(f))
centraldiff(f::AbstractArray,args...) = centraldiff(Grid(f),args...)

centraldiff(f::Grid{1},dt::Float64,l=size(f.v)) = [centraldiff(f,l,i)/centraldiff(i,dt,l) for i ∈ 1:l]
centraldiff(f::Grid{1},dt::Vector,l=size(f.v)) = [centraldiff(f,l,i)/dt[i] for i ∈ 1:l[1]]
centraldiff(f::Grid{1},l=size(f.v)) = [centraldiff(f,l,i) for i ∈ 1:l[1]]

centraldiff(f::Grid{2},dt::AbstractMatrix,l::Tuple=size(f.v)) = [Chain(centraldiff(f,l,i,j).v./(dt[i,j].v)) for i ∈ 1:l[1], j ∈ 1:l[2]]
centraldiff(f::Grid{2},l::Tuple=size(f.v)) = [centraldiff(f,l,i,j) for i ∈ 1:l[1], j ∈ 1:l[2]]

centraldiff(f::Grid{3},dt::AbstractArray{T,3} where T,l::Tuple=size(f.v)) = [Chain(centraldiff(f,l,i,j,k).v./(dt[i,j,k].v)) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]
centraldiff(f::Grid{3},l::Tuple=size(f.v)) = [centraldiff(f,l,i,j,k) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]

centraldiff(f::Grid{1},l,i::Int) = centraldiff(f,l[1],Val(1),i)
@generated function centraldiff(f::Grid{M},l,i::Vararg{Int}) where M
    :(Chain($([:(centraldiff(f,l[$k],Val($k),i...)) for k ∈ 1:M]...)))
end
function centraldiff(f::Grid,l,k::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
    if isone(i[N])
        18f[1,k,i...]-9f[2,k,i...]+2f[3,k,i...]-11f.v[i...]
    elseif i[N]==l
        11f.v[i...]-18f[-1,k,i...]+9f[-2,k,i...]-2f[-3,k,i...]
    elseif i[N]==2
        6f[1,k,i...]-f[2,k,i...]-3f.v[i...]-2f[-1,k,i...]
    elseif i[N]==l-1
        3f.v[i...]-6f[-1,k,i...]+f[-2,k,i...]+2f[1,k,i...]
    else
        f[-2,k,i...]+8f[1,k,i...]-8f[-1,k,i...]-f[2,k,i...]
    end
end

centraldiff(f::StepRangeLen,l=length(f)) = [centraldiff(i,Float64(f.step),l) for i ∈ 1:l]
centraldiff(dt::Float64,l::Int) = [centraldiff(i,dt,l) for i ∈ 1:l]
function centraldiff(i::Int,dt::Float64,l::Int)
    if i∈(1,2,l-1,l)
        6dt
    else
        12dt
    end
end

function centraldiff_fast(i::Int,h::Int,f::Vector{<:Chain},l=length(f))
    if isone(i) # 4f[2]-f[3]-3f[1]
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l # 3f[end]-4f[end-1]+f[end-2]
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    else
        (i-h<1)||(i+h)>l ? centraldiff_(i,h-1,f,l) : f[i+h]-f[i-h]
    end
end

qnumber(n,q) = (q^n-1)/(q-1)
qfactorial(n,q) = prod(cumsum([q^k for k ∈ 0:n-1]))
qbinomial(n,k,q) = qfactorial(n,q)/(qfactorial(n-k,q)*qfactorial(k,q))

richardson(k) = [(-1)^(j+k)*2^(j*(1+j))*qbinomial(k,j,4)/prod([4^n-1 for n ∈ 1:k]) for j ∈ k:-1:0]

export arclength, trapz, linetrapz
export centraldiff, tangent, unittangent, speed, normal, unitnormal

arclength(f::Vector) = sum(abs.(diff(f)))
function arclength(f::IntervalMap)
    int = cumsum(abs.(diff(codomain(f))))
    pushfirst!(int,zero(eltype(int)))
    domain(f) → int
end # trapz(speed(f))
function trapz(f::IntervalMap,d=diff(domain(f)))
    int = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    domain(f) → int
end
function trapz(f::Vector,h::Float64)
    int = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return int
end
function linetrapz(γ::IntervalMap,f::Function)
    trapz(domain(γ)→(f.(codomain(γ)).⋅codomain(tangent(γ))))
end
function tangent(f::IntervalMap,d=centraldiff(domain(f)))
    domain(f) → centraldiff(codomain(f),d)
end
function tangent(f::GridParametric,d=centraldiff(domain(f)))
    domain(f) → centraldiff(Grid(codomain(f)),d)
end
function tangent(f::MeshFunction)
    domain(f) → interp(domain(f),gradient(domain(f),codomain(f)))
end
function unittangent(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → (t./abs.(t))
end
function unittangent(f::MeshFunction)
    t = interp(domain(f),gradient(domain(f),codomain(f)))
    domain(f) → (t./abs.(t))
end
function speed(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → abs.(t)
end
function normal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → centraldiff(t,d)
end
function unitnormal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    domain(f) → (n./abs.(n))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, curvaturebivector, torsion, curvaturetrivector, trihedron, frenet, wronskian, bishoppolar, bishop, curvaturetorsion

function curvature(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    domain(f) → (abs.(centraldiff(t./a,d))./a)
end
function radius(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    domain(f) → (a./abs.(centraldiff(t./a,d)))
end
function osculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (t.∧n)
end
function unitosculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → ((t./abs.(t)).∧(n./abs.(n)))
end
function binormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (.⋆(t.∧n))
end
function unitbinormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    a = t./abs.(t)
    n = centraldiff(t./abs.(t),d)
    domain(f) → (.⋆(a.∧(n./abs.(n))))
end
function curvaturebivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    a=abs.(t); ut=t./a
    domain(f) → (abs.(centraldiff(ut,d))./a.*(ut.∧(n./abs.(n))))
end
function torsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    domain(f) → ((b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
function curvaturetrivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t); ut=t./a
    domain(f) → ((abs.(centraldiff(ut,d)./a).^2).*(b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
#torsion(f::TensorField,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),a=abs.(t)) = domain(f) → (abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
function trihedron(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    domain(f) → Chain.(ut,un,.⋆(ut.∧un))
end
function frenet(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    domain(f) → centraldiff(Chain.(ut,un,.⋆(ut.∧un)),d)
end
function wronskian(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (f.cod.∧t.∧n)
end

#???
function compare(f::TensorField)#???
    d = centraldiff(f.dom)
    t = centraldiff(f.cod,d)
    n = centraldiff(t,d)
    domain(f) → (centraldiff(t./abs.(t)).-n./abs.(t))
end #????

function curvaturetorsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a = abs.(t)
    domain(f) → Chain.(value.(abs.(centraldiff(t./a,d))./a),getindex.((b.∧centraldiff(n,d))./abs.(.⋆b).^2,1))
end

function bishoppolar(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    domain(f) → Chain.(κ,θ)
end
function bishop(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    domain(f) → Chain.(κ.*cos.(θ),κ.*sin.(θ))
end
#bishoppolar(f::TensorField) = domain(f) → Chain.(value.(curvature(f).cod),getindex.(trapz(torsion(f)).cod,1))
#bishop(f::TensorField,κ=value.(curvature(f).cod),θ=getindex.(trapz(torsion(f)).cod,1)) = domain(f) → Chain.(κ.*cos.(θ),κ.*sin.(θ))

export ProductSpace, RealRegion, Interval, Rectangle, Hyperrectangle, ⧺, ⊕
import DirectSum: ⊕

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
RealRegion(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{ℝ^N,T,N}(v)
ProductSpace{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
ProductSpace(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{ℝ^N,T,N}(v)

@generated Base.size(m::RealRegion{V}) where V = :(($([:(size(m.v[$i])...) for i ∈ 1:mdims(V)]...),))
@generated Base.getindex(m::RealRegion{V,T,N},i::Vararg{Int}) where {V,T,N} = :(Chain{V,1,T}($([:(m.v[$j][i[$j]]) for j ∈ 1:N]...)))
@pure Base.eltype(::Type{ProductSpace{V,T,N}}) where {V,T,N} = Chain{V,1,T,N}

centraldiff(f::RealRegion) = ProductSpace(centraldiff.(f.v))

(m::TensorField{R,<:Rectangle} where R)(x,y) = m(Chain(x,y))
function (m::TensorField{R,<:Rectangle} where R)(t::Chain)
    x,y = domain(m).v[1],domain(m).v[2]
    i,j = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1
    f1 = m.cod[i,j]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j]-m.cod[i,j])
    f2 = m.cod[i,j+1]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j+1]-m.cod[i,j+1])
    f1+(t[2]-y[i])/(y[i+1]-y[i])*(f2-f1)
end

(m::TensorField{R,<:Hyperrectangle} where R)(x,y,z) = m(Chain(x,y,z))
function (m::TensorField{R,<:Hyperrectangle} where R)(t::Chain)
    x,y,z = domain(m).v[1],domain(m).v[2],domain(m).v[3]
    i,j,k = searchsortedfirst(x,t[1])-1,searchsortedfirst(y,t[2])-1,searchsortedfirst(z,t[3])-1
    f1 = m.cod[i,j,k]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j,k]-m.cod[i,j,k])
    f2 = m.cod[i,j+1,k]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j+1,k]-m.cod[i,j+1,k])
    g1 = f1+(t[2]-y[i])/(y[i+1]-y[i])*(f2-f1)
    f3 = m.cod[i,j,k+1]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j,k+1]-m.cod[i,j,k+1])
    f4 = m.cod[i,j+1,k+1]+(t[1]-x[i])/(x[i+1]-x[i])*(m.cod[i+1,j+1,k+1]-m.cod[i,j+1,k+1])
    g2 = f3+(t[2]-y[i])/(y[i+1]-y[i])*(f4-f3)
    g1+(t[3]-z[i])/(z[i+1]-z[i])*(g2-g1)
end

⊕(a::AbstractVector{A},b::AbstractVector{B}) where {A<:Real,B<:Real} = RealRegion(Values(a,b))

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

include("constants.jl")
include("element.jl")

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        export linegraph, linegraph!
        for lines ∈ (:lines,:lines!,:linesegments,:linesegments!)
            @eval begin
                Makie.$lines(t::SpaceCurve;args...) = Makie.$lines(t.cod;color=value.(speed(t).cod),args...)
                Makie.$lines(t::PlaneCurve;args...) = Makie.$lines(t.cod;color=value.(speed(t).cod),args...)
                Makie.$lines(t::RealFunction;args...) = Makie.$lines(t.dom,getindex.(value.(t.cod),1);color=value.(speed(t).cod),args...)
                Makie.$lines(t::ComplexMap{B,<:AbstractVector{B}};args...) where B<:Real = Makie.$lines(real.(value.(t.cod)),imag.(value.(t.cod));color=value.(speed(t).cod),args...)
                Makie.$lines(t::ComplexMapping{B,<:AbstractVector{B}};args...) where B<:Real = Makie.$lines(real.(t.cod),imag.(t.cod);color=value.(speed(t).cod),args...)
            end
        end
        Makie.lines(t::TensorField{B,<:AbstractVector{B}};args...) where B<:Real = linegraph(t;args...)
        function linegraph(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:Real}
            x,y = domain(t),value.(codomain(t))
            display(Makie.lines(x,getindex.(y,1);args...))
            for i ∈ 2:binomial(mdims(codomain(t)),G)
                Makie.lines!(x,getindex.(y,i);args...)
            end
        end
        Makie.lines!(t::TensorField{B,<:AbstractVector{B}};args...) where B<:Real = linegraph(t;args...)
        function linegraph!(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:Real}
            x,y = domain(t),value.(codomain(t))
            display(Makie.lines!(x,getindex.(y,1);args...))
            for i ∈ 2:binomial(mdims(codomain(t)),G)
                Makie.lines!(x,getindex.(y,i);args...)
            end
        end
        Makie.volume(t::GridParametric{<:Chain,<:Hyperrectangle};args...) = Makie.volume(t.dom.v[1],t.dom.v[2],t.dom.v[3],t.cod;args...)
        Makie.volumeslices(t::GridParametric{<:Chain,<:Hyperrectangle};args...) = Makie.volumeslices(t.dom.v[1],t.dom.v[2],t.dom.v[3],t.cod;args...)
        Makie.surface(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.surface(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contour(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contour(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contourf(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contourf(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contour3d(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contour3d(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.heatmap(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.heatmap(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.wireframe(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.wireframe(t.dom.v[1],t.dom.v[2],t.cod;args...)
        #Makie.spy(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.spy(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.streamplot(f::Function,t::Rectangle;args...) = Makie.streamplot(f,t.v[1],t.v[2];args...)
        Makie.streamplot(f::Function,t::Hyperrectangle;args...) = Makie.streamplot(f,t.v[1],t.v[2],t.v[3];args...)
        Makie.streamplot(m::TensorField{R,<:RealRegion,<:Real} where R;args...) = Makie.streamplot(tangent(m);args...)
        Makie.streamplot(m::TensorField{R,<:ChainBundle,<:Real} where R,dims...;args...) = Makie.streamplot(tangent(m),dims...;args...)
        Makie.streamplot(m::VectorField{R,<:ChainBundle} where R,dims...;args...) = Makie.streamplot(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
        Makie.streamplot(m::VectorField{R,<:RealRegion} where R;args...) = Makie.streamplot(p->Makie.Point(m(Chain(p.data...))),domain(m).v...;args...)
        Makie.arrows(t::VectorField{<:Chain,<:Rectangle};args...) = Makie.arrows(t.dom.v[1],t.dom.v[2],getindex.(t.cod,1),getindex.(t.cod,2);args...)
        Makie.arrows!(t::VectorField{<:Chain,<:Rectangle};args...) = Makie.arrows!(t.dom.v[1],t.dom.v[2],getindex.(t.cod,1),getindex.(t.cod,2);args...)
        Makie.arrows(t::VectorField{<:Chain,<:Hyperrectangle};args...) = Makie.arrows(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
        Makie.arrows!(t::VectorField{<:Chain,<:Hyperrectangle};args...) = Makie.arrows!(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
        Makie.arrows(t::Rectangle,f::Function;args...) = Makie.arrows(t.v[1],t.v[2],f;args...)
        Makie.arrows!(t::Rectangle,f::Function;args...) = Makie.arrows!(t.v[1],t.v[2],f;args...)
        Makie.arrows(t::Hyperrectangle,f::Function;args...) = Makie.arrows(t.v[1],t.v[2],t.v[3],f;args...)
        Makie.arrows!(t::Hyperrectangle,f::Function;args...) = Makie.arrows!(t.v[1],t.v[2],t.v[3],f;args...)
        Makie.arrows(t::MeshFunction;args...) = Makie.arrows(points(domain(t)),codomain(t);args...)
        Makie.arrows!(t::MeshFunction;args...) = Makie.arrows!(points(domain(t)),codomain(t);args...)
        Makie.mesh(t::ElementFunction;args...) = Makie.mesh(domain(t);color=codomain(t),args...)
        Makie.mesh!(t::ElementFunction;args...) = Makie.mesh!(domain(t);color=codomain(t),args...)
        Makie.mesh(t::MeshFunction;args...) = Makie.mesh(domain(t);color=codomain(t),args...)
        Makie.mesh!(t::MeshFunction;args...) = Makie.mesh!(domain(t);color=codomain(t),args...)
        Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=codomain(t),args...)
        Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=codomain(t),args...)
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(t.cod,1),getindex.(t.cod,2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(t.dom,getindex.(value.(t.cod),1);args...)
        UnicodePlots.lineplot(t::ComplexMapping{B,<:AbstractVector{B}};args...) where B<:Real = UnicodePlots.lineplot(real.(t.cod),imag.(t.cod);args...)
        UnicodePlots.lineplot(t::ComplexMap{B,<:AbstractVector{B}};args...) where B<:Real = UnicodePlots.lineplot(real.(value.(t.cod)),imag.(value.(t.cod));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,<:AbstractVector{B}};args...) where {G,B<:Real} = UnicodePlots.lineplot(domain(t),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.spy(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.spy(t.cod;args...)
        UnicodePlots.heatmap(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.heatmap(t.cod;args...)
        #Base.show(io::IO,t::PlaneCurve) = UnicodePlots.lineplot(getindex.(t.cod,1),getindex.(t.cod,2))
        #Base.show(io::IO,t::RealFunction) = UnicodePlots.lineplot(t.dom,getindex.(value.(t.cod),1))
        #Base.show(io::IO,t::GridSurface{<:Chain,<:RealRegion}) = UnicodePlots.heatmap(t.cod)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMapping{B,<:AbstractVector{B}}) where B<:Real = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractVector{B}}) where B<:Real = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::GradedField{G,B,<:AbstractVector{B}}) where {G,B<:Real} = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::GridSurface{<:Chain,<:RealRegion}) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
    end
end

end # module
