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

export Values, odesolve
export initmesh, pdegrad

export ElementFunction, IntervalMap, PlaneCurve, SpaceCurve, GridSurface, GridParametric
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export RealFunction, ComplexMap, ComplexMapping, SpinorField, CliffordField
export MeshFunction, QuaternionField # PhasorField
export Section, FiberBundle

struct Section{R,T} <: Number
    v::Pair{R,T}
    Section(v::Pair{R,T}) where {R,T} = new{R,T}(v)
    Section(r::R,t::T) where {R,T} = new{R,T}(r=>t)
end

Base.:+(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second+b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Base.:-(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second-b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Grassmann.:∧(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second∧b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Grassmann.:∨(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second∨b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Grassmann.contraction(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,Grassmann.contraction(a.v.second,b.v.second)) : error("Section $(a.v.first) ≠ $(b.v.first)")
Base.:*(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second*b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Base.:/(a::Section{R},b::Section{R}) where R = a.v.first==b.v.first ? Section(a.v.first,a.v.second/b.v.second) : error("Section $(a.v.first) ≠ $(b.v.first)")
Base.:*(a::Number,b::Section) = Section(b.v.first,a*b.v.second)
Base.:*(a::Section,b::Number) = Section(a.v.first,a.v.second*b)
Base.:/(a::Section,b::Number) = Section(a.v.first,a.v.second/b)
Base.abs(a::Section) = Section(a.v.first,abs(a.v.second))
Grassmann.value(a::Section) = Section(a.v.first,value(a.v.second))

abstract type FiberBundle{T,N} <: AbstractArray{T,N} end
Base.@pure isfiber(::FiberBundle) = true
Base.@pure isfiber(::Any) = false

struct TensorField{R,B,T,N} <: FiberBundle{Section{R,T},N}
    dom::B
    cod::Array{T,N}
    TensorField{R}(dom::B,cod::Array{T,N}) where {R,B,T,N} = new{R,B,T,N}(dom,cod)
    TensorField(dom::B,cod::Array{T,N}) where {N,R,B<:AbstractArray{R,N},T} = new{R,B,T,N}(dom,cod)
    TensorField(dom::B,cod::Vector{T}) where {B<:ChainBundle,T} = new{eltype(value(points(dom))),B,T,1}(dom,cod)
end

#const ParametricMesh{T<:AbstractVector{<:Chain},S} = TensorField{T,S,1}
const MeshFunction{R,B<:ChainBundle,T<:Real} = TensorField{R,B,T,1}
const ElementFunction{R,B<:AbstractVector{R},T<:Real} = TensorField{R,B,T,1}
const IntervalMap{R<:Real,B<:AbstractVector{R},T} = TensorField{R,B,T,1}
const RealFunction{R<:Real,B<:AbstractVector{R},T<:Union{Real,Single,Chain{V,G,<:Real,1} where {V,G}}} = ElementFunction{R,B,T}
const PlaneCurve{R<:Real,B<:AbstractVector{R},T<:Chain{V,G,Q,2} where {V,G,Q}} = TensorField{R,B,T,1}
const SpaceCurve{R<:Real,B<:AbstractVector{R},T<:Chain{V,G,Q,3} where {V,G,Q}} = TensorField{R,B,T,1}
const GridSurface{R,B<:AbstractMatrix{R},T<:Real} = TensorField{R,B,T,2}
const GridParametric{R,B<:AbstractArray{R},T<:Real,N} = TensorField{R,B,T,N}
const ComplexMapping{R,B,T<:Complex,N} = TensorField{R,B,T,N}
const ComplexMap{R,B,T<:Couple,N} = TensorField{R,B,T,N}
const ScalarField{R,B,T<:Single,N} = TensorField{R,B,T,N}
const VectorField{R,B,T<:Chain{V,1} where V,N} = TensorField{R,B,T,N}
const SpinorField{R,B,T<:Spinor,N} = TensorField{R,B,T,N}
#const PhasorField{R,B,T<:Phasor,N} = TensorField{R,B,T,N}
const QuaternionField{R,B,T<:Quaternion,N} = TensorField{R,B,T,N}
const CliffordField{R,B,T<:Multivector,N} = TensorField{R,B,T,N}
const BivectorField{R,B,T<:Chain{V,2} where V,N} = TensorField{R,B,T,N}
const TrivectorField{R,B,T<:Chain{V,3} where V,N} = TensorField{R,B,T,N}

TensorField(dom::AbstractArray,fun::Function) = TensorField(dom,fun.(dom))
TensorField(dom::ChainBundle,fun::Function) = TensorField(dom,fun.(value(points(dom))))

Base.size(m::TensorField) = size(m.cod)
Base.getindex(m::TensorField,i::Vararg{Int}) = Section(getindex(domain(m),i...),getindex(codomain(m),i...))
Base.getindex(m::ElementFunction{R,<:ChainBundle} where R,i::Vararg{Int}) = Section(getindex(value(points(domain(m))),i...),getindex(codomain(m),i...))
@pure Base.eltype(::Type{TensorField{R,B,T}}) where {R,B,T} = Section{R,T}

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

function (m::TensorField{R,<:ChainBundle} where R)(t)
    i = domain(m)[findfirst(t,domain(m))]
    (codomain(m)[i])⋅(points(domain(m))[i]/t)
end

export domain, codomain
domain(t::TensorField) = t.dom
codomain(t::TensorField) = t.cod

Base.:*(n::Number,t::TensorField) = TensorField(domain(t),n*codomain(t))
Base.:*(t::TensorField,n::Number) = TensorField(domain(t),codomain(t)*n)
Base.:/(n::Number,t::TensorField) = TensorField(domain(t),n./codomain(t))
Base.:/(t::TensorField,n::Number) = TensorField(domain(t),codomain(t)/n)
Base.:+(n::Number,t::TensorField) = TensorField(domain(t),n.+codomain(t))
Base.:+(t::TensorField,n::Number) = TensorField(domain(t),codomain(t).+n)
Base.:-(n::Number,t::TensorField) = TensorField(domain(t),n.-codomain(t))
Base.:-(t::TensorField,n::Number) = TensorField(domain(t),codomain(t).-n)
Base.:*(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).*codomain(b))
Base.:/(a::TensorField,t::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a)./codomain(b))
Base.:+(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).+codomain(b))
Base.:-(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).-codomain(b))
Grassmann.:∧(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).∧codomain(b))
Grassmann.:∨(a::TensorField,t::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).∨codomain(b))
Grassmann.:⋅(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(t),codomain(a).⋅codomain(b))
Base.abs(t::TensorField) = TensorField(domain(t),abs.(codomain(t)))
Grassmann.value(t::TensorField) = TensorField(domain(t),value.(codomain(t)))
absvalue(t::TensorField) = TensorField(domain(t),value.(abs.(codomain(t))))

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
    TensorField(domain(f),int)
end # trapz(speed(f))
function trapz(f::IntervalMap,d=diff(domain(f)))
    int = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return TensorField(domain(f),int)
end
function trapz(f::Vector,h::Float64)
    int = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return int
end
function linetrapz(γ::IntervalMap,f::Function)
    trapz(TensorField(γ.dom,f.(codomain(γ)).⋅codomain(tangent(γ))))
end
function tangent(f::IntervalMap,d=centraldiff(domain(f)))
    TensorField(domain(f),centraldiff(codomain(f),d))
end
function tangent(f::GridParametric,d=centraldiff(domain(f)))
    TensorField(domain(f),centraldiff(Grid(codomain(f)),d))
end
function tangent(f::MeshFunction)
    TensorField(domain(f),interp(domain(f),gradient(domain(f),codomain(f))))
end
function unittangent(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    TensorField(domain(f),t./abs.(t))
end
function unittangent(f::MeshFunction)
    t = interp(domain(f),gradient(domain(f),codomain(f)))
    TensorField(domain(f),t./abs.(t))
end
function speed(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    TensorField(domain(f),abs.(t))
end
function normal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    TensorField(domain(f),centraldiff(t,d))
end
function unitnormal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    TensorField(domain(f),n./abs.(n))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, curvaturebivector, torsion, curvaturetrivector, trihedron, frenet, wronskian, bishoppolar, bishop, curvaturetorsion

function curvature(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    TensorField(f.dom,abs.(centraldiff(t./a,d))./a)
end
function radius(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    TensorField(f.dom,a./abs.(centraldiff(t./a,d)))
end
function osculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,t.∧n)
end
function unitosculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,(t./abs.(t)).∧(n./abs.(n)))
end
function binormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,.⋆(t.∧n))
end
function unitbinormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    a = t./abs.(t)
    n = centraldiff(t./abs.(t),d)
    TensorField(f.dom,.⋆(a.∧(n./abs.(n))))
end
function curvaturebivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    a=abs.(t); ut=t./a
    TensorField(f.dom,abs.(centraldiff(ut,d))./a.*(ut.∧(n./abs.(n))))
end
function torsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    TensorField(f.dom,(b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
function curvaturetrivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t); ut=t./a
    TensorField(f.dom,((abs.(centraldiff(ut,d)./a).^2).*(b.∧centraldiff(n,d))./abs.(.⋆b).^2))
end
#torsion(f::TensorField,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),a=abs.(t)) = TensorField(f.dom,abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
function trihedron(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    Chain.(ut,un,.⋆(ut.∧un))
end
function frenet(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    centraldiff(Chain.(ut,un,.⋆(ut.∧un)),d)
end
function wronskian(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    f.cod.∧t.∧n
end

#???
function compare(f::TensorField)#???
    d = centraldiff(f.dom)
    t = centraldiff(f.cod,d)
    n = centraldiff(t,d)
    centraldiff(t./abs.(t)).-n./abs.(t)
end #????

function curvaturetorsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a = abs.(t)
    TensorField(f.dom,Chain.(value.(abs.(centraldiff(t./a,d))./a),getindex.((b.∧centraldiff(n,d))./abs.(.⋆b).^2,1)))
end

function bishoppolar(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(f.dom,Chain.(κ,θ))
end
function bishop(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(f.dom,Chain.(κ.*cos.(θ),κ.*sin.(θ)))
end
#bishoppolar(f::TensorField) = TensorField(f.dom,Chain.(value.(curvature(f).cod),getindex.(trapz(torsion(f)).cod,1)))
#bishop(f::TensorField,κ=value.(curvature(f).cod),θ=getindex.(trapz(torsion(f)).cod,1)) = TensorField(f.dom,Chain.(κ.*cos.(θ),κ.*sin.(θ)))

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
        Makie.lines(t::SpaceCurve;args...) = Makie.lines(t.cod;color=value.(speed(t).cod),args...)
        Makie.lines(t::PlaneCurve;args...) = Makie.lines(t.cod;color=value.(speed(t).cod),args...)
        Makie.lines(t::RealFunction;args...) = Makie.lines(t.dom,getindex.(value.(t.cod),1);color=value.(speed(t).cod),args...)
        Makie.linesegments(t::SpaceCurve;args...) = Makie.linesegments(t.cod;color=value.(speed(t).cod),args...)
        Makie.linesegments(t::PlaneCurve;args...) = Makie.linesegments(t.cod;color=value.(speed(t).cod),args...)
        Makie.linesegments(t::RealFunction;args...) = Makie.linesegments(t.dom,getindex.(value.(t.cod),1);color=value.(speed(t).cod),args...)
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
        UnicodePlots.contourplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.spy(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.spy(t.cod;args...)
        UnicodePlots.heatmap(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.heatmap(t.cod;args...)
    end
end

end # module
