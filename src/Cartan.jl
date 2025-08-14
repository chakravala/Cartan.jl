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

using SparseArrays, LinearAlgebra, ElasticArrays, Base.Threads, Grassmann
import Grassmann: value, vector, valuetype, tangent, istangent, Derivation, radius, ⊕
import Grassmann: realvalue, imagvalue, points, metrictensor, metricextensor
import Grassmann: Values, Variables, FixedVector, list, volume, compound
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Base: @pure, OneTo, getindex
import LinearAlgebra: cross
import ElasticArrays: resize_lastdim!

export Values, Derivation
export initmesh, pdegrad, det, graphbundle

macro elastic(itr)
    :(elastic($(itr.args[1]),$(itr.args[2])))
end
function elastic(T,itr)
    out = ElasticArray{T}(undef,size(itr))
    for e ∈ enumerate(itr)
        @inbounds out[e[1]] = e[2]
    end
    return out
end

include("topology.jl")
include("quotient.jl")
include("fiber.jl")

export ElementMap, SimplexMap, FaceMap, Components
export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid, Variation
export RealFunction, ComplexMap, SpinorField, CliffordField
export ScalarMap, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap, AbstractCurve
export metrictensorfield, metricextensorfield, polarize, complexify, vectorize
export leaf, alteration, variation, modification, alteration!, variation!, modification!

# TensorField

"""
    TensorField{B,F,N} <: FiberBundle{LocalTensor{B,F},N}

Defines a section of a `FrameBundle` with `eltype` of `LocalTensor{B,F}` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{B,N}
fiber(s) # ::AbstractArray{F,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # B
fibertype(s) # F
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `TensorField`, such as `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
Due to the versatility of the `TensorField` type instances, it's possible to disambiguate them into these type alias specifications with associated methods:
`ElementMap`, `SimplexMap`, `FaceMap`, `IntervalMap`, `RectangleMap`, `HyperrectangleMap`, `ParametricMap`, `Variation`, `RealFunction`, `PlaneCurve`, `SpaceCurve`, `AbstractCurve`,
`SurfaceGrid`, `VolumeGrid`, `ScalarGrid`, `DiagonalField`, `EndomorphismField`, `OutermorphismField`, `CliffordField`, `QuaternionField`, `ComplexMap`, `PhasorField`, `SpinorField`, `GradedField{G} where G`, `ScalarField`, `VectorField`, `BivectorField`, `TrivectorField`.

In the `Cartan` package, a technique is employed where a `TensorField` is constructed from an interval, product manifold, or topology, to generate an algebra of sections which can be used to compose parametric maps on manifolds.
Constructing a `TensorField` can be accomplished in various ways,
there are explicit techniques to construct a `TensorField` as well as implicit methods.
Additional packages such as `Adapode` build on the `TensorField` concept by generating them from differential equations.
Many of these methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
"""
struct TensorField{B,F,N,M<:FrameBundle{B,N},A<:AbstractArray{F,N}} <: FiberBundle{LocalTensor{B,F},N}
    dom::M
    cod::A
    function TensorField(dom::M,cod::A) where {B,F,N,M<:FrameBundle{B,N},A<:AbstractArray{F,N}}
        new{B,F,N,M,A}(dom,cod)
    end
end

#TensorField(dom::FrameBundle,cod::Array) = TensorField(dom,ElasticArray(cod))
function TensorField(dom::AbstractArray{P,N},cod::AbstractArray,met::AbstractArray=Global{N}(InducedMetric())) where {N,P}
    TensorField(GridBundle(PointArray(0,dom,met)),cod)
end
TensorField(dom::PointArray,cod::AbstractArray) = TensorField(GridBundle(dom),cod)
TensorField(dom,cod::AbstractArray,met::FiberBundle) = TensorField(dom,cod,fiber(met))
TensorField(dom::FrameBundle,cod::FrameBundle) = TensorField(dom,points(cod))
TensorField(a::TensorField,b::TensorField) = TensorField(fiber(a),fiber(b))

#const ParametricMesh{B,F,P<:AbstractVector{<:Chain},A} = TensorField{B,F,1,P,A}
const ScalarMap{B,F<:AbstractReal,P<:SimplexBundle,A} = TensorField{B,F,1,P,A}
const ElementMap{B,F,P<:ElementBundle,A} = TensorField{B,F,1,P,A}
const SimplexMap{B,F,P<:SimplexBundle,A} = TensorField{B,F,1,P,A}
const FaceMap{B,F,P<:FaceBundle,A} = TensorField{B,F,1,P,A}
#const ElementFunction{B,F<:AbstractReal,P<:AbstractVector,A} = TensorField{B,F,1,P,A}
const IntervalMap{B,F,P<:Interval,A} = TensorField{B,F,1,P,A}
const RectangleMap{B,F,P<:RealSpace{2},A} = TensorField{B,F,2,P,A}
const HyperrectangleMap{B,F,P<:RealSpace{3},A} = TensorField{B,F,3,P,A}
const ParametricMap{B,F,N,P<:RealSpace,A} = TensorField{B,F,N,P,A}
const Variation{B,F<:TensorField,N,P,A} = TensorField{B,F,N,P,A}
const RealFunction{B,F<:AbstractReal,P<:Interval,A} = TensorField{B,F,1,P,A}
const PlaneCurve{B,F<:Chain{V,G,Q,2} where {V,G,Q},P<:Interval,A} = TensorField{B,F,1,P,A}
const SpaceCurve{B,F<:Chain{V,G,Q,3} where {V,G,Q},P<:Interval,A} = TensorField{B,F,1,P,A}
const AbstractCurve{B,F<:Chain,P<:Interval,A} = TensorField{B,F,1,P,A}
const SurfaceGrid{B,F<:AbstractReal,P<:RealSpace{2},A} = TensorField{B,F,2,P,A}
const VolumeGrid{B,F<:AbstractReal,P<:RealSpace{3},A} = TensorField{B,F,3,P,A}
const ScalarGrid{B,F<:AbstractReal,N,P<:RealSpace{N},A} = TensorField{B,F,N,P,A}
#const GlobalFrame{B<:LocalFiber{P,<:TensorNested} where P,N} = GlobalSection{B,N}
const DiagonalField{B,F<:DiagonalOperator,N,P,A} = TensorField{B,F,N,P,A}
const EndomorphismField{B,F<:Endomorphism,N,P,A} = TensorField{B,F,N,P,A}
const OutermorphismField{B,F<:Outermorphism,N,P,A} = TensorField{B,F,N,P,A}
const CliffordField{B,F<:Multivector,N,P,A} = TensorField{B,F,N,P,A}
const QuaternionField{B,F<:Quaternion,N,P,A} = TensorField{B,F,N,P,A}
const ComplexMap{B,F<:AbstractComplex,N,P,A} = TensorField{B,F,N,P,A}
const PhasorField{B,F<:Phasor,N,P,A} = TensorField{B,F,N,P,A}
const SpinorField{B,F<:AbstractSpinor,N,P,A} = TensorField{B,F,N,P,A}
const GradedField{G,B,F<:Chain{V,G} where V,N,P,A} = TensorField{B,F,N,P,A}
const ScalarField{B,F<:AbstractReal,N,P,A} = TensorField{B,F,N,P,A}
const VectorField = GradedField{1}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

TensorField(dom::FrameBundle{B,N},fun::BitArray{N}) where {B,N} = TensorField(dom, Float64.(fun))
TensorField(dom,fun::TensorField) = TensorField(dom, fiber(fun))
TensorField(dom::TensorField,fun::AbstractArray) = TensorField(base(dom), fun)
TensorField(dom::TensorField,fun::Function) = TensorField(base(dom), fun)
TensorField(dom::TensorField,fun::Number) = TensorField(base(dom), fun)
TensorField(dom::AbstractArray,fun::Function) = TensorField(dom, fun.(dom))
TensorField(dom::FrameBundle,fun::Function) = fun.(dom)
TensorField(dom::AbstractArray,fun::Number) = TensorField(dom, fill(fun,size(dom)...))
TensorField(dom::AbstractArray) = TensorField(dom, dom)
TensorField(f::F,r::AbstractVector{<:Real}=-2π:0.0001:2π) where F<:Function = TensorField(r,vector.(f.(r)))
coordinatetype(t::TensorField) = basetype(t)
coordinatetype(t::Type{<:TensorField}) = basetype(t)
basetype(::TensorField{B}) where B = B
basetype(::Type{<:TensorField{B}}) where B = B
fibertype(::TensorField{B,F} where B) where F = F
fibertype(::Type{<:TensorField{B,F} where B}) where F = F
Base.broadcast(f,t::TensorField) = TensorField(base(t), f.(fiber(t)))
#TensorField(dom::SimplexBundle{1}) = TensorField(dom,getindex.(points(dom),2))

"""
    PrincipalFiber{M,G,N} <: FiberBundle{LocalPrincipal{M,G},N}

Defines a principal `FiberBundle` morphism type with `eltype` of `LocalPrincipal{M,G}`.
```Julia
coordinates(s) # ::AbstractArray{X,N}
principalbase(s) # ::AbstractArray{M,N}
principalfiber(s) # ::AbstractArray{G,N}
coordinatetype(s) # X
principalbasetype(s) # M
principalfibertype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `PrincipalFiber`, such as `isbundle`, `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
"""
struct PrincipalFiber{M,G,N,XM<:TensorField{X,M,N} where X,XG<:TensorField{X,G,N} where X} <: FiberBundle{LocalPrincipal{M,G},N}
    dom::XM
    cod::XG
end

coordinatetype(t::PrincipalFiber) = coordinatetype(base(t))
coordinatetype(t::Type{<:PrincipalFiber{M,G,N,XM} where {M,G,N}}) where XM = coordinatetype(XM)
basetype(::PrincipalFiber{M}) where M = M
basetype(::Type{<:PrincipalFiber{M}}) where M = M
fibertype(::PrincipalFiber{M,G} where M) where G = G
fibertype(::Type{<:PrincipalFiber{M,G} where M}) where G = G

for fun ∈ (:points,:metricextensor,:coordinates,:immersion,:vertices,:fullcoordinates,:metricextensorfield,:metrictensorfield)
    @eval begin
        $fun(t::TensorField) = $fun(base(t))
        $fun(t::PrincipalFiber) = $fun(base(t))
    end
end

←(F,B) = B → F
const → = TensorField
metricextensorfield(t::GridBundle) = TensorField(GridBundle(PointArray(0,points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::GridBundle) = TensorField(GridBundle(PointArray(0,points(t)),immersion(t)),metrictensor(t))
metricextensorfield(t::SimplexBundle) = TensorField(SimplexBundle(PointCloud(0,points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::SimplexBundle) = TensorField(SimplexBundle(PointCloud(0,points(t)),immersion(t)),metrictensor(t))
Grassmann.grade(::GradedField{G}) where G = G
Grassmann.antigrade(t::GradedField) = antigrade(fibertype(t))

resize(t::TensorField) = TensorField(resize(base(t)),fiber(t))

function resample(t::TensorField,i::NTuple)
    rg = resample(base(t),i)
    TensorField(rg,t.(points(rg)))
end

@pure Base.eltype(::Type{<:TensorField{B,F}}) where {B,F} = LocalTensor{B,F}
Base.getindex(m::TensorField,i::Vararg{Int}) = LocalTensor(getindex(base(m),i...), getindex(fiber(m),i...))
Base.getindex(m::TensorField,i::Vararg{Union{Int,Colon}}) = TensorField(base(m)(i...), getindex(fiber(m),i...))
#Base.setindex!(m::TensorField{B,F,1,<:Interval},s::LocalTensor,i::Vararg{Int}) where {B,F} = setindex!(fiber(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(fiber(m),s,i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,i::Int) where B = setindex!(fiber(m),fiber(s),:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,i::Int) where B = setindex!(fiber(m),fiber(s),:,:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,::Colon,i::Int) where B = setindex!(fiber(m),fiber(s),:,:,:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,::Colon,::Colon,i::Int) where B = setindex!(fiber(m),fiber(s),:,:,:,:,i)
function Base.setindex!(m::TensorField{B,F,N,<:IntervalRange} where {B,F,N},s::LocalTensor,i::Vararg{Int})
    setindex!(fiber(m),fiber(s),i...)
    return s
end
function Base.setindex!(m::TensorField{B,F,N,<:AlignedSpace} where {B,F,N},s::LocalTensor,i::Vararg{Int})
    setindex!(fiber(m),fiber(s),i...)
    return s
end
function Base.setindex!(m::TensorField,s::LocalTensor,i::Vararg{Int})
    #setindex!(base(m),base(s),i...)
    setindex!(fiber(m),fiber(s),i...)
    return s
end

extract(x::AbstractVector,i) = (@inbounds x[i])
extract(x::AbstractMatrix,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,i]))
extract(x::AbstractArray{T,3} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,i]))
extract(x::AbstractArray{T,4} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,:,i]))
extract(x::AbstractArray{T,5} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,:,:,i]))

assign!(x::AbstractVector,i,s) = (@inbounds x[i] = s)
assign!(x::AbstractMatrix,i,s) = (@inbounds x[:,i] = s)
assign!(x::AbstractArray{T,3} where T,i,s) = (@inbounds x[:,:,i] = s)
assign!(x::AbstractArray{T,4} where T,i,s) = (@inbounds x[:,:,:,i] = s)
assign!(x::AbstractArray{T,5} where T,i,s) = (@inbounds x[:,:,:,:,i] = s)

Base.BroadcastStyle(::Type{<:TensorField{B,F,N,P,L}}) where {B,F,N,P,L} = Broadcast.ArrayStyle{TensorField{B,F,N,P,L}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,F,N,P,L}}}, ::Type{ElType}) where {B,F,N,P,L,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    TensorField(base(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Array{ElType,N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SimplexBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Vector{ElType}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{FaceBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Vector{ElType}, axes(bc)))
end

@findobject find_tf TensorField

@pure Base.eltype(::Type{<:PrincipalFiber{M,G}}) where {M,G} = LocalPrincipal{M,G}
Base.getindex(m::PrincipalFiber,i::Vararg{Int}) = LocalPrincipal(getindex(principalbase(m),i...), getindex(principalfiber(m),i...))
Base.getindex(m::PrincipalFiber,i::Vararg{Union{Int,Colon}}) = PrincipalFiber(base(m)(i...), getindex(fiber(m),i...))
Base.getindex(m::PrincipalFiber,i::TensorField) = PrincipalFiber(base(m).(i), fiber(m).(i))
#Base.setindex!(m::TensorField{M,G,1,<:Interval},s::LocalTensor,i::Vararg{Int}) where {M,G} = setindex!(principalfiber(m),fiber(s),i...)
Base.setindex!(m::PrincipalFiber{M,Gm} where Gm,s::G,i::Vararg{Int}) where {M,G} = setindex!(principalfiber(m),s,i...)
function Base.setindex!(m::PrincipalFiber,s::LocalTensor,i::Vararg{Int})
    setindex!(principalbase(m),base(s),i...)
    setindex!(principalfiber(m),fiber(s),i...)
    return s
end

Base.BroadcastStyle(::Type{<:PrincipalFiber{M,G,N,XM,XG}}) where {M,G,N,XM,XG} = Broadcast.ArrayStyle{PrincipalFiber{M,G,N,XM,XG}}()

#=function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PrincipalFiber{M,G,N,XM,XG}}}, ::Type{ElType}) where {M,G,N,XM,XG,ElType}
    # Scan the inputs for the PrincipalFiber:
    t = find_pf(bc)
    # Use the domain field of t to create the output
    #TensorField(base(t), similar(Array{basetype(ElType),N}, axes(bc)))
    TensorField(base(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end=#

@findobject find_pf PrincipalFiber

(m::TensorField{B,F,N,<:SimplexBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(coordinates(m)(i),fiber(m)[vertices(i)])
(m::TensorField{B,F,N,<:GridBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(base(m)(i),fiber(m))
for fun ∈ (:Open,:Mirror,:Clamped,:Torus,:Cylinder,:Wing,:Mobius,:Klein,:Cone,:Tube,:Ball,:Sphere,:Geographic,:Hopf)
    for top ∈ (Symbol(fun,:Topology),)
        @eval begin
            $top(m::TensorField{B,F,N,<:GridBundle} where {B,F,N}) = TensorField($top(base(m)),fiber(m))
            $top(m::GridBundle) = m($top(size(m)))
            $top(p::PointArray) = TensorField(GridBundle(p,$top(size(p))))
            $(Symbol(fun,:Parameter))(p::PointArray) = $top(p)
        end
    end
end

spacing(x::AbstractVector) = sum(norm.(diff(fiber(x))))/(length(x)-1)
spacing(x::AbstractArray{T,N}) where {T,N} = minimum(spacing.(Ref(x),list(1,N)))
function spacing(x::AbstractArray,i)
    n = norm.(diff(fiber(x),dims=i))
    sum(n)/length(n)
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
Base.:^(t::TensorField,n::Int) = TensorField(base(t), .^(fiber(t),n,refmetric(t)))
for op ∈ (:+,:-,:&,:∧,:∨)
    let bop = op ∈ (:∧,:∨) ? :(Grassmann.$op) : :(Base.$op)
        @eval begin
            $bop(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(base(a), $op.(fiber(a),fiber(b)))
            $bop(a::TensorField,b::Number) = TensorField(base(a), $op.(fiber(a),Ref(b)))
            $bop(a::Number,b::TensorField) = TensorField(base(b), $op.(Ref(a),fiber(b)))
        end
    end
end
(m::TensorNested)(t::TensorField) = TensorField(base(t), m.(fiber(t)))
(m::TensorField{B,<:TensorNested} where B)(t::TensorField) = m⋅t
@inline Base.:<<(a::FiberBundle,b::FiberBundle) = contraction(b,~a)
@inline Base.:>>(a::FiberBundle,b::FiberBundle) = contraction(~a,b)
@inline Base.:<(a::FiberBundle,b::FiberBundle) = contraction(b,a)
Base.sign(a::TensorField) = TensorField(base(a), sign.(fiber(Real(a))))
Base.inv(a::TensorField{B,<:Real} where B) = TensorField(base(a), inv.(fiber(a)))
Base.inv(a::TensorField{B,<:Complex} where B) = TensorField(base(a), inv.(fiber(a)))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:/(a::TensorField,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a)./fiber(b))
Base.:/(a::TensorField,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a)./fiber(b))
LinearAlgebra.:×(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a), .⋆(fiber(a).∧fiber(b),refmetric(base(a))))
Grassmann.compound(t::TensorField,i::Val) = TensorField(base(t), compound.(fiber(t),i))
Grassmann.compound(t::TensorField,i::Int) = TensorField(base(t), compound.(fiber(t),Val(i)))
Grassmann.eigen(t::TensorField,i::Val) = TensorField(base(t), eigen.(fiber(t),i))
Grassmann.eigen(t::TensorField,i::Int) = TensorField(base(t), eigen.(fiber(t),Val(i)))
Grassmann.eigvals(t::TensorField,i::Val) = TensorField(base(t), eigvals.(fiber(t),i))
Grassmann.eigvals(t::TensorField,i::Int) = TensorField(base(t), eigvals.(fiber(t),Val(i)))
Grassmann.eigvecs(t::TensorField,i::Val) = TensorField(base(t), eigvecs.(fiber(t),i))
Grassmann.eigvecs(t::TensorField,i::Int) = TensorField(base(t), eigvecs.(fiber(t),Val(i)))
Grassmann.eigpolys(t::TensorField,G::Val) = TensorField(base(t), eigpolys.(fiber(t),G))
for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:contraction,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
    let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
    @eval begin
        $bop(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a),Grassmann.$mop.(fiber(a),fiber(b),refmetric(base(a))))
        $bop(a::Number,b::TensorField) = TensorField(base(b), Grassmann.$op.(a,fiber(b)))
        $bop(a::TensorField,b::Number) = TensorField(base(a), Grassmann.$op.(fiber(a),b,$((op≠:^ ? () : (:(refmetric(base(a))),))...)))
    end end
end
for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad,:transpose)
    @eval Base.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
end
for fun ∈ (:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:inv)
    @eval Base.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t),ref(metricextensor(t))))
end
for fun ∈ (:reverse,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:complementleft,:realvalue,:imagvalue,:outermorphism,:Outermorphism,:DiagonalOperator,:TensorOperator,:eigen,:eigvecs,:eigvals,:eigvalsreal,:eigvalscomplex,:eigvecsreal,:eigvecscomplex,:eigpolys,:pfaffian,:∧,:↑,:↓)
    @eval Grassmann.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
end
for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric,:unit)
    @eval Grassmann.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t),ref(metricextensor(t))))
end
for fun ∈ (:sum,:prod)
    @eval Base.$fun(t::TensorField) = LocalTensor(base(t)[end], $fun(fiber(t)))
end
for fun ∈ (:cumsum,:cumprod)
    @eval function Base.$fun(t::TensorField)
         out = $fun(fiber(t))
         pushfirst!(out,zero(eltype(out)))
         TensorField(base(t), out)
    end
end

Base.:/(a::TensorField,b::TensorAlgebra) = TensorField(base(a),./(fiber(a),b,refmetric(base(a))))
Grassmann.signbit(::TensorField) = false
#Base.inv(t::TensorField) = TensorField(fiber(t), base(t))
Base.diff(t::TensorField) = TensorField(diff(base(t)), diff(fiber(t)))
absvalue(t::TensorField) = TensorField(base(t), value.(abs.(fiber(t))))
LinearAlgebra.tr(t::TensorField) = TensorField(base(t), tr.(fiber(t)))
LinearAlgebra.det(t::TensorField) = TensorField(base(t), det.(fiber(t)))
LinearAlgebra.norm(t::TensorField) = TensorField(base(t), norm.(fiber(t)))
(V::Submanifold)(t::TensorField) = TensorField(base(t), V.(fiber(t)))
(::Type{T})(t::TensorField) where T<:Real = TensorField(base(t), T.(fiber(t)))
(::Type{Complex})(t::TensorField) = TensorField(base(t), Complex.(fiber(t)))
(::Type{Complex{T}})(t::TensorField) where T = TensorField(base(t), Complex{T}.(fiber(t)))
Grassmann.Phasor(s::TensorField) = TensorField(base(s), Phasor(fiber(s)))
Grassmann.Couple(s::TensorField) = TensorField(base(s), Couple(fiber(s)))

checkdomain(a::FiberBundle,b::FiberBundle) = base(a)≠base(b) ? error("GlobalFiber base not equal") : true

_aff(x::Chain{V,1,T,2}) where {V,T} = Chain{varmanifold(3)}(one(T),x[1],x[2])
_aff(x::Chain{V,1,T,3}) where {V,T} = Chain{varmanifold(4)}(one(T),x[1],x[2],x[3])

function SimplexTopology(t::EndomorphismField)
    n = length(t)
    SimplexTopology(0,[Values(i,i+n,i+2n) for i ∈ 1:n])
end
function SimplexBundle(M::TensorField,t::EndomorphismField)
    cols = columns(fiber(value(t)))
    SimplexBundle(0,_aff.(vcat((Ref(fiber(M)).+cols)...)),SimplexTopology(t))
end
function SimplexBundle(t::EndomorphismField)
    SimplexBundle(0,_aff.(vcat(columns(fiber(value(t)))...)),SimplexTopology(t))
end
SimplexBundle(M::TensorField,t::EndomorphismField,n::Int...) = SimplexBundle(resample(M,n...),resample(t,n...))
SimplexBundle(t::EndomorphismField,n::Int...) = SimplexBundle(resample(t,n...))

import Grassmann: complexify, polarize, vectorize
complexify(t::LocalFiber) = complexify(fiber(t))
complexify(t::TensorField) = TensorField(base(t),complexify.(fiber(t)))
polarize(t::LocalFiber) = polarize(fiber(t))
polarize(t::TensorField) = TensorField(base(t),polarize.(fiber(t)))
vectorize(t::LocalFiber) = vectorize(fiber(t))
vectorize(t::TensorField) = TensorField(base(t),vectorize.(fiber(t)))

(z::Phasor)(t::TensorField,θ...) = TensorField(t,z.(fiber(t),Ref.(θ)...))
(z::Phasor)(t::TensorField,θ::TensorField) = TensorField(t,z.(fiber(t),fiber(θ)))
(z::Phasor)(t::LocalTensor,θ...) = z(fiber(t),θ...)
(z::Phasor)(t::Coordinate,θ...) = z(point(t),θ...)
(z::Phasor)(t::LocalTensor,θ::LocalTensor) = z(fiber(t),fiber(θ))
(z::Phasor)(t::LocalTensor,θ::Coordinate) = z(fiber(t),point(θ))
(z::Phasor)(t::Coordinate,θ::LocalTensor) = z(point(t),fiber(θ))
(z::Phasor)(t::Coordinate,θ::Coordinate) = z(point(t),point(θ))

disconnect(t::FaceMap) = TensorField(disconnect(base(t)),fiber(t))
discontinuous(t::FaceMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::FaceMap,m) = TensorField(m,fiber(t))
discontinuous(t::SimplexMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::SimplexMap,m) = isdiscontinuous(t) ? t : TensorField(m,view(fiber(t),vertices(m)))
graphbundle(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N}) = SimplexBundle(PointCloud(0,fiber(graph.(t))),isdiscontinuous(t) ? disconnect(immersion(t)) : immersion(t))

const Components{T<:TensorField} = AbstractVector{T}
export boundarycomponents
function boundarycomponents(f::TensorField,n::Int=1)
    x = TensorField(base(f))
    N = ndims(f)
    siz = size(f)
    if N == 1
        Grassmann.FixedVector{2}([f[n],f[end-n+1]])
    elseif N == 2
        F1 = leaf(f,n,1)
        F2 = leaf(f,siz[1]-n+1,1)
        F3 = leaf(f,n,2)
        F4 = leaf(f,siz[2]-n+1,2)
        Grassmann.FixedVector{4}([F1,F2,F3,F4])
    elseif N == 3
        F1 = leaf(f,n,1)
        F2 = leaf(f,siz[1]-n+1,1)
        F3 = leaf(f,n,2)
        F4 = leaf(f,siz[2]-n+1,2)
        F5 = leaf(f,n,3)
        F6 = leaf(f,siz[3]-n+1,3)
        Grassmann.FixedVector{6}([F1,F2,F3,F4,F5,F6])
    elseif N == 4
        F1 = leaf(f,n,1)
        F2 = leaf(f,siz[1]-n+1,1)
        F3 = leaf(f,n,2)
        F4 = leaf(f,siz[2]-n+1,2)
        F5 = leaf(f,n,3)
        F6 = leaf(f,siz[3]-n+1,3)
        F7 = leaf(f,n,4)
        F8 = leaf(f,siz[4]-n+1,4)
        Grassmann.FixedVector{8}([F1,F2,F3,F4,F5,F6,F7,F8])
    else
        F1 = leaf(f,n,1)
        F2 = leaf(f,siz[1]-n+1,1)
        F3 = leaf(f,n,2)
        F4 = leaf(f,siz[2]-n+1,2)
        F5 = leaf(f,n,3)
        F6 = leaf(f,siz[3]-n+1,3)
        F7 = leaf(f,n,4)
        F8 = leaf(f,siz[4]-n+1,4)
        F9 = leaf(f,n,5)
        F10 = leaf(f,siz[5]-n+1,5)
        Grassmann.FixedVector{8}([F1,F2,F3,F4,F5,F6,F7,F8,F9,F10])
    end
end
function boundarycomponents(f::TensorField,n::AbstractVector{Int})
    vcat(Vector.(boundarycomponents.(Ref(f),n))...)
end
function boundarycomponents(f::TensorField,n::AbstractVector{Int},m::AbstractVector{Int})
    vcat(boundarycomponents.(boundarycomponents(f,n),Ref(m))...)
end
boundarycomponents(f::TensorField,::Colon) = boundarycomponents(f,1:minimum(size(f))÷2)
boundarycomponents(f::TensorField,::Colon,::Colon) = boundarycomponents(f,1:minimum(size(f))÷2,1:minimum(size(f))÷2)

Variation(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function Variation(cod::TensorField); p = points(cod).v[end]
    TensorField(p,leaf.(Ref(cod),1:length(p)))
end
function Variation(cod::TensorField{B,F,2,<:FiberProductBundle} where {B,F})
    p = base(cod).g.v[1]
    TensorField(p,leaf.(Ref(cod),1:length(p)))
end
const variation = Variation

alteration(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function alteration(cod::TensorField); p = points(cod).v[1]
    TensorField(p,leaf.(Ref(cod),1:length(p),1))
end

modification(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function modification(cod::TensorField); p = points(cod).v[2]
    TensorField(p,leaf.(Ref(cod),1:length(p),2))
end

variation(v::TensorField,fun::Function,args...) = variation(v,0.0,fun,args...)
variation(v::TensorField,fun::Function,fun!::Function,args...) = variation(v,0.0,fun,fun!,args...)
variation(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = variation(v,0.0,fun,fun!,f,args...)
function variation(v::Variation,t,fun::Function,args...)
    for i ∈ 1:length(v)
        display(fun(v.cod[i],args...))
        sleep(t)
    end
end
function variation(v::Variation,t,fun::Function,fun!::Function,::Val{T},args...) where T
    out = fun(v.cod[1],args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(v)
        T && empty!(ax)
        fun!(v.cod[i],args...)
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[end])
        display(fun(leaf(v,i),args...))
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...) where T
    out = fun(leaf(v,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[end])
        T && empty!(ax)
        fun!(leaf(v,i),args...)
        sleep(t)
    end
    return out
end
function variation(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...) where T
    x = resample(points(v).v[end],n)
    out = fun(leaf(v,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(x)-1
        T && empty!(ax)
        fun!(leaf(v,float(x[i])),args...)
        sleep(t)
    end
    T && empty!(ax)
    fun!(leaf(v,size(v)[end]),args...)
    return out
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,args...)
    for i ∈ 1:length(base(v).g.v[1])
        display(fun(leaf(v,i),args...))
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,fun!::Function,::Val{T},args...) where T
    out = fun(leaf(v,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(base(v).g.v[1])
        T && empty!(ax)
        fun!(leaf(v,i),args...)
        sleep(t)
    end
    return out
end

for fun ∈ (:variation,:alteration,:modification)
    let fun! = Symbol(fun,:!)
    @eval begin
        $fun(v::TensorField,t,fun::Function,fun!::Function,args...) = $fun(v,t,fun,fun!,Val(true),args...)
        $fun(v::TensorField,t,fun::Function,fun!::Function,n::Int,args...) = $fun(v,t,fun,fun!,n,Val(true),args...)
        $fun!(v::TensorField,fun::Function,args...) = $fun(v,0.0,fun,args...)
        $fun!(v::TensorField,fun::Function,fun!::Function,args...) = $fun!(v,0.0,fun,fun!,args...)
        $fun!(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = $fun!(v,0.0,fun,fun!,f,args...)
        $fun!(v::TensorField,fun::Function,fun!::Function,n::Int,args...) = $fun!(v,0.0,fun,fun!,n,args...)
        $fun!(v::TensorField,t,fun::Function,args...) = $fun(v,t,fun,args...)
        $fun!(v::TensorField,t,fun::Function,fun!::Function,args...) = $fun(v,t,fun,fun!,Val(false),args...)
        $fun!(v::TensorField,t,fun::Function,fun!::Function,n::Int,args...) = $fun(v,t,fun,fun!,n,Val(false),args...)
        #$fun!(v::TensorField,t,fun::Function,fun!::Function,n::Int,f::Function,args...) = $fun(v,t,fun,fun!,n,Val(false),f,args...)
    end end
end

alteration(v::TensorField,fun::Function,args...) = alteration(v,0.0,fun,args...)
alteration(v::TensorField,fun::Function,fun!::Function,args...) = alteration(v,0.0,fun,fun!,args...)
alteration(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = alteration(v,0.0,fun,fun!,f,args...)
function alteration(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[1])
        display(fun(leaf(v,i,1),args...))
        sleep(t)
    end
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...) where T
    out = fun(leaf(v,1,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[1])
        T && empty!(ax)
        fun!(leaf(v,i,1),args...)
        sleep(t)
    end
    return out
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...) where T
    x = resample(points(v).v[1],n)
    out = fun(leaf(v,float(x[1]),1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),1),args...)
        sleep(t)
    end
    return out
end
function _alteration(out,v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...) where T
    fig,ax,plt = out
    for i ∈ 1:length(points(v).v[1])
        T && empty!(ax)
        fun!(leaf(v,i,1),args...)
        sleep(t)
    end
    return out
end
function _alteration(out,v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...) where T
    x = resample(points(v).v[1],n)
    fig,ax,plt = out
    for i ∈ 1:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),1),args...)
        sleep(t)
    end
    return out
end

modification(v::TensorField,fun::Function,args...) = modification(v,0.0,fun,args...)
modification(v::TensorField,fun::Function,fun!::Function,args...) = modification(v,0.0,fun,fun!,args...)
modification(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = modification(v,0.0,fun,fun!,f,args...)
function modification(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[2])
        display(fun(leaf(v,i,2),args...))
        sleep(t)
    end
end
function modification(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...) where T
    out = fun(leaf(v,1,2),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[2])
        T && empty!(ax)
        fun!(leaf(v,i,2),args...)
        sleep(t)
    end
    return out
end
function modification(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...) where T
    x = resample(points(v).v[2],n)
    out = fun(leaf(v,float(x[1]),2),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),2),args...)
        sleep(t)
    end
    return out
end

export tensorfield
tensorfield(t,V=Manifold(t),W=V) = p->V(vector(↓(↑((V∪Manifold(t))(fiber(p)))⊘t)))
function tensorfield(t,ϕ::T) where T<:AbstractVector
    M = Manifold(t)
    V = Manifold(M)
    z = mdims(V) ≠ 4 ? Chain{V,1}(0.0,0.0) : Chain{V,1}(0.0,0.0,0.0)
    p->begin
        P = Chain{V,1}(one(valuetype(p)),value(p)...)
        for i ∈ 1:length(t)
            ti = value(t[i])
            Pi = Chain{V,1}(M[ti])
            P ∈ Pi && (return (Pi\P)⋅Chain{V,1}(ϕ[ti]))
        end
        return z
    end
end

_unorientedplane(p,v1,v2) = p.+[-v1-v2 v1+v2; v1-v2 v2-v1]
_orientedplane(p,v1,v2) = p.+[zero(v1) v1+v2; v1 v2]
unorientedplane(p,v1,v2) = TensorField(base(OpenParameter(2,2)),_unorientedplane(p,v1,v2))
orientedplane(p,v1,v2) = TensorField(base(OpenParameter(2,2)),_orientedplane(p,v1,v2))

include("grid.jl")
include("element.jl")
include("diffgeo.jl")

for fun ∈ (:unorientedpoly,:orientedpoly,:makietransform)
    @eval function $fun end
end
for fun ∈ (:linegraph,:tangentbundle,:normalbundle,:planesbundle,:arrowsbundle,:spacesbundle,:scaledbundle,:scaledfield,:scaledarrows,:scaledplanes,:scaledspaces,:planes,:spaces)
    @eval begin
        function $fun end
        function $(Symbol(fun,:!)) end
        export $fun, $(Symbol(fun,:!))
    end
end

point2chain(x,V=Submanifold(2)) = Chain(x[1],x[2])
point3chain(x,V=Submanifold(3)) = Chain(x[1],x[2],x[3])

polytransform(x) = vec(x)#[x[1],x[3],x[2],x[4]]
argarrows2(s,siz=s) = (;lengthscale=s)
argarrows3(s,siz=2s/33) = (;lengthscale=s)#,tipradius=siz/3,tiplength=siz,shaftradius=siz/7)
function argarrows(t::TensorField{B,<:TensorOperator{V,W}},s,siz=2s/33) where {B,V,W}
    mdims(W) ≠ 3 ? argarrows2(s) : argarrows3(s,siz)
end
function argarrows(t::TensorField{B,<:Chain{V}},s,siz=2s/33) where {B,V}
    mdims(V) ≠ 3 ? argarrows2(s,siz) : argarrows3(s,siz)
end

streamargs(t,args) = args
streamargs(t::TensorField{B,F,3} where {B,F},args) = streamargs(args)
function streamargs(args)
    if haskey(args,:gridsize)
        wargs = Dict(args)
        delete!(wargs,:gridsize)
        (;:gridsize => args[:gridsize],wargs...)
    else
        pairs((;:gridsize => (11,11,11),args...))
    end
end
function streamargs(dim::Bool,args)
    if haskey(args,:gridsize)
        wargs = Dict(args)
        delete!(wargs,:gridsize)
        (;:gridsize => dim ? (args[:gridsize]...,1) : args[:gridsize],wargs...)
    else
        pairs((;:gridsize => dim ? (32,32,1) : (32,32),args...))
    end
end

function gridargs(M,t,args)
    if haskey(args,:gridsize)
        wargs = Dict(args)
        delete!(wargs,:gridsize)
        resample(M,args[:gridsize]),resample(t,args[:gridsize]),(;wargs...)
    elseif haskey(args,:arcgridsize)
        wargs = Dict(args)
        delete!(wargs,:arcgridsize)
        aM = arcresample(M,args[:arcgridsize])
        aM,TensorField(base(aM),t.(points(aM))),(;wargs...)
    else
        M,t,args
    end
end
function gridargs(t,args)
    if haskey(args,:gridsize)
        wargs = Dict(args)
        delete!(wargs,:gridsize)
        resample(t,args[:gridsize]),(;wargs...)
    elseif haskey(args,:arcgridsize)
        wargs = Dict(args)
        delete!(wargs,:arcgridsize)
        arcresample(t,args[:arcgridsize]),(;wargs...)
    else
        t,args
    end
end

if !isdefined(Base, :get_extension)
using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/MakieExt.jl")
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" include("../ext/UnicodePlotsExt.jl")
    @require Meshes = "eacbb407-ea5a-433e-ab97-5258b1ca43fa" include("../ext/MeshesExt.jl")
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" include("../ext/GeometryBasicsExt.jl")
    @require Delaunay="07eb4e4e-0c6d-46ef-bc4e-83d5e5d860a9" include("../ext/DelaunayExt.jl")
    @require QHull="a8468747-bd6f-53ef-9e5c-744dbc5c59e7" include("../ext/QHullExt.jl")
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" include("../ext/MiniQhullExt.jl")
    @require Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344" include("../ext/TriangulateExt.jl")
    @require TetGen="c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea" include("../ext/TetGenExt.jl")
    @require MATLAB="10e44e05-a98a-55b3-a45b-ba969058deb6" include("../ext/MATLABExt.jl")
end
end

end # module Cartan
