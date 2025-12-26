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

using SparseArrays, LinearAlgebra, ElasticArrays, Base.Threads, Grassmann, AbstractFFTs
import Grassmann: value, vector, valuetype, tangent, istangent, Derivation, radius, ⊕
import Grassmann: realvalue, imagvalue, points, metrictensor, metricextensor
import Grassmann: Values, Variables, FixedVector, list, volume, compound
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Grassmann: complexify, polarize, vectorize, gradient
import Base: @pure, OneTo, getindex
import LinearAlgebra: cross
import ElasticArrays: resize_lastdim!

export Values, Derivation, differential, codifferential, boundary
export initmesh, pdegrad, det, graphbundle, divergence, grad, nabla

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
export metrictensorfield, metricextensorfield, polarize, complexify, vectorize, findroot
export leaf, alteration, variation, modification, alteration!, variation!, modification!
export graylines, graylines!

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

isrange(t::TensorField) = isrange(points(t))
function resample(t::TensorField,i::NTuple=size(t))
    i == size(t) && isrange(t) && (return t)
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
assign!(x::AbstractArray{T,3} where T,i,s) = (@inbounds x[:,:,i] .= s)
assign!(x::AbstractArray{T,4} where T,i,s) = (@inbounds x[:,:,:,i] = s)
assign!(x::AbstractArray{T,5} where T,i,s) = (@inbounds x[:,:,:,:,i] = s)

Base.collect(x::TensorField) = TensorField(x,collect(fiber(x)))
Base.broadcastable(x::TensorField{B,F,N,P,<:AbstractRange} where {B,F,N,P}) = collect(x)

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
for op ∈ (:+,:-,:&,:∧,:∨,:min,:max,:div,:rem,:mod,:mod1,:ldexp)
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
Base.:<(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a),Grassmann.contraction_metric.(fiber(b),fiber(a),refmetric(base(a))))
Base.:<(a::Number,b::TensorField) = TensorField(base(b), Grassmann.contraction.(fiber(b),a))
Base.:<(a::TensorField,b::Number) = TensorField(base(a), Grassmann.contraction.(b,fiber(a)))
for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:contraction,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
    let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
    @eval begin
        $bop(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a),Grassmann.$mop.(fiber(a),fiber(b),refmetric(base(a))))
        $bop(a::Number,b::TensorField) = TensorField(base(b), Grassmann.$op.(a,fiber(b)))
        $bop(a::TensorField,b::Number) = TensorField(base(a), Grassmann.$op.(fiber(a),b,$((op≠:^ ? () : (:(refmetric(base(a))),))...)))
    end end
end
for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad,:transpose,:iszero,:isone,:isnan,:isinf,:isfinite,:floor,:ceil,:round)
    @eval Base.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
end
for fun ∈ (:exp,:exp2,:exp10,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:acsch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:inv)
    @eval Base.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t),ref(metricextensor(t))))
end
for fun ∈ (:reverse,:clifford,:even,:odd,:scalar,:vector,:bivector,:pseudoscalar,:value,:complementleft,:realvalue,:imagvalue,:outermorphism,:Outermorphism,:DiagonalOperator,:TensorOperator,:eigen,:eigvecs,:eigvals,:eigvalsreal,:eigvalscomplex,:eigvecsreal,:eigvecscomplex,:eigpolys,:pfaffian,:∧,:↑,:↓,:vectorize,:discriminant,:discriminantreal,:discriminantcomplex,:vandermonde,:vandermondereal,:vandermondecomplex)
    @eval Grassmann.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
end
for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric,:unit,:complexify,:polarize,:amplitude,:phase)
    @eval Grassmann.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t),ref(metricextensor(t))))
end
for fun ∈ (:sum,:prod)
    @eval Base.$fun(t::TensorField) = $fun(fiber(t))
end
for fun ∈ (:cumsum,:cumprod)
    @eval Base.$fun(t::TensorField) = TensorField(base(t),$fun(fiber(t)))
end

Base.:+(m::TensorField,t::LocalTensor) = LocalTensor(base(t),m+fiber(t))
Base.:-(m::TensorField,t::LocalTensor) = LocalTensor(base(t),m-fiber(t))
Base.:+(t::LocalTensor,m::TensorField) = LocalTensor(base(t),fiber(t)+m)
Base.:-(t::LocalTensor,m::TensorField) = LocalTensor(base(t),fiber(t)-m)

Base.:*(m::AbstractMatrix,t::LocalTensor) = LocalTensor(base(t),m*fiber(t))
Base.:\(m::AbstractMatrix,t::LocalTensor) = LocalTensor(base(t),m\fiber(t))
Base.:*(t::LocalTensor,m::AbstractMatrix) = LocalTensor(base(t),fiber(t)*m)
Base.:\(t::LocalTensor,m::AbstractMatrix) = LocalTensor(base(t),fiber(t)\m)

#Base.:*(m::DenseMatrix,t::TensorField) = TensorField(base(t),reshape(m*fiber(vec(t)),size(t)))
#Base.:*(m::AbstractSparseMatrix,t::TensorField) = TensorField(base(t),reshape(m*fiber(vec(t)),size(t)))
Base.:*(m::AbstractMatrix,t::RealFunction) = TensorField(base(t),reshape(m*fiber(t),size(t)))
#Base.:*(m::RectangleMap,t::RealFunction) = fiber(m)*t
#Base.:\(m::DenseMatrix,t::TensorField) = TensorField(base(t),reshape(m\fiber(vec(t)),size(t)))
#Base.:\(m::AbstractSparseMatrix,t::TensorField) = TensorField(base(t),reshape(m\fiber(vec(t)),size(t)))
Base.:\(m::AbstractMatrix,t::RealFunction) = TensorField(base(t),reshape(m\fiber(t),size(t)))
#Base.:\(m::RectangleMap,t::RealFunction) = fiber(m)\t
#Base.:*(t::TensorField,m::DenseMatrix) = TensorField(base(t),reshape(vec(transpose(fiber(vec(t)))*m),size(t)))
#Base.:*(t::TensorField,m::AbstractSparseMatrix) = TensorField(base(t),reshape(vec(transpose(fiber(vec(t)))*m),size(t)))
Base.:*(t::RealFunction,m::AbstractMatrix) = TensorField(base(t),reshape(vec(transpose(fiber(vec(t)))*m),size(t)))
#Base.:*(t::RealFunction,m::RectangleMap) = t*fiber(m)
#Base.:\(t::TensorField,m::DenseMatrix) = TensorField(base(t),reshape(vec(fiber(vec(t))\m),size(t)))
#Base.:\(t::TensorField,m::AbstractSparseMatrix) = TensorField(base(t),reshape(vec(fiber(vec(t))\m),size(t)))
Base.:\(t::RealFunction,m::AbstractMatrix) = TensorField(base(t),reshape(vec(fiber(vec(t))\m),size(t)))
#Base.:\(t::RealFunction,m::RectangleMap) = t\fiber(m)

Base.log(t::TensorField) = TensorField(base(t), Grassmann.log_metric.(fiber(t),ref(metricextensor(t))))
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

function Base.split(t::TensorField{B,<:Chain{V,G,T,1} where {V,G,T}} where B)
    getindex.(t,1)
end
function Base.split(t::TensorField{B,<:Chain{V,G,T,2} where {V,G,T}} where B)
    getindex.(t,1),getindex.(t,2)
end
function Base.split(t::TensorField{B,<:Chain{V,G,T,3} where {V,G,T}} where B)
    getindex.(t,1),getindex.(t,2),getindex.(t,3)
end
function Base.split(t::TensorField{B,<:Chain{V,G,T,4} where {V,G,T}} where B)
    getindex.(t,1),getindex.(t,2),getindex.(t,3),getindex.(t,4)
end
function Base.split(t::TensorField{B,<:Chain{V,G,T,5} where {V,G,T}} where B)
    getindex.(t,1),getindex.(t,2),getindex.(t,3),getindex.(t,4),getindex.(t,5)
end

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

findroot(t::TensorField) = minimum(norm(t))
findroot(t::TensorField,x) = minimum(norm(t-x))

(z::Phasor)(t::TensorField) = TensorField(t,z.(fiber(t)))
(z::Phasor)(t::TensorField,θ) = TensorField(t,z.(fiber(t),Ref(θ)))
(z::Phasor)(t::TensorField,θ::TensorField) = TensorField(t,z.(fiber(t),fiber(θ)))
(z::Phasor)(t,θ::TensorField) = TensorField(θ,z.(Ref(t),fiber(θ)))
(z::Phasor)(t::LocalTensor,θ...) = z(fiber(t),θ...)
(z::Phasor)(t::Coordinate,θ...) = z(point(t),θ...)
(z::Phasor)(t::LocalTensor,θ::LocalTensor) = z(fiber(t),fiber(θ))
(z::Phasor)(t::LocalTensor,θ::Coordinate) = z(fiber(t),point(θ))
(z::Phasor)(t::Coordinate,θ::LocalTensor) = z(point(t),fiber(θ))
(z::Phasor)(t::Coordinate,θ::Coordinate) = z(point(t),point(θ))

export FourierSpace
struct FourierSpace{T,F<:AbstractVector{T},G} <: AbstractVector{T}
    f::F
    v::G
end

Base.size(f::FourierSpace) = size(f.f)
Base.getindex(f::FourierSpace,i::Int) = f.f[i]
invdim(f::FourierSpace,dims=1) = length(f.v)
invdim(f::ProductSpace,dims=1) = invdim(f.v[dims])
invdim(f::AbstractVector,dims=1) = length(f)

export fftspace, rfftspace, r2rspace
import AbstractFFTs: fftfreq, rfftfreq, fftshift, ifftshift
for fun ∈ (:fftspace,:r2rspace)
    @eval begin
        $fun(t::TensorField) = TensorField($fun(base(t)))
        $fun(x::GridBundle) = GridBundle($fun(points(x)))
        $fun(x::ProductSpace{V}) where V = ProductSpace{V}($fun.(x.v))
    end
end
rfftspace(t::TensorField) = TensorField(rfftspace(base(t)))
rfftspace(x::GridBundle) = GridBundle(rfftspace(points(x)))
rfftspace(x::ProductSpace{V}) where V = ProductSpace{V}(rfftspace(x.v[2]),fftspace.(x.v[2:end])...)
r2rspace(t::TensorField,kind) = TensorField(r2rspace(base(t),kind))
r2rspace(x::GridBundle,kind) = GridBundle(r2rspace(points(x),kind))
r2rspace(x::ProductSpace{V},kind) where V = ProductSpace{V}(r2rspace.(x.v,kind))

GridBundle(x::FourierSpace) = GridBundle(x,ClampedTopology(size(x)))
rfftspace(N::Real,ω=1/N) = rfftfreq(N,N*ω)
rfftspace(x::AbstractRange) = FourierSpace(rfftspace(length(x),2π/(x[end]-x[1])),x)
rfftspace(x::FourierSpace) = x.v
rfftspace(x::Frequencies) = Base.OneTo(length(x))
fftspace(x::AbstractRange) = FourierSpace(fftspace(length(x),2π/(x[end]-x[1])),x)
function fftspace(N::Real,ω=1/N)
    n=(2(N-1)+iseven(N))
    (n/N)*rfftfreq(n,N*ω)
end
fftspace(x::FourierSpace) = x.v
fftspace(x::Frequencies) = Base.OneTo(length(x))

r2rspace(N::Real,ω::Float64=1/N) = fftspace(N,ω)
r2rspace(x::AbstractRange) = FourierSpace(r2rspace(length(x),π/(x[end]-x[1])),x)
r2rspace(x::FourierSpace) = x.v
r2rspace(x::Frequencies) = Base.OneTo(length(x))

function r2rspace(N::Real,kind::Int,fs=1)
    out = r2rspace(N,1/fs)
    kind ∈ (9,6,10) ? out .+ out[2] : out
end
r2rspace(x::AbstractRange,kind) = FourierSpace(r2rspace(length(x),kind,π/(x[end]-x[1])),x)
r2rspace(x::FourierSpace,kind) = x.v
r2rspace(x::Frequencies,kind) = Base.OneTo(length(x))

fftshiftalias(x) = fftshift(x).-x[Int(ceil(length(x)/2))]
fftshift(x::FourierSpace) = FourierSpace(fftshiftalias(x.f),x.v)
ifftshift(x::FourierSpace) = fftspace(x.v)
for fun ∈ (:fftshift,:ifftshift)
    @eval begin
        $fun(t::TensorField) = TensorField($fun(base(t)),$fun(fiber(t)))
        $fun(x::GridBundle) = GridBundle($fun(points(x)))
        $fun(x::ProductSpace{V}) where V = ProductSpace{V}($fun.(x.v))
    end
end
for fun ∈ (:fft,:fft!,:ifft,:ifft!,:bfft,:bfft!)
    @eval AbstractFFTs.$fun(t::TensorField,args...) = TensorField(fftspace(base(t)), $fun(fiber(t),args...))
end
for fun ∈ (:rfft,)
    @eval AbstractFFTs.$fun(t::TensorField,args...) = TensorField(rfftspace(base(t)), $fun(fiber(t),args...))
end
for fun ∈ (:irfft,:brfft)
    @eval begin
        AbstractFFTs.$fun(t::TensorField) = TensorField(rfftspace(base(t)), $fun(fiber(t),invdim(points(t))))
        AbstractFFTs.$fun(t::TensorField,dims) = TensorField(rfftspace(base(t)), $fun(fiber(t),invdim(points(t),dims[1]),dims))
    end
end

flt(f::TensorField,σ::Number) = fft(exp((-σ)*TensorField(base(f)))*f)
bflt(f::TensorField,σ::Number) = bfft(exp((-σ)*TensorField(base(f)))*f)
rflt(f::TensorField,σ::Number) = rfft(exp((-σ)*TensorField(base(f)))*f)
brflt(f::TensorField,σ::Number) = brfft(exp((-σ)*TensorField(base(f)))*f)
iflt(f::TensorField,σ::Number) = ifft(exp(σ*TensorField(base(f)))*f)
irflt(f::TensorField,σ::Number) = irfft(exp(σ*TensorField(base(f)))*f)

export flt, bflt, rflt, brflt, iflt, irflt
function flt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(fft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function bflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(bfft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function rflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(rfft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end
function brflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    id = length(t)
    out = Matrix{Complex{Float64}}(undef,length(σ),id)
    for i ∈ 1:length(σ)
        out[i,:] = fiber(brfft(exp((-fiber(σ)[i])*t)*f,id))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end

function iflt(f::TensorField)
    σ = TensorField(points(f).v[1])
    t = TensorField(fftspace(points(f).v[2]))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = exp.(fiber(σ)[i]*fiber(t)).*ifft(view(fiber(f),i,:))
    end
    return TensorField(base(t),vec(sum(out;dims=1))/length(σ))
end
function irflt(f::TensorField)
    σ = TensorField(points(f).v[1])
    t = TensorField(rfftspace(points(f).v[2]))
    id = length(t)
    out = Matrix{Complex{Float64}}(undef,length(σ),id)
    for i ∈ 1:length(σ)
        out[i,:] = exp.(fiber(σ)[i]*fiber(t)).*irfft(view(fiber(f),i,:),id)
    end
    return TensorField(base(t),vec(sum(out;dims=1))/length(σ))
end

export Chebyshev, ChebyshevMatrix, ChebyshevVector, chebyshevfft, chebyshevifft, unitpoints

struct Chebyshev{T,A<:AbstractVector} <: DenseVector{T}
    v::Vector{T}
    a::A
end

function Chebyshev(N::Int)
    θ = (π/(N-1))*(0:N-1)
    x = .-cos.(θ)
    Chebyshev(x,θ)
end
function Chebyshev(x::AbstractVector)
    c = Chebyshev(length(x))
    p = (points(c).+1)*((x[end]-x[1])/2).+x[1]
    Chebyshev(p,angle(c))
end

points(t::Chebyshev) = t.v
function unitpoints(t::Chebyshev)
    x = points(t)
    (x.-x[1])*(2/(x[end]-x[1])).-1
end
Base.angle(t::Chebyshev) = t.a
Base.getindex(t::Chebyshev,i::Integer) = getindex(points(t),i)
Base.size(t::Chebyshev) = size(points(t))

resample(m::Chebyshev,i::NTuple{1,Int}) = resample(m,i...)
resample(m::Chebyshev,i::Int=length(m)) = LinRange(m[1],m[end],i)

#ChebyshevVector(x::TensorField) = ChebyshevVector(points(x))
#ChebyshevVector(x::Chebyshev,N=length(x)) = vcat(0,reverse(inv(ChebyshevMatrix(-unitpoints(x))[1:N-1,1:N-1])[1,:]))*(interval_scale(x)/2)
#ChebyshevVector(N::Int) = vcat(0,reverse(inv(ChebyshevMatrix(N)[1:N-1,1:N-1])[1,:]))

ChebyshevVector(x,N=length(x)) = vcat(0,reverse(inv(ChebyshevMatrix(x)[1:N-1,1:N-1])[1,:]))
ChebyshevVector(N::Int) = vcat(0,reverse(inv(ChebyshevMatrix(N)[1:N-1,1:N-1])[1,:]))
ChebyshevMatrix(x::TensorField) = ChebyshevMatrix(points(x))
ChebyshevMatrix(x::Chebyshev) = ChebyshevMatrix(-points(x))
ChebyshevMatrix(N::Int) = iszero(N) ? [0;;] : ChebyshevMatrix(Chebyshev(N))
function ChebyshevMatrix(x) # differentiation matrix
    N = length(x)-1
    c = vcat(2,ones(N-1),2).*(-1).^(0:N)
    X = repeat(x,1,N+1)
    D = (c*inv.(c)')./((X-X')+I) # off-diagonal entries
    D-Diagonal(vec(sum(D,dims=2))) # diagonal entries
end

chebyshevfft(v::TensorField) = chebyshevfft(fiber(v))
chebyshevfft(v::TensorField,i::Int) = chebyshevfft(fiber(v),i)
chebyshevfft(v::AbstractVector) = fft(vcat(v,reverse(v[2:length(v)-1])))
function chebyshevfft(v::AbstractMatrix,i::Int)
    N,M = size(v)
    if isone(i)
        fft(vcat(v,reverse(v[2:N-1,:],dims=1)),i)
    else
        fft(hcat(v,reverse(v[:,2:M-1],dims=2)),i)
    end
end
function chebyshevfft(v::AbstractArray{T,3} where T,i::Int)
    N,M,R = size(v)
    if isone(i)
        fft(vcat(v,reverse(v[2:N-1,:,:],dims=1)),i)
    elseif i==2
        fft(hcat(v,reverse(v[:,2:M-1,:],dims=2)),i)
    else
        fft(cat(v,reverse(v[:,:,2:R-1],dims=2),dims=3),i)
    end
end

function chebyshevifft(V::AbstractVector,U::AbstractVector,N)
    ii = 0:N-2
    W = real.(ifft(V))
    w = zeros(N)
    w[2:N-1] = -W[2:N-1]./sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
    w[1] = sum((ii.^2).*U[ii.+1])/(N-1) .+ (0.5(N-1))*U[N]
    w[N] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N]
    return w
end

function chebyshevifft(V::AbstractMatrix,U::AbstractMatrix,i,N,M)
    W = real.(ifft(V,i))
    w = zeros(N,M)
    if isone(i)
        ii = 0:N-2
        x2 = sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
        for i ∈ 1:M
            w[2:N-1,i] = -W[2:N-1,i]./x2
            w[1,i] = sum((ii.^2).*U[ii.+1,i])/(N-1) .+ (0.5(N-1))*U[N,i]
            w[N,i] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1,i])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N,i]
        end
    else
        ii = 0:M-2
        y2 = sqrt.(1.0.-cos.(π*(1:M-2)/(M-1)).^2)
        for i ∈ 1:N
            w[i,2:M-1] = -W[i,2:M-1]./y2
            w[i,1] = sum((ii.^2).*U[i,ii.+1])/(M-1) .+ (0.5(M-1))*U[i,M]
            w[i,M] = sum((-1).^(ii.+1).*(ii.^2).*U[i,ii.+1])/(M-1) .+ 0.5(M-1)*(-1)^M*U[i,M]
        end
    end
    return w
end

function chebyshevifft(V::AbstractArray{T,3} where T,U::AbstractArray{T,3} where T,i,N,M,R)
    W = real.(ifft(V,i))
    w = zeros(N,M,R)
    if isone(i)
        ii = 0:N-2
        x2 = sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
        for i ∈ 1:M
            for j ∈ 1:R
                w[2:N-1,i,j] = -W[2:N-1,i,j]./x2
                w[1,i,j] = sum((ii.^2).*U[ii.+1,i,j])/(N-1) .+ (0.5(N-1))*U[N,i,j]
                w[N,i,j] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1,i,j])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N,i,j]
            end
        end
    elseif i==2
        ii = 0:M-2
        y2 = sqrt.(1.0.-cos.(π*(1:M-2)/(M-1)).^2)
        for i ∈ 1:N
            for j ∈ 1:R
                w[i,2:M-1,j] = -W[i,2:M-1,j]./y2
                w[i,1,j] = sum((ii.^2).*U[i,ii.+1,j])/(M-1) .+ (0.5(M-1))*U[i,M,j]
                w[i,M,j] = sum((-1).^(ii.+1).*(ii.^2).*U[i,ii.+1,j])/(M-1) .+ 0.5(M-1)*(-1)^M*U[i,M,j]
            end
        end
    else
        ii = 0:R-2
        y2 = sqrt.(1.0.-cos.(π*(1:R-2)/(R-1)).^2)
        for i ∈ 1:N
            for j ∈ 1:M
                w[i,j,2:R-1] = -W[i,j,2:R-1]./y2
                w[i,j,1] = sum((ii.^2).*U[i,j,ii.+1])/(R-1) .+ (0.5(R-1))*U[i,j,R]
                w[i,j,R] = sum((-1).^(ii.+1).*(ii.^2).*U[i,j,ii.+1])/(R-1) .+ 0.5(R-1)*(-1)^R*U[i,j,R]
            end
        end
    end
    return w
end


function chebyshevifft2(V1::AbstractVector,V2::AbstractVector,U::AbstractVector,N)
    W1 = real.(ifft(V1))
    W2 = real.(ifft(V2))
    u = zeros(N)
    ii = 2:N-1
    x = cos.(π*(1:N-2)/(N-1))
    x2 = sqrt.(1.0.-x.^2)
    u[ii] = W2[ii]./x2 - x.*W1[ii]./x2.^(3/2)
    return u
end


function chebyshevifft2(V1::AbstractMatrix,V2::AbstractMatrix,U::AbstractMatrix,i,N,M)
    W1 = real.(ifft(V1,i))
    W2 = real.(ifft(V2,i))
    u = zeros(N,M)
    if isone(i)
        ii = 2:N-1
        x = cos.(π*(1:N-2)/(N-1))
        x2 = sqrt.(1.0.-x.^2)
        for i ∈ 1:M
            u[ii,i] = W2[ii,i]./x2 - x.*W1[ii,i]./x2.^(3/2)
        end
    else
        ii = 2:M-1
        y = cos.(π*(1:M-2)/(M-1))
        y2 = sqrt.(1.0.-y.^2)
        for i ∈ 1:N
            u[i,ii] = W2[i,ii]./y2 - y.*W1[i,ii]./y2.^(3/2)
        end
    end
    return u
end

function chebyshevifft2(V1::AbstractArray{T,3} where T,V2::AbstractArray{T,3} where T,U::AbstractArray{T,3} where T,i,N,M,R)
    W1 = real.(ifft(V1,i))
    W2 = real.(ifft(V2,i))
    u = zeros(N,M,R)
    if isone(i)
        ii = 2:N-1
        x = cos.(π*(1:N-2)/(N-1))
        x2 = sqrt.(1.0.-x.^2)
        for i ∈ 1:M
            for j ∈ 1:R
                u[ii,i,j] = W2[ii,i,j]./x2 - x.*W1[ii,i,j]./x2.^(3/2)
            end
        end
    elseif i==2
        ii = 2:M-1
        y = cos.(π*(1:M-2)/(M-1))
        y2 = sqrt.(1.0.-y.^2)
        for i ∈ 1:N
            for j ∈ 1:R
                u[i,ii,j] = W2[i,ii,j]./y2 - y.*W1[i,ii,j]./y2.^(3/2)
            end
        end
    else
        ii = 2:R-1
        z = cos.(π*(1:R-2)/(R-1))
        z2 = sqrt.(1.0.-z.^2)
        for i ∈ 1:N
            for j ∈ 1:M
                u[i,j,ii] = W2[i,j,ii]./z2 - z.*W1[i,j,ii]./z2.^(3/2)
            end
        end
    end
    return u
end

export besseljzero
besseljzero(n,m,x=(m+n/2-1/4)*pi) = x - (4n^2-1)/(8x) # + ...

disconnect(t::FaceMap) = TensorField(disconnect(base(t)),fiber(t))
discontinuous(t::FaceMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::FaceMap,m) = TensorField(m,fiber(t))
discontinuous(t::SimplexMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::SimplexMap,m) = isdiscontinuous(t) ? t : TensorField(m,view(fiber(t),vertices(m)))
graphbundle(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N}) = TensorField(SimplexBundle(PointCloud(0,fiber(graph.(t))),isdiscontinuous(t) ? disconnect(immersion(t)) : immersion(t)),fiber(t))

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

variation(v::TensorField,fun::Function,args...;kw...) = variation(v,0.0,fun,args...;kw...)
variation(v::TensorField,fun::Function,fun!::Function,args...;kw...) = variation(v,0.0,fun,fun!,args...;kw...)
variation(v::TensorField,fun::Function,fun!::Function,f::Function,args...;kw...) = variation(v,0.0,fun,fun!,f,args...;kw...)
function variation(v::Variation,t,fun::Function,args...;kw...)
    for i ∈ 1:length(v)
        display(fun(v.cod[i],args...;kw...))
        sleep(t)
    end
end
function variation(v::Variation,t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    out = fun(v.cod[1],args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(v)
        T && empty!(ax)
        fun!(v.cod[i],args...;kw...)
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,args...;kw...)
    for i ∈ 1:length(points(v).v[end])
        display(fun(leaf(v,i),args...;kw...))
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    out = fun(leaf(v,1),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[end])
        T && empty!(ax)
        fun!(leaf(v,i),args...;kw...)
        sleep(t)
    end
    return out
end
function variation(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...;kw...) where T
    x = resample(points(v).v[end],n)
    out = fun(leaf(v,1),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(x)-1
        T && empty!(ax)
        fun!(leaf(v,float(x[i])),args...;kw...)
        sleep(t)
    end
    T && empty!(ax)
    fun!(leaf(v,size(v)[end]),args...;kw...)
    return out
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,args...;kw...)
    for i ∈ 1:length(base(v).g.v[1])
        display(fun(leaf(v,i),args...;kw...))
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    out = fun(leaf(v,1),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(base(v).g.v[1])
        T && empty!(ax)
        fun!(leaf(v,i),args...;kw...)
        sleep(t)
    end
    return out
end

for fun ∈ (:variation,:alteration,:modification)
    let fun! = Symbol(fun,:!)
    @eval begin
        $fun(v::TensorField,t,fun::Function,fun!::Function,args...;kw...) = $fun(v,t,fun,fun!,Val(true),args...;kw...)
        $fun(v::TensorField,t,fun::Function,fun!::Function,n::Int,args...;kw...) = $fun(v,t,fun,fun!,n,Val(true),args...;kw...)
        $fun!(v::TensorField,fun::Function,args...;kw...) = $fun(v,0.0,fun,args...;kw...)
        $fun!(v::TensorField,fun::Function,fun!::Function,args...;kw...) = $fun!(v,0.0,fun,fun!,args...;kw...)
        $fun!(v::TensorField,fun::Function,fun!::Function,f::Function,args...;kw...) = $fun!(v,0.0,fun,fun!,f,args...;kw...)
        $fun!(v::TensorField,fun::Function,fun!::Function,n::Int,args...;kw...) = $fun!(v,0.0,fun,fun!,n,args...;kw...)
        $fun!(v::TensorField,t,fun::Function,args...;kw...) = $fun(v,t,fun,args...;kw...)
        $fun!(v::TensorField,t,fun::Function,fun!::Function,args...;kw...) = $fun(v,t,fun,fun!,Val(false),args...;kw...)
        $fun!(v::TensorField,t,fun::Function,fun!::Function,n::Int,args...;kw...) = $fun(v,t,fun,fun!,n,Val(false),args...;kw...)
        #$fun!(v::TensorField,t,fun::Function,fun!::Function,n::Int,f::Function,args...) = $fun(v,t,fun,fun!,n,Val(false),f,args...)
    end end
end

alteration(v::TensorField,fun::Function,args...;kw...) = alteration(v,0.0,fun,args...;kw...)
alteration(v::TensorField,fun::Function,fun!::Function,args...;kw...) = alteration(v,0.0,fun,fun!,args...;kw...)
alteration(v::TensorField,fun::Function,fun!::Function,f::Function,args...;kw...) = alteration(v,0.0,fun,fun!,f,args...;kw...)
function alteration(v::TensorField,t,fun::Function,args...;kw...)
    for i ∈ 1:length(points(v).v[1])
        display(fun(leaf(v,i,1),args...;kw...))
        sleep(t)
    end
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    out = fun(leaf(v,1,1),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[1])
        T && empty!(ax)
        fun!(leaf(v,i,1),args...;kw...)
        sleep(t)
    end
    return out
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...;kw...) where T
    x = resample(points(v).v[1],n)
    out = fun(leaf(v,float(x[1]),1),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),1),args...;kw...)
        sleep(t)
    end
    return out
end
function _alteration(out,v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    for i ∈ 1:length(points(v).v[1])
        T && empty!(ax)
        fun!(leaf(v,i,1),args...;kw...)
        sleep(t)
    end
    return out
end
function _alteration(out,v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...;kw...) where T
    x = resample(points(v).v[1],n)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    for i ∈ 1:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),1),args...;kw...)
        sleep(t)
    end
    return out
end

modification(v::TensorField,fun::Function,args...;kw...) = modification(v,0.0,fun,args...;kw...)
modification(v::TensorField,fun::Function,fun!::Function,args...;kw...) = modification(v,0.0,fun,fun!,args...;kw...)
modification(v::TensorField,fun::Function,fun!::Function,f::Function,args...;kw...) = modification(v,0.0,fun,fun!,f,args...;kw...)
function modification(v::TensorField,t,fun::Function,args...;kw...)
    for i ∈ 1:length(points(v).v[2])
        display(fun(leaf(v,i,2),args...;kw...))
        sleep(t)
    end
end
function modification(v::TensorField,t,fun::Function,fun!::Function,::Val{T},args...;kw...) where T
    out = fun(leaf(v,1,2),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[2])
        T && empty!(ax)
        fun!(leaf(v,i,2),args...;kw...)
        sleep(t)
    end
    return out
end
function modification(v::TensorField,t,fun::Function,fun!::Function,n::Int,::Val{T},args...;kw...) where T
    x = resample(points(v).v[2],n)
    out = fun(leaf(v,float(x[1]),2),args...;kw...)
    fig,ax,plg = fun≠fun! ? out : (0,0,0)
    fun≠fun! && display(out)
    sleep(t)
    for i ∈ 2:length(x)
        T && empty!(ax)
        fun!(leaf(v,float(x[i]),2),args...;kw...)
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

include("grid.jl")
include("element.jl")
include("diffgeo.jl")

@doc """
    ∂(ω) = ω⋅∇ # boundary

Defined by Grassmann's interior differential acting on `TensorAlgebra` elements, the `boundary` operator is symbolized as `∂`.
The operator contracts `ω` with the vector‑valued `nabla` derivation `∇`, lowering grade by one.
This operator is nilpotent (`∂(∂(ω)) == 0`) and constitutes the negative of the adjoint operator corresponding to exterior differential `d`.
""" Grassmann.boundary, Grassmann.∂

@doc """
    d(ω) = ∇∧ω # differential

Exterior `differential` symbolized as `d` is defined with action based on Grassmann's `TensorAlgebra` elements.
It raises the grade of `ω` by taking exterior products with the vector-valued `nabla` derivation `∇`.
Hence, `d` is nilpotent (`d(d(ω)) == 0`) and, together with interior `codifferential` `δ` these form the de Rham cochain complex (`Δ = d⋅δ + δ⋅d`).
""" Grassmann.differential, Grassmann.d

@doc """
    δ(ω) = -∂(ω) # codifferential

Interior `codifferential` symbolized as `δ` is the adjoint action of the exterior `differential` operator on `TensorAlgebra` elements.
In `Grassmann` it is defined as the negative of the interior `boundary` operator, so `δ` lowers the grade by one and satisfies `δ(δ(ω)) == 0` also.
Together with exterior `differential` `d` these form de Rham cochain complex `Δ = d⋅δ + δ⋅d`.
""" Grassmann.codifferential, Grassmann.δ

_unorientedplane(p,v1,v2) = p.+[-v1-v2 v1+v2; v1-v2 v2-v1]
_orientedplane(p,v1,v2) = p.+[zero(v1) v1+v2; v1 v2]
unorientedplane(p,v1,v2) = TensorField(base(OpenParameter(2,2)),_unorientedplane(p,v1,v2))
orientedplane(p,v1,v2) = TensorField(base(OpenParameter(2,2)),_orientedplane(p,v1,v2))

for fun ∈ (:unorientedpoly,:orientedpoly,:makietransform,:graylines,:graylines!)
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
    @require FFTW="7a1cc6ca-52ef-59f5-83cd-3a7055c09341" include("../ext/FFTWExt.jl")
    @require ToeplitzMatrices="c751599d-da0a-543b-9d20-d0a503d91d24" include("../ext/ToeplitzMatricesExt.jl")
    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" include("../ext/SpecialFunctionsExt.jl")
    @require FewSpecialFunctions="6fcbd3ca-4273-49c4-98b3-81b765566de6" include("../ext/FewSpecialFunctionsExt.jl")
    @require EllipticFunctions="6a4e32cb-b31a-4929-85af-fb29d9a80738" include("../ext/EllipticFunctionsExt.jl")
    @require Elliptic="b305315f-e792-5b7a-8f41-49f472929428" include("../ext/EllipticExt.jl")
    @require JacobiElliptic="2a8b799e-c098-4961-872a-356c768d184c" include("../ext/JacobiEllipticExt.jl")
end
end

end # module Cartan
