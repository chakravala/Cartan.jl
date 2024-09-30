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

include("topology.jl")

export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid, Variation
export RealFunction, ComplexMap, SpinorField, CliffordField
export ScalarMap, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap, AbstractCurve
export alteration, variation

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
TensorField(a::TensorField,b::TensorField) = TensorField(fiber(a),fiber(b))

#const ParametricMesh{B,F,P<:AbstractVector{<:Chain}} = TensorField{B,F,1,P}
const ScalarMap{B,F<:AbstractReal,P<:SimplexFrameBundle} = TensorField{B,F,1,P}
#const ElementFunction{B,F<:AbstractReal,P<:AbstractVector} = TensorField{B,F,1,P}
const IntervalMap{B,F,P<:Interval} = TensorField{B,F,1,P}
const RectangleMap{B,F,P<:RealSpace{2}} = TensorField{B,F,2,P}
const HyperrectangleMap{B,F,P<:RealSpace{3}} = TensorField{B,F,3,P}
const ParametricMap{B,F,N,P<:RealSpace} = TensorField{B,F,N,P}
const Variation{B,F<:TensorField,N,P} = TensorField{B,F,N,P}
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
        $bundle(dom::AbstractArray) = $bundle(dom, dom)
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
PointCloud(t::TensorField) = PointCloud(base(t))
Grassmann.grade(::GradedField{G}) where G = G

@pure Base.eltype(::Type{<:TensorField{B,F}}) where {B,F} = LocalTensor{B,F}
Base.getindex(m::TensorField,i::Vararg{Int}) = LocalTensor(getindex(domain(m),i...), getindex(codomain(m),i...))
#Base.setindex!(m::TensorField{B,F,1,<:Interval},s::LocalTensor,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::TensorField{B,F,N,<:IntervalRange} where {B,F,N},s::LocalTensor,i::Vararg{Int})
    setindex!(codomain(m),fiber(s),i...)
    return s
end
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

Base.BroadcastStyle(::Type{<:TensorField{B,F,N,P}}) where {B,F,N,P} = Broadcast.ArrayStyle{TensorField{B,F,N,P}}()
Base.BroadcastStyle(::Type{<:GlobalSection{B,F,N,BA,FA}}) where {B,F,N,BA,FA} = Broadcast.ArrayStyle{TensorField{B,F,N,BA,FA}}()

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

(m::TensorField{B,F,N,<:SimplexFrameBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(PointCloud(m)(i),fiber(m)[vertices(i)])

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
function (m::IntervalMap)(t); p = points(m)
    i = searchsortedfirst(p,t[1])-1
    linterp(t[1],p[i],p[i+1],m.cod[i],m.cod[i+1])
end
function (m::IntervalMap)(t::Vector,d=diff(m.cod)./diff(m.dom))
    [parametric(i,m,d) for i ∈ t]
end
function parametric(t,m,d=diff(codomain(m))./diff(domain(m)))
    p = points(m)
    i = searchsortedfirst(p,t)-1
    codomain(m)[i]+(t-p[i])*d[i]
end

(m::RectangleMap)(i::Int,j::Int=1) = TensorField(points(m).v[j],isone(j) ? m.cod[:,i] : m.cod[i,:])
function (m::RectangleMap)(t::AbstractFloat,j::Int=1)
    k = isone(j) ? 2 : 1; p = points(m).v[k]
    i = searchsortedfirst(p,t)-1
    TensorField(points(m).v[j],linterp(t,p[i],p[i+1],isone(j) ? m.cod[:,i] : m.cod[i,:],isone(j) ? m.cod[:,i+1] : m.cod[i+1,:]))
end

(m::TensorField{B,F,2,<:FiberProductBundle} where {B,F})(i::Int) = TensorField(base(base(m)),fiber(m)[:,i])
function (m::TensorField{B,F,2,<:FiberProductBundle} where {B,F})(t::AbstractFloat)
    k = 2; p = base(m).cod.v[1]
    i = searchsortedfirst(p,t)-1
    TensorField(base(base(m)),linterp(t,p[i],p[i+1],m.cod[:,i],m.cod[:,i+1]))
end


(m::TensorField)(t::TensorField) = TensorField(base(t),m.(fiber(t)))
(m::GridFrameBundle)(t::TensorField) = GridFrameBundle(base(t),TensorField(base(m),fiber(m)).(fiber(t)))

function (m::TensorField{B,F,N,<:SimplexFrameBundle} where {B,N})(t) where F
    j = findfirst(t,domain(m)); iszero(j) && (return zero(F))
    i = immersion(m)[j]
    Chain(codomain(m)[i])⋅(Chain(points(domain(m))[i])\t)
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
for fun ∈ (:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2)#:inv
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

Variation(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function Variation(cod::TensorField); p = points(cod).v[end]
    TensorField(p,cod.(1:length(p)))
end
function Variation(cod::TensorField{B,F,2,<:FiberProductBundle} where {B,F})
    p = base(cod).cdo.v[1]
    TensorField(p,cod.(1:length(p)))
end

alteration(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function alteration(cod::TensorField); p = points(cod).v[1]
    TensorField(p,cod.(1:length(p),2))
end

include("diffgeo.jl")
include("constants.jl")
include("element.jl")

variation(v::TensorField,fun::Function,args...) = variation(v,0.0,fun,args...)
variation(v::TensorField,fun::Function,fun!::Function,args...) = variation(v,0.0,fun,fun!,args...)
function variation(v::Variation,t,fun::Function,args...)
    for i ∈ 1:length(v)
        display(fun(v.cod[i],args...))
        sleep(t)
    end
end
function variation(v::Variation,t,fun::Function,fun!::Function,args...)
    display(fun(v.cod[1],args...))
    for i ∈ 2:length(v)
        fun!(v.cod[i],args...)
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[2])
        display(fun(v(i),args...))
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,fun!::Function,args...)
    display(fun(v(1),args...))
    for i ∈ 2:length(points(v).v[2])
        fun!(v(i),args...)
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,args...)
    for i ∈ 1:length(base(v).cod.v[1])
        display(fun(v(i),args...))
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,fun!::Function,args...)
    display(fun(v(1),args...))
    for i ∈ 2:length(base(v).cod.v[1])
        fun!(v(i),args...)
        sleep(t)
    end
end

alteration(v::TensorField,fun::Function,args...) = alteration(v,0.0,fun,args...)
alteration(v::TensorField,fun::Function,fun!::Function,args...) = alteration(v,0.0,fun,fun!,args...)
function alteration(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[1])
        display(fun(v(i,2),args...))
        sleep(t)
    end
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,args...)
    display(fun(v(1,2),args...))
    for i ∈ 2:length(points(v).v[1])
        fun!(v(i,2),args...)
        sleep(t)
    end
end

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
            x,y = Real.(points(t)),value.(codomain(t))
            display(Makie.lines(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),grade(t))
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function linegraph!(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
            x,y = Real.(points(t)),value.(codomain(t))
            display(Makie.lines!(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),grade(t))
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function Makie.lines(t::IntervalMap{B,<:Endomorphism};args...) where B<:Coordinate{<:AbstractReal}
            display(Makie.lines(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.lines!(getindex.(t,i);args...)
            end
        end
        function Makie.lines!(t::IntervalMap{B,<:Endomorphism};args...) where B<:Coordinate{<:AbstractReal}
            display(Makie.lines!(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.lines!(getindex.(t,i);args...)
            end
        end
        function Makie.arrows(M::VectorField,t::TensorField{B,<:Endomorphism,N,<:GridFrameBundle} where B;args...) where N
            Makie.arrows(TensorField(fiber(M),fiber(t));args...)
        end
        function Makie.arrows!(M::VectorField,t::TensorField{B,<:Endomorphism,N,<:GridFrameBundle} where B;args...) where N
            Makie.arrows!(TensorField(fiber(M),fiber(t)))
        end
        function Makie.arrows(t::TensorField{<:Coordinate{<:Chain},<:Endomorphism,N,<:GridFrameBundle};args...) where N
            display(Makie.arrows(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        function Makie.arrows!(t::TensorField{<:Coordinate{<:Chain},<:Endomorphism,N,<:GridFrameBundle};args...) where N
            display(Makie.arrows!(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        function Makie.arrows(t::VectorField{<:Coordinate{<:Chain},<:Endomorphism,2,<:RealSpace{2}};args...)
            display(Makie.arrows(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        function Makie.arrows!(t::VectorField{<:Coordinate{<:Chain},<:Endomorphism,2,<:RealSpace{2}};args...)
            display(Makie.arrows!(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        Makie.volume(t::VolumeGrid;args...) = Makie.volume(points(t).v...,Real.(codomain(t));args...)
        Makie.volume!(t::VolumeGrid;args...) = Makie.volume!(points(t).v...,Real.(codomain(t));args...)
        Makie.volumeslices(t::VolumeGrid;args...) = Makie.volumeslices(points(t).v...,Real.(codomain(t));args...)
        for fun ∈ (:surface,:surface!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(points(t).v...,Real.(codomain(t));color=Real.(abs.(codomain(gradient_fast(Real(t))))),args...)
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F<:AbstractComplex};args...) = Makie.$fun(points(t).v...,Real.(radius.(codomain(t)));color=Real.(angle.(codomain(t))),colormap=:twilight,args...)
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G;args...) where {B,F<:Chain}
                    x,y = points(t),value.(codomain(t))
                    yi = Real.(getindex.(y,1))
                    display(Makie.$fun(x.v...,yi;color=Real.(abs.(codomain(gradient_fast(x→yi)))),args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),grade(t))
                        yi = Real.(getindex.(y,i))
                        Makie.$(funsym(fun))(x.v...,yi;color=Real.(abs.(codomain(gradient_fast(x→yi)))),args...)
                    end
                end
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = Makie.$fun(points(t).v...,Real.(radius.(codomain(t)));args...)
            end
        end
        for fun ∈ (:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = Makie.$fun(points(t).v...,Real.(angle.(codomain(t)));colormap=:twilight,args...)
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!)
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
        Makie.wireframe(t::SurfaceGrid;args...) = Makie.wireframe(graph(t);args...)
        Makie.wireframe!(t::SurfaceGrid;args...) = Makie.wireframe!(graph(t);args...)
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
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain{W,L,F,2} where {W,L,F}},<:Chain{V,G,T,2} where {V,G,T},2,<:AlignedRegion{2}};args...) = Makie.$fun(domain(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,2,<:RealSpace{2}} where F;args...) = Makie.$fun(Makie.Point.(points(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,N,<:GridFrameBundle} where {F,N};args...) = Makie.$fun(Makie.Point.(points(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::VectorField,f::VectorField;args...) = Makie.$fun(Makie.Point.(fiber(t))[:],Makie.Point.(fiber(f))[:];args...)
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
        for fun ∈ (:mesh,:mesh!,:wireframe,:wireframe!)
            @eval Makie.$fun(M::GridFrameBundle;args...) = Makie.$fun(GeometryBasics.Mesh(M);args...)
        end
        function linegraph(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            variation(M,Makie.lines,Makie.lines!)
            alteration(M,Makie.lines!,Makie.lines!)
        end
        function linegraph!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            variation(M,Makie.lines!,Makie.lines!)
            alteration(M,Makie.lines!,Makie.lines!)
        end
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            Makie.mesh(GridFrameBundle(fiber(M));args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            Makie.mesh!(GridFrameBundle(fiber(M));args...)
        end
        function Makie.mesh(M::TensorField{B,<:AbstractReal,2,<:GridFrameBundle} where B;args...)
            Makie.mesh(GeometryBasics.Mesh(base(M));color=fiber(M)[:],args...)
        end
        function Makie.mesh!(M::TensorField{B,<:AbstractReal,2,<:GridFrameBundle} where B;args...)
            Makie.mesh!(GeometryBasics.Mesh(base(M));color=fiber(M)[:],args...)
        end
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::TensorField;args...)
            Makie.mesh(GridFrameBundle(fiber(M));color=fiber(f)[:],args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::TensorField;args...)
            Makie.mesh!(GridFrameBundle(fiber(M));color=fiber(f)[:],args...)
        end
        Makie.wireframe(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...) = Makie.wireframe(GridFrameBundle(fiber(M));args...)
        Makie.wireframe!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...) = Makie.wireframe!(GridFrameBundle(fiber(M));args...)
        function Makie.mesh(M::SimplexFrameBundle;args...)
            if mdims(points(M)) == 2
                sm = submesh(M)[:,1]
                Makie.lines(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh(submesh(M),array(ImmersedTopology(M));args...)
            end
        end
        function Makie.mesh!(M::SimplexFrameBundle;args...)
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
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ScalarMap;args...) = UnicodePlots.lineplot!(p,getindex.(domain(t),2),codomain(t);args...)
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::PlaneCurve;args...) = UnicodePlots.lineplot!(p,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(Real.(points(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::RealFunction;args...) = UnicodePlots.lineplot!(p,Real.(points(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot(t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(Real.(points(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,Real.(points(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
        UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
        UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(codomain(t));args...)
        UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(codomain(t));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],args...)
        UnicodePlots.heatmap(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.heatmap(Real.(angle.(codomain(t)));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],colormap=:twilight,args...)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractComplex,1,<:Interval}) where B = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
        Base.display(t::GradedField{G,B,<:TensorGraded,1,<:Interval} where {G,B}) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
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
            s = size(eltype(c))[1]+1; V = varmanifold(s) # s
            n = size(eltype(f))[1]
            p = [Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c]
            M = s ≠ n ? p(list(s-n+1,s)) : p
            t = SimplexManifold([Values{n,Int}(k) for k ∈ f])
            return (p,∂(t),t)
        end
    end
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" begin
        MiniQhull.delaunay(p::Vector{<:Chain},n=1:length(p)) = MiniQhull.delaunay(PointCloud(p),n)
        function MiniQhull.delaunay(p::PointCloud,n=1:length(p))
            l = list(1,mdims(p))
            T = MiniQhull.delaunay(Matrix(submesh(length(n)==length(p) ? p : p[n])'))
            SimplexManifold([Values{4,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)],length(p))
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
