
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

export bound, boundabove, boundbelow, boundlog, isclosed, updatetopology

# analysis

boundabove(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = fiber(s)≤lim ? s : LocalTensor(base(s),T(lim))
boundabove(x::T,lim=10) where T<:Real = x≤lim ? x : T(lim)
boundabove(t::TensorField,lim=10) = TensorField(base(t), boundabove.(codomain(t),lim))
boundbelow(s::LocalTensor{B,T},lim=-10) where {B,T<:Real} = fiber(s)≥lim ? s : LocalTensor(base(s),T(lim))
boundbelow(x::T,lim=-10) where T<:Real = x≥lim ? x : T(lim)
boundbelow(t::TensorField,lim=-10) = TensorField(base(t), boundbelow.(codomain(t),lim))
bound(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = abs(fiber(s))≤lim ? s : LocalTensor(base(s),T(sign(fiber(s)*lim)))
bound(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : ((lim/x)*s))
bound(x::T,lim=10) where T<:Real = abs(x)≤lim ? x : T(sign(x)*lim)
bound(z,lim=10) = (x=abs(z); x≤lim ? z : (lim/x)*z)
bound(t::TensorField,lim=10) = TensorField(base(t), bound.(codomain(t),lim))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(sign(fiber(s))*(lim+log(x+1-lim)))))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(z::T,lim=10) where T<:Real = (x=abs(z); x≤lim ? z : T(sign(z)*(lim+log(x+1-lim))))
boundlog(z,lim=10) = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(t::TensorField,lim=10) = TensorField(base(t), boundlog.(codomain(t),lim))

isclosed(t::IntervalMap) = norm(codomain(t)[end]-codomain(t)[1]) ≈ 0
updatetopology(t::IntervalMap) = isclosed(t) ? TorusTopology(t) : t

(::Derivation)(t::TensorField) = getnabla(t)
function getnabla(t::TensorField)
    n = ndims(t)
    V = Submanifold(tangent(S"0",1,n))
    Chain(Values{n,Any}(Λ(V).b[2:n+1]...))
end

export invd, cartan, firststructure, secondstructure

cartan(ξ) = invd(ξ)⋅ξ
firststructure(θ,ω) = d(θ)+ω∧θ
secondstructure(ω) = d(ω)+ω∧ω

Grassmann.curl(t::TensorField) = ⋆d(t)
Grassmann.d(t::TensorField) = TensorField(fromany(∇(t)∧Chain(t)))
Grassmann.∂(t::TensorField) = TensorField(fromany(Chain(t)⋅∇(t)))
Grassmann.d(t::ScalarField{B,<:AbstractReal,N,<:FrameBundle,<:AbstractArray} where {B,N}) = gradient(t)
#Grassmann.∂(t::ScalarField) = gradient(t)
#Grassmann.∂(t::VectorField) = TensorField(domain(t), sum.(value.(codomain(t))))
#=function Grassmann.∂(t::VectorField{G,B,<:Chain{V},N,T} where {B,N,T}) where {G,V}
    n = mdims(V)
    TensorField(domain(t), Real.(Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅codomain(t)))
end
function Grassmann.∂(t::GradedField{G,B,<:Chain{V}} where B) where {G,V}
    n = mdims(V)
    TensorField(domain(t), (Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅codomain(t)))
end=#

@generated function dvec(t::TensorField{B,<:Chain{V,G} where {V,G}} where B)
    V = Manifold(fibertype(t)); N = mdims(V)
    Expr(:call,:TensorField,:(base(t)),Expr(:.,:(Chain{$V,1}),Expr(:tuple,[:(fiber(gradient(getindex.(t,$i)))) for i ∈ list(1,N)]...)))
end

@generated function Grassmann.d(t::TensorField{B,<:Chain{V,G,<:Chain} where {V,G}} where B)
    V = Manifold(fibertype(t)); N = mdims(V)
    Expr(:call,:TensorField,:(base(t)),Expr(:.,:(Chain{$V,1}),Expr(:tuple,[:(fiber(d(getindex.(t,$i)))) for i ∈ list(1,N)]...)))
end
Grassmann.d(t::DiagonalField{B,<:DiagonalOperator,N,<:FrameBundle,<:AbstractArray} where{B,N}) = DiagonalOperator(dvec(value(t)))
@generated function Grassmann.d(t::EndomorphismField{B,<:Endomorphism,N,<:FrameBundle,<:AbstractArray} where {B,N})
    V = Manifold(fibertype(t)); N = mdims(V)
    syms = Symbol.(:x,list(1,N))
    Expr(:block,:(V = $V),
        :($(Expr(:tuple,syms...)) = $(Expr(:tuple,[:(getindex.(t,$i)) for i ∈ list(1,N)]...))),
        Expr(:call,:TensorField,:(base(t)),Expr(:.,:TensorOperator,Expr(:tuple,Expr(:.,:(Chain{V,1}),Expr(:tuple,[Expr(:.,:(Chain{V,1}),Expr(:tuple,
        [:(fiber(d(getindex.($(syms[j]),$i)))) for i ∈ list(1,N)]...)) for j ∈ list(1,N)]...))))))
end
@generated function invd(t::EndomorphismField)
    fibertype(t) <: DiagonalOperator && (return :(DiagonalOperator.(dvec(.-value.(t)))))
    V = Manifold(fibertype(t)); N = mdims(V)
    syms = Symbol.(:x,list(1,N))
    Expr(:block,:(V = $V),
        :($(Expr(:tuple,syms...)) = $(Expr(:tuple,[:(-getindex.(t,$i)) for i ∈ list(1,N)]...))),
        Expr(:call,:TensorField,:(base(t)),Expr(:.,:TensorOperator,Expr(:tuple,Expr(:.,:(Chain{V,1}),Expr(:tuple,[Expr(:.,:(Chain{V,1}),Expr(:tuple,
        [:(fiber(d(getindex.($(syms[i]),$j)))) for i ∈ list(1,N)]...)) for j ∈ list(1,N)]...))))))
end

for op ∈ (:(Base.:*),:(Base.:/),:(Grassmann.:∧),:(Grassmann.:∨))
    @eval begin
        $op(::Derivation,t::TensorField) = TensorField(fromany($op(∇(t),Chain(t))))
        $op(t::TensorField,::Derivation) = TensorField(fromany($op(Chain(t),∇(t))))
    end
end
LinearAlgebra.dot(::Derivation,t::TensorField) = TensorField(fromany(Grassmann.contraction(∇(t),Chain(t))))
LinearAlgebra.dot(t::TensorField,::Derivation) = TensorField(fromany(Grassmann.contraction(Chain(t),∇(t))))

function Base.:*(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), (Ref(n).*codomain(t)))
    end
end
function Base.:*(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), (codomain(t).*Ref(n)))
    end
end
function LinearAlgebra.dot(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), dot.(Ref(n),codomain(t)))
    end
end
function LinearAlgebra.dot(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), dot.(codomain(t),Ref(n)))
    end
end

# differential geometry

import Grassmann: 𝓛, Lie, LieBracket, LieDerivative, bracket
export 𝓛, Lie, LieBracket, LieDerivative, bracket, Connection, CovariantDerivative, action

(X::LieDerivative)(f::ScalarField) = action(X.v,f)
(X::VectorField{B,<:Chain{V,1,T,N} where T,N} where B)(Y::VectorField{B,<:Chain{V,1,T,N} where T,N} where B) where {V,N} = action(X,Y)
(X::VectorField)(f::ScalarField) = action(X,f)
(X::GradedVector)(f::ScalarField) = action(X,f)
(X::ScalarField)(f::ScalarField) = action(X,f)
𝓛dot(x::Chain,y::Simplex{V}) where V = Chain{V}(Real.(x.⋅value(y)))
action(X::VectorField,f::ScalarField) = X⋅gradient(f)
action(X::GradedVector,f::ScalarField) = X⋅gradient(f)
action(X::ScalarField,f::ScalarField) = X⋅gradient(f)
function action(X::VectorField,Y::VectorField)
    TensorField(base(X),𝓛dot.(fiber(X),fiber(gradient(Y))))
end

struct Connection{T}
    ω::T
    Connection(ω::T) where T = new{T}(ω)
end

(∇::Connection)(X::VectorField) = CovariantDerivative(∇.ω⋅X,X)
(∇::Connection)(X::VectorField,Y::VectorField) = X(Y)+((∇.ω⋅X)⋅Y)

struct CovariantDerivative{T,X}
    ωv::T
    v::X
    CovariantDerivative(ωv::T,v::X) where {T,X} = new{T,X}(ωv,v)
end

CovariantDerivative(∇::Connection,X) = ∇(X)
(∇x::CovariantDerivative)(Y::VectorField) = ∇x.v(Y)+(∇x.ωv⋅Y)

export arclength, arctime, totalarclength, trapz, cumtrapz, linecumtrapz, psum, pcumsum
export centraldiff, tangent, tangent_fast, unittangent, speed, normal, unitnormal
export arcresample, arcsample, ribbon, tangentsurface, planecurve, link, linkmap
export normalnorm, area, surfacearea, weingarten, gausssign, jacobian, evolute, involute
export normalnorm_slow, area_slow, surfacearea_slow, weingarten_slow, gausssign_slow
export gaussintrinsic, gaussextrinsic, gaussintrinsicnorm, gaussextrinsicnorm
export gaussintrinsic_slow, gausseintrinsicnorm_slow, curvatures, meancurvature
export gaussextrinsic_slow, gaussextrinsicnorm_slow, principals, principalaxes
export tangent_slow, normal_slow, unittangent_slow, unitnormal_slow, jacobian_slow
export sector, sectordet, sectorintegral, sectorintegrate, linkintegral, linknumber
export sector_slow, sectordet_slow, sectorintegral_slow, sectorintegrate_slow
export unitjacobian, unitjacobian_slow

# use graph for IntervalMap? or RealFunction!
tangent(f::IntervalMap) = gradient(f)
tangent(f::ScalarField) = tangent(graph(f))
tangent(f::VectorField) = ∧(gradient(f))
normal(f::ScalarField) = ⋆tangent(f)
normal(f::VectorField) = ⋆tangent(f)
unittangent(f::ScalarField,n=tangent(f)) = unit(n)
unittangent(f::VectorField,n=tangent(f)) = unit(n)
unittangent(f::IntervalMap) = unitgradient(f)
unitnormal(f) = ⋆unittangent(f)
normalnorm(f) = Real(abs(normal(f)))
jacobian(f::IntervalMap) = gradient(f)
jacobian(f::ScalarField) = jacobian(graph(f))
jacobian(f::VectorField) = TensorOperator(gradient(f))
unitjacobian(f::IntervalMap) = unitgradient(f)
unitjacobian(f::ScalarField) = unitjacobian(graph(f))
unitjacobian(f::VectorField) = TensorField(base(f),TensorOperator.(map.(unit,fiber(gradient(f)))))
weingarten(f::VectorField) = jacobian(unitnormal(f))
Base.adjoint(f::IntervalMap) = jacobian(f)
Base.adjoint(f::ScalarField) = jacobian(f)
Base.adjoint(f::VectorField) = jacobian(f)

tangent_slow(f::IntervalMap) = gradient_slow(f)
tangent_slow(f::ScalarField) = tangent_slow(graph(f))
tangent_slow(f::VectorField) = ∧(gradient_slow(f))
normal_slow(f::ScalarField) = ⋆tangent_slow(f)
normal_slow(f::VectorField) = ⋆tangent_slow(f)
unittangent_slow(f::ScalarField,n=tangent_slow(f)) = unit(n)
unittangent_slow(f::VectorField,n=tangent_slow(f)) = unit(n)
unittangent_slow(f::IntervalMap) = unitgradient_slow(f)
unitnormal_slow(f) = ⋆unittangent_slow(f)
normalnorm_slow(f) = Real(abs(normal_slow(f)))
jacobian_slow(f::IntervalMap) = gradient_slow(f)
jacobian_slow(f::ScalarField) = jacobian_slow(graph(f))
jacobian_slow(f::VectorField) = TensorOperator(gradient_slow(f))
unitjacobian_slow(f::IntervalMap) = unitgradient_slow(f)
unitjacobian_slow(f::ScalarField) = unitjacobian_slow(graph(f))
unitjacobian_slow(f::VectorField) = TensorField(base(f),TensorOperator.(map.(unit,fiber(gradient_slow(f)))))
weingarten_slow(f::VectorField) = jacobian_slow(unitnormal_slow(f))

ribbon(f::AbstractCurve,g::Vector{<:AbstractCurve}) = TensorField(points(f)⊕LinRange(0,1,length(g)+1),hcat(fiber(f),fiber.(g)...))
ribbon(f::AbstractCurve,g::AbstractCurve,n::Int=100) = tangentsurface(f,g-f,n)
tangentsurface(f::AbstractCurve,g::AbstractCurve,n::Int=100) = ribbon(f,Ref(f).+(LinRange(inv(n),1,n).*Ref(g)))
tangentsurface(f::AbstractCurve,v::Real=1,n::Int=100) = tangentsurface(f,v*tangent(f),n)

sector(f::Chain{V,G,<:Real},J::Chain{V,G,<:Real}) where {V,G} = Chain{V}(f,J)
sector(f::Real,J::Chain{W,1,<:Real} where W) = Chain(f,value(J)...)
sector(f::Chain{V,G},J::Chain{W,1,<:Chain{V,G}}) where {W,V,G} = Chain{V}(f,value(J)...)
sector(f::TensorField,J::TensorField) = TensorField(base(f), sector.(fiber(f),fiber(J)))
sector(f::RealFunction) = f
sector(f::TensorField) = TensorOperator(sector(f,gradient(f)))
sectordet(f::RealFunction) = f
sectordet(f::PlaneCurve) = f∧gradient(f)
sectordet(f::VectorField) = f∧(∧(gradient(f)))
sectorintegral(f::RealFunction) = integral(f)
sectorintegral(f::TensorField) = integral(sectordet(f))/mdims(fibertype(f))
sectorintegrate(f::RealFunction) = integrate(f)
sectorintegrate(f::TensorField) = integrate(sectordet(f))/mdims(fibertype(f))
sector_slow(f::RealFunction) = f
sector_slow(f::TensorField) = TensorOperator(sector(f,gradient_slow(f)))
sectordet_slow(f::RealFunction) = f
sectordet_slow(f::PlaneCurve) = f∧gradient_slow(f)
sectordet_slow(f::VectorField) = f∧(∧(gradient_slow(f)))
sectorintegral_slow(f::RealFunction) = integral(f)
sectorintegral_slow(f::TensorField) = integral(sectordet_slow(f))/mdims(fibertype(f))
sectorintegrate_slow(f::RealFunction) = integrate(f)
sectorintegrate_slow(f::TensorField) = integrate(sectordet_slow(f))/mdims(fibertype(f))

area(f::VectorField) = integral(normalnorm(f))
surfacearea(f::VectorField) = integrate(normalnorm(f))
surfacearea(f::ElementBundle) = sum(fiber(volumes(f)))
surfacearea(f::ScalarMap) = surfacearea(graphbundle(f))
surfacearea(f::FaceMap) = surfacearea(interp(f))
principals(f::VectorField) = eigvals(shape(f))
principals(f::VectorField,i) = eigvals(shape(f),i)
principalaxes(f::VectorField) = eigvecs(shape(f))
principalaxes(f::VectorField,i) = eigvecs(shape(f),i)
curvatures(f::VectorField) = eigpolys(shape(f))
curvatures(f::VectorField,i) = eigpolys(shape(f),i)
meancurvature(f::VectorField) = eigpolys(shape(f),Val(1))
gausssign(f::VectorField) = sign(sectordet(unitnormal(f)))
gaussextrinsicnorm(f::VectorField) = norm(gaussextrinsic(f))
gaussintrinsicnorm(f::VectorField,N=normal(f)) = norm(gaussintrinsic(f,N))
gaussextrinsic(f::VectorField) = sectordet(unitnormal(f))
function gaussintrinsic(f::VectorField,N=normal(f))
    n,T = Real(abs(N)),pointtype(f)
    Chain{Manifold(T),mdims(T)}.(Real(sectordet(N/n))/n)
end

area_slow(f::VectorField) = integral(normalnorm_slow(f))
surfacearea_slow(f::VectorField) = integrate(normalnorm_slow(f))
gausssign_slow(f::VectorField) = sign(sectordet_slow(unitnormal_slow(f)))
gaussextrinsicnorm_slow(f::VectorField) = norm(gaussextrinsic_slow(f))
gaussintrinsicnorm_slow(f::VectorField,N=normal(f)) = norm(gaussintrinsic_slow(f,N))
gaussextrinsic_slow(f::VectorField) = sectordet_slow(unitnormal_slow(f))
function gaussintrinsic_slow(f::VectorField,N=normal_slow(f))
    n,T = Real(abs(N)),pointtype(f)
    Chain{Manifold(T),mdims(T)}(Real(sectordet_slow(N/n))/n)
end

area(f::PlaneCurve) = sectorintegral(f)
#volume(f::VectorField) = sectorintegral(f)
sectorarea(f::PlaneCurve) = sectorintegrate(f)
sectorvolume(f::VectorField) = sectorintegrate(f)

function speed(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(base(f), Real.(abs.(t,refmetric(f))))
end
function normal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(domain(f), centraldiff(t./s,d)./s)
end
function unitnormal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(domain(f), unit.(centraldiff(unit.(t,refmetric(f)),d),refmetric(f)))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, torsion, frame, unitframe, frenet, darboux, wronskian, bishoppolar, bishop, bishopframe, bishopunitframe

function curvature(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(domain(f), Real.(abs.(centraldiff(t./s,d),refmetric(f))./s))
end
function radius(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(domain(f), Real.(s./abs.(centraldiff(t./s,d),refmetric(f))))
end
function localevolute(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    n = centraldiff(t./s,d)
    an2 = abs2.(n,refmetric(f))
    TensorField(domain(f), (s./an2).*n)
end
evolute(f::AbstractCurve) = f+localevolute(f)
Grassmann.involute(f::AbstractCurve) = f-unittangent(f)*arclength(f)
function osculatingplane(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(domain(f), TensorOperator.(Chain.(t,fiber(normal(f,d,t)))))
end
function unitosculatingplane(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    T = unit.(t,refmetric(f))
    TensorField(domain(f),TensorOperator.(Chain.(T,unit.(centraldiff(T,d),refmetric(f)))))
end
function binormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(domain(f), .⋆(t.∧fiber(normal(f,d,t)),refmetric(f)))
end
function unitbinormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    T = unit.(t)
    TensorField(domain(f), .⋆(T.∧(unit.(centraldiff(T,d),refmetric(f))),refmetric(f)))
end
function torsion(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,d)
    b = t.∧n
    TensorField(domain(f), Real.((b.∧centraldiff(n,d))./abs2.(.⋆b,refmetric(f))))
end
#=function curvaturetrivector(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t,refmetric(f)); ut=t./a
    TensorField(domain(f), (abs2.(centraldiff(ut,d)./a)).*(b.∧centraldiff(n,d))./abs2.(.⋆b))
end=#
#torsion(f::TensorField,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),a=abs.(t)) = TensorField(domain(f), abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
frame(f::PlaneCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d)) = osculatingplane(f,d,e1)
@generated function frame(f::AbstractCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(s = abs.(e1,refmetric(f))),
        [:($(syms[i]) = centraldiff(unit.($(syms[i-1]),refmetric(f)),d)./s) for i ∈ list(2,N-1)]...,
        :(TensorField(base(f),TensorOperator.(Chain.($(syms...),.⋆(.∧($(syms...)),refmetric(f)))))))
end
unitframe(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = unitosculatingplane(f,d,t)
@generated function unitframe(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(e1 = unit.(t,refmetric(f))),
        [:($(syms[i]) = unit.(centraldiff($(syms[i-1]),d),refmetric(f))) for i ∈ list(2,N-1)]...,
        :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(f))),
        :(TensorField(base(f),TensorOperator.(Chain.($([:($(Symbol(:e,i))) for i ∈ list(1,N)]...))))))
end
function cartan(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    κ = curvature(f,d,t)
    TensorField(domain(f), Chain.(Chain.(0,κ),Chain.(.-κ,0)))
end
function cartan(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,d),refmetric(f))./s)
    τ = Real.((b.∧centraldiff(n,d))./abs2.(.⋆(b,refmetric(f)),refmetric(f)))
    TensorField(base(f),TensorOperator.(Chain.(Chain.(0,κ,0),Chain.(.-κ,0,τ),Chain.(0,.-τ,0))))
end
@generated function cartan(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    N = mdims(V)
    bas = Grassmann.bladeindex.(N,UInt.(Λ(V).b[list(2,N)].∧Λ(V).b[list(3,N+1)]))
    syms,curvs = Symbol.(:e,list(1,N)),Symbol.(:κ,list(1,N-1))
    vals = [:($(curvs[i]) = Real.(Grassmann.contraction_metric.($(syms[i+1]),centraldiff($(syms[i]),d),refmetric(f))./s)) for i ∈ list(1,N-1)]
    Expr(:block,:(s=abs.(t)),:(uf = fiber(unitframe(f,d,t))),
        [:($(syms[i]) = getindex.(uf,$i)) for i ∈ list(1,N)]...,vals...,
        :(TensorField(base(f),TensorOperator.(Chain.($([:(Chain.($([j ∈ (i-1,i+1) ? j==i-1 ? :(.-$(curvs[j])) : curvs[j-1] : 0 for j ∈ list(1,N)]...))) for i ∈ list(1,N)]...))))))
end # cartan(unitframe(f))
function frenet(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    T = t./s
    N = unit.(centraldiff(T,d),refmetric(f))
    TensorField(domain(f), TensorOperator.(centraldiff(Chain.(T,N),d)./s))
end
function frenet(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    T = t./s
    N = unit.(centraldiff(T,d),refmetric(f))
    TensorField(domain(f), TensorOperator.(centraldiff(Chain.(T,N,.⋆(T.∧N,refmetric(f))),d)./s))
end
frenet(f::AbstractCurve) = (F = unitframe(f); F⋅cartan(F))

# curvature, torsion, etc... invariant vector
curvatures(f::PlaneCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::SpaceCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::AbstractCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = curvatures(f,Val(1),d,t)
function curvatures(f::SpaceCurve,::Val{2},d=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    torsion(f,d,t)*Λ(V).b[7]
end
function curvatures(f::SpaceCurve,d::Vector=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    s = abs.(t,refmetric(f))
    n = centraldiff(t./s,d)./s
    b = t.∧n
    TensorField(domain(f), Chain{V,2}.(value.(abs.(n,refmetric(f))),0,getindex.((b.∧centraldiff(n,d))./abs.(.⋆(b,refmetric(f)),refmetric(f)).^2,1)))
end
@generated function curvatures(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    N = mdims(V)
    bas = Grassmann.bladeindex.(N,UInt.(Λ(V).b[list(2,N)].∧Λ(V).b[list(3,N+1)]))
    syms = Symbol.(:e,list(1,N))
    vals = [:(Real.(Grassmann.contraction_metric.($(syms[i+1]),centraldiff($(syms[i]),d),refmetric(f))./s)) for i ∈ list(1,N-1)]
    Expr(:block,:(s=abs.(t)),:(uf = fiber(unitframe(f,d,t))),
        [:($(syms[i]) = getindex.(uf,$i)) for i ∈ list(1,N)]...,
        :(TensorField(base(f),Chain{$V,2}.($([j ∈ bas ? vals[searchsortedfirst(bas,j)] : 0 for j ∈ list(1,Grassmann.binomial(N,2))]...)))))
end
@generated function curvatures(f::AbstractCurve,::Val{j},d=centraldiffpoints(f),e1=centraldifffiber(f,d)) where j
    V = Manifold(fibertype(f))
    N = mdims(V)
    j==1 && (return :(curvature(f,d,e1)*$(Λ(V).b[N+2])))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(s=abs.(e1)),
        [:($(syms[i]) = centraldiff($(syms[i-1])./s,d)) for i ∈ list(2,j<N-1 ? j+1 : N-1)]...,
        j==N-1 ? :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(f))) : nothing,
        :(TensorField(base(f),Real.(Grassmann.contraction_metric.(unit.($(Symbol(:e,j+1)),refmetric(f)),centraldiff(unit.($(syms[j]),refmetric(f)),d),refmetric(f))./abs.(e1)).*$(Λ(V).b[j+1]∧Λ(V).b[j+2]))))
end
darboux(f::AbstractCurve) = compound(unitframe(f),Val(2))⋅curvatures(f)
darboux(f::AbstractCurve,j) = compound(unitframe(f),Val(2))⋅curvatures(f,j)

function bishopframe(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s,b = Real.(abs.(t,refmetric(f)))
    T = t./s
    N = centraldiff(T,d)./s
    B = .⋆(T.∧N,refmetric(f))
    τs = fiber(torsion(f,d,t)).*s
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    cθ,sθ = cos.(θ),sin.(θ)
    TensorField(domain(f), TensorOperator.(Chain.(T,cθ.*N.-sθ.*B,sθ.*N+cθ.*B)))
end
function bishopunitframe(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s,b = Real.(abs.(t,refmetric(f)))
    T = unit.(t)
    N = unit.(centraldiff(t./s,d))
    B = .⋆(T.∧N,refmetric(f))
    τs = fiber(torsion(f,d,t)).*s
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    cθ,sθ = cos.(θ),sin.(θ)
    TensorField(domain(f), TensorOperator.(Chain.(T,cθ.*N.-sθ.*B,sθ.*N+cθ.*B)))
end
function bishoppolar(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,d),refmetric(f))./s)
    τs = Real.(((b.∧centraldiff(n,d))./abs2.(.⋆(b,refmetric(f)),refmetric(f))).*s)
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    TensorField(domain(f), Chain.(κ,θ))
end
function bishop(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,d),refmetric(f))./s)
    τs = Real.(((b.∧centraldiff(n,d))./abs2.(.⋆(b,refmetric(f)),refmetric(f))).*s)
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    TensorField(domain(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))
end
#bishoppolar(f::TensorField) = TensorField(domain(f), Chain.(value.(codomain(curvature(f))),getindex.(codomain(cumtrapz(torsion(f))),1)))
#bishop(f::TensorField,κ=value.(codomain(curvature(f))),θ=getindex.(codomain(cumtrapz(torsion(f))),1)) = TensorField(domain(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))

function planecurve(κ::RealFunction,φ::Real=0.0)
    int = iszero(φ) ? integral(κ) : integral(κ)+φ
    integral(Chain.(cos(int),sin(int)))
end

function wronskian(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), f.cod.∧t.∧n)
end

function linkmap(f::SpaceCurve,g::SpaceCurve)
    TensorField(base(f)×base(g),[fiber(g)[j]-fiber(f)[i] for i ∈ OneTo(length(f)), j ∈ OneTo(length(g))])
end
link(tf::Chain,tg::Chain,f::Chain,g::Chain) = (gf = g-f; ngf = norm(gf); ∧(gf,tf,tg)/(ngf*ngf*ngf))
function link(f::SpaceCurve,g::SpaceCurve)
    tf,tg = fiber(tangent(f)),fiber(tangent(g))
    TensorField(base(f)×base(g),[link(tf[i],tg[j],fiber(f)[i],fiber(g)[j]) for i ∈ OneTo(length(f)), j ∈ OneTo(length(g))])
end
linkintegral(f::SpaceCurve,g::SpaceCurve) = integral(link(f,g))/4π
linknumber(f::SpaceCurve,g::SpaceCurve) = integrate(link(f,g))/4π

#???
function compare(f::TensorField)#???
    d = centraldiffpoints(f)
    t = centraldifffiber(f,d)
    n = centraldiff(t,d)
    s = abs.(t)
    TensorField(domain(f), centraldiff(t./s,d).-n./s)
end #????

function frame(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    TensorField(domain(f), TensorOperator.(Chain.(fiber(Ψu),fiber(⋆(ξ3∧Ψu)),fiber(ξ3))))
end
function unitframe(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    ξ2 = ⋆(ξ3∧Ψu)
    TensorField(domain(f), TensorOperator.(Chain.(fiber(Ψu/abs(Ψu)),fiber(ξ2/abs(ξ2)),fiber(ξ3/abs(ξ3)))))
end

export surfacemetric, surfacemetricdiag, surfaceframe, shape
export firstform, firstformdiag, secondform, firstsecondform, thirdform
export applymetric, firstkind, secondkind, geodesic

function EFG(V,dfdx,dfdy)
    E,F,G = fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdx⋅dfdy)),fiber(Real(dfdy⋅dfdy))
    TensorOperator.(Chain{V}.(Chain{V}.(E,F),Chain{V}.(F,G)))
end
function EG(V,dfdx,dfdy)
    DiagonalOperator.(Chain{V}.(fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdy⋅dfdy))))
end
function LMN(V,n,ddfdx2,ddfdxdy,ddfdy2)
    L,M,N = fiber(Real(ddfdx2⋅n)),fiber(Real(ddfdxdy⋅n)),fiber(Real(ddfdy2⋅n))
    TensorOperator.(Chain{V}.(Chain{V}.(L,M),Chain{V}.(M,N)))
end

firstform(dom::ScalarField,f::Function) = firstform(TensorField(dom,f))
function firstform(t::ScalarField,g=gradient(t),V=Submanifold(2))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    E,F,G = fiber(dfdx*dfdx),fiber(dfdx*dfdy),fiber(dfdy*dfdy)
    TensorField(base(t),TensorOperator.(Chain{V}.(Chain{V}.(E.+1,F),Chain{V}.(F,G.+1))))
end

firstformdiag(dom::ScalarField,f::Function) = firstformdiag(TensorField(dom,f))
function firstformdiag(t::ScalarField,g=gradient(t),V=Submanifold(2))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    E1,G1 = fiber(1+dfdx*dfdx),fiber(1+dfdy*dfdy)
    TensorField(base(t),DiagonalOperator.(Chain{V}.(E1,G1)))
end

firstform(dom,f::Function) = firstform(TensorField(dom,f))
function firstform(t,g=gradient(t),V=Submanifold(2))
    TensorField(base(t),EFG(V,getindex.(g,1),getindex.(g,2)))
end

firstformdiag(dom,f::Function) = firstformdiag(TensorField(dom,f))
function firstformdiag(t,g=gradien(t),V=Submanifold(2))
    TensorField(base(t),EG(V,getindex.(g,1),getindex.(g,2)))
end

secondform(dom,f::Function) = secondform(TensorField(dom,f))
function secondform(t,g=gradient(t),V=Submanifold(2))
    n = ⋆unittangent(t,∧(g))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    ddfdx,ddfdy2 = gradient(dfdx),gradient(dfdy,Val(2))
    ddfdx2,ddfdxdy = getindex.(ddfdx,1),getindex.(ddfdx,2)
    TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2))
end

firstsecondform(dom,f::Function) = firstsecondform(TensorField(dom,f))
function firstsecondform(t,g=gradient(t),V=Submanifold(2))
    n = ⋆unittangent(t,∧(g))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    ddfdx,ddfdy2 = gradient(dfdx),gradient(dfdy,Val(2))
    ddfdx2,ddfdxdy = getindex.(ddfdx,1),getindex.(ddfdx,2)
    return (TensorField(base(t),EFG(V,dfdx,dfdy)),
        TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2)))
end

thirdform(dom,f::Function) = thirdform(TensorField(dom,f))
thirdform(t,V=Submanifold(2)) = firstform(t,gradient(unitnormal(t)),V)

shape(dom,f::Function) = shape(TensorField(dom,f))
function shape(t,g=gradient(t),V=Submanifold(2))
    EFG,LMN = firstsecondform(t,g,V)
    return inv(EFG)⋅LMN
end

surfacemetric(dom,f::Function) = surfacemetric(TensorField(dom,f))
function surfacemetric(t::DiagonalField)
    GridBundle(PointArray(points(t),fiber(outermorphism(t))),immersion(t))
end
function surfacemetric(t::EndomorphismField)
    GridBundle(PointArray(points(t),fiber(Outermorphism(t))),immersion(t))
end
function surfacemetric(t,g=gradient(t))
    V = Submanifold(MetricTensor([1 1; 1 1]))
    EFG = Outermorphism.(fiber(firstform(t,g,V)))
    GridBundle(PointArray(points(t),EFG),immersion(t))
end

surfacemetricdiag(dom,f::Function) = surfacemetricdiag(TensorField(dom,f))
surfacemetricdiag(t::DiagonalField) = surfacemetric(t)
function surfacemetricdiag(t,g=gradient(t))
    V = Submanifold(DiagonalForm(Values(1,1)))
    EG = outermorphism.(fiber(firstformdiag(t,g,V)))
    GridBundle(PointArray(points(t),EG),immersion(t))
end

surfaceframe(t::DiagonalField) = surfaceframediag(t)
surfaceframe(t::FrameBundle) = surfaceframe(metrictensorfield(t))
function surfaceframe(t::EndomorphismField)
    surfaceframe(base(t),getindex.(t,1,1),getindex.(t,1,2),getindex.(t,2,2))
end
function surfaceframe(t,g=gradient(t))
    dx,dy = getindex.(g,1),getindex.(g,2)
    surfaceframe(base(t),fiber(Real(dx⋅dx)),fiber(Real(dx⋅dy)),fiber(Real(dy⋅dy)))
end
function surfaceframe(b,E,F,G)
    F2 = F.*F; mag,sig = sqrt.((E.*E).+F2), sign.(F2.-(E.*G))
    TensorField(b,TensorOperator.(Chain.(Chain.(E,F)./mag,Chain.(F,.-E)./(sig.*mag))))
end

surfaceframediag(t) = surfaceframediag(firstformdiag(t))
surfaceframediag(t::FrameBundle) = surfaceframediag(metrictensorfield(t))
function surfaceframediag(t::DiagonalField)
    E,G = getindex.(value.(fiber(g)),1),getindex.(value.(fiber(g)),2)
    mag,sig = sqrt.(E.*E), sign.(.-(E.*G))
    TensorField(base(t),DiagonalOperator.(Chain.(E./mag,.-E./(sig.*mag))))
end

_firstkind(dg,k,i,j) = dg[k,j][i] + dg[i,k][j] - dg[i,j][k]
firstkind(g::FrameBundle) = firstkind(metrictensorfield(g))
firstkind(g::TensorField) = TensorField(base(g),firstkind.(d(g/2)))
firstkind(dg::DiagonalOperator,i,j,k) = _firstkind(dg,k,i,j)
@generated function firstkind(dg,i,j,k)
    Expr(:call,:+,[:(_firstkind(dg,$l,i,j)) for l ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg,i,j)
    Expr(:call,:Chain,[:(firstkind(dg,i,j,$k)) for k ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg,j)
    Expr(:call,:Chain,[:(firstkind(dg,$i,j)) for i ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg)
    Expr(:call,:TensorOperator,Expr(:call,:Chain,[:(firstkind(dg,$j)) for j ∈ list(1,mdims(fibertype(dg)))]...))
end

secondkind(g::FrameBundle) = secondkind(metrictensorfield(g))
secondkind(g::TensorField) = TensorField(base(g),secondkind.(inv(g),d(g/2)))
secondkind(ig::DiagonalOperator,dg,i,j,k) = ig[k,k]*_firstkind(dg,k,i,j)
@generated function secondkind(ig,dg,i,j,k)
    Expr(:call,:+,[:(ig[$l,k]*_firstkind(dg,$l,i,j)) for l ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg,i,j)
    Expr(:call,:Chain,[:(secondkind(ig,dg,i,j,$k)) for k ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg,j)
    Expr(:call,:Chain,[:(secondkind(ig,dg,$i,j)) for i ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg)
    Expr(:call,:TensorOperator,Expr(:call,:Chain,[:(secondkind(ig,dg,$j)) for j ∈ list(1,mdims(fibertype(dg)))]...))
end

geodesic(Γ) = x -> geodesic(x,Γ)
geodesic(g::FrameBundle) = geodesic(secondkind(g))
geodesic(x,Γ::Function) = (x2 = x[2]; Chain(x2,-geodesic(x2,Γ(x[1]))))
geodesic(x,Γ::TensorField) = (x2 = x[2]; Chain(x2,-geodesic(x2,Γ(x[1]))))
@generated function geodesic(x::Chain{V,G,T,N} where {V,G,T},Γ) where N
    Expr(:call,:+,vcat([[:(Γ[$i,$j]*(x[$i]*x[$j])) for i ∈ list(1,N)] for j ∈ list(1,N)]...)...)
end
@generated function metricscale(x::Chain{V,G,T,N} where {G,T},g::Simplex) where {V,N}
    Expr(:call,:(Chain{V}),[:(x[$k]*sqrt(g[$k,$k])) for k ∈ list(1,N)]...)
end

#export beta, betafunction

function beta(a,b,n=30000)
    x = OpenParameter(n)
    integrate(x^(a-1)*(1-x)^(b-1))
end
function betafunction(a,b,n=100)
    x = OpenParameter(n)
    integral(x^(a-1)*(1-x)^(b-1))
end

