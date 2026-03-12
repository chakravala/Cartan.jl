
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
boundabove(t::TensorField,lim=10) = TensorField(base(t), boundabove.(fiber(t),lim))
boundbelow(s::LocalTensor{B,T},lim=-10) where {B,T<:Real} = fiber(s)≥lim ? s : LocalTensor(base(s),T(lim))
boundbelow(x::T,lim=-10) where T<:Real = x≥lim ? x : T(lim)
boundbelow(t::TensorField,lim=-10) = TensorField(base(t), boundbelow.(fiber(t),lim))
bound(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = abs(fiber(s))≤lim ? s : LocalTensor(base(s),T(sign(fiber(s)*lim)))
bound(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : ((lim/x)*s))
bound(x::T,lim=10) where T<:Real = abs(x)≤lim ? x : T(sign(x)*lim)
bound(z,lim=10) = (x=abs(z); x≤lim ? z : (lim/x)*z)
bound(t::TensorField,lim=10) = TensorField(base(t), bound.(fiber(t),lim))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(sign(fiber(s))*(lim+log(x+1-lim)))))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(z::T,lim=10) where T<:Real = (x=abs(z); x≤lim ? z : T(sign(z)*(lim+log(x+1-lim))))
boundlog(z,lim=10) = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(t::TensorField,lim=10) = TensorField(base(t), boundlog.(fiber(t),lim))

isclosed(t::IntervalMap) = norm(fiber(t)[end]-fiber(t)[1]) ≈ 0
updatetopology(t::IntervalMap) = isclosed(t) ? TorusTopology(t) : t

#(::Derivation)(t::TensorField) = getnabla(t)
function getnabla(t::TensorField)
    W = Manifold(t)
    n = ndims(t)
    V = tangent(Grassmann.supermanifold(W),1,n)(list(n+1,2n))
    Chain{W}(Values{n,Any}(Λ(V).b[2:n+1]...))
end

export invd, cartan, firststructure, secondstructure

cartan(ξ) = invd(ξ)⋅ξ
firststructure(θ,ω) = d(θ)+ω∧θ
secondstructure(ω) = d(ω)+ω∧ω

Base.div(t::TensorField) = divergence(t)
divergence(t::TensorField) = ∂(t)
Grassmann.curl(t::TensorField) = ⋆d(t)
Grassmann.δ(t::TensorField) = -∂(t)
Grassmann.d(t::TensorField) = TensorField(fromany(getnabla(t)∧Chain(t)))
Grassmann.∂(t::TensorField) = TensorField(fromany(Chain(t)⋅getnabla(t)))
Grassmann.d(t::ScalarField{B,<:AbstractReal,N,<:FrameBundle,<:AbstractArray} where {B,N}) = gradient(t)
#Grassmann.∂(t::ScalarField) = gradient(t)
#Grassmann.∂(t::VectorField) = TensorField(base(t), sum.(value.(fiber(t))))
#=function Grassmann.∂(t::VectorField{G,B,<:Chain{V},N,T} where {B,N,T}) where {G,V}
    n = mdims(V)
    TensorField(base(t), Real.(Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅fiber(t)))
end
function Grassmann.∂(t::GradedField{G,B,<:Chain{V}} where B) where {G,V}
    n = mdims(V)
    TensorField(base(t), (Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅fiber(t)))
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
        $op(::Nabla,t::TensorField) = TensorField(fromany($op(getnabla(t),Chain(t))))
        $op(t::TensorField,::Nabla) = TensorField(fromany($op(Chain(t),getnabla(t))))
    end
end
LinearAlgebra.dot(::Nabla,t::TensorField) = TensorField(fromany(Grassmann.contraction(getnabla(t),Chain(t))))
LinearAlgebra.dot(t::TensorField,::Nabla) = TensorField(fromany(Grassmann.contraction(Chain(t),getnabla(t))))

# need AbstractScalar + AbstractReal
Grassmann.:∧(::Nabla,t::ScalarField) = gradient(t)
Grassmann.:∧(t::ScalarField,::Nabla) = gradient(t)
Grassmann.:∧(::Laplacian,t::ScalarField) = _laplacian(t)
Grassmann.:∧(t::ScalarField,::Laplacian) = _laplacian(t)
Base.:*(::Nabla,t::ScalarField) = gradient(t)
Base.:*(t::ScalarField,::Nabla) = gradient(t)
Base.:*(::Laplacian,t::ScalarField) = _laplacian(t)
Base.:*(t::ScalarField,::Laplacian) = _laplacian(t)
LinearAlgebra.dot(::Nabla,t::ScalarField) = gradient(t)
LinearAlgebra.dot(t::ScalarField,::Nabla) = gradient(t)
LinearAlgebra.dot(::Laplacian,t::ScalarField) = _laplacian(t)
LinearAlgebra.dot(t::ScalarField,::Laplacian) = _laplacian(t)
#=LinearAlgebra.dot(::Nabla,t::GradedField) = ∂(t)
LinearAlgebra.dot(t::GradedField,::Nabla) = ∂(t)
function LinearAlgebra.dot(t::GradedField{G,B,<:Chain{V}} where B) where {G,V}
    n = mdims(V)
    TensorField(base(t), Real.(fiber(t).⋅Chain{V,G}(ones(Values{binomial(n,G),Int}))))
end
function LinearAlgebra.dot(t::GradedField{G,B,<:Chain{V}} where B) where {G,V}
    n = mdims(V)
    TensorField(base(t), (fiber(t).⋅
end=#
function Base.:*(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(base(t), (Ref(n).*fiber(t)))
    end
end
function Base.:*(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(base(t), (fiber(t).*Ref(n)))
    end
end
function LinearAlgebra.dot(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(base(t), dot.(Ref(n),fiber(t)))
    end
end
function LinearAlgebra.dot(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(base(t), dot.(fiber(t),Ref(n)))
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
action(X::VectorField,f::ScalarField) = Real(X⋅gradient(f))
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

#=export Riemann

struct Riemann{X,Y,XY}
    ∇x::X
    ∇y::Y
    ∇xy::XY
    Riemann(∇x::X,∇y::Y,∇xy::XY) where {X<:CovariantDerivative,Y<:CovariantDerivative,XY<:CovariantDerivative} = new{X,Y,XY}(∇x,∇y,∇xy)
end

#Riemann2() = (𝓛[X,Y](Z) + (Ω⋅(X∧Y))⋅Z
Riemann(∇::Connection,X::VectorField,Y::VectorField) = Riemann(∇(X),∇(Y),∇(𝓛[X,Y]))
function (R::Riemann)(Z::VectorField)
    ∇yZ = R.∇y.v(Z)+(R.∇y.ωv⋅Z)
    ∇xZ = R.∇x.v(Z)+(R.∇x.ωv⋅Z)
    (R.∇x.v(∇yZ)+(R.∇x.ωv⋅∇yZ)) - (R.∇y.v(∇xZ)+(R.∇y.ωv⋅∇xZ) + R.∇xy.v(Z)+(R.∇xy.ωv⋅Z))
end=#

export arclength, arctime, totalarclength, trapz, cumtrapz, linecumtrapz, psum, pcumsum
export centraldiff, tangent, tangent_fast, unittangent, speed, normal, unitnormal
export arcparametrize, arcresample, arcsample, ribbon, tangentsurface, planecurve
export spherearea, ballvolume, degreeintegrate, degreeintegrate_slow, link, linkmap
export normalnorm, area, surfacearea, weingarten, gausssign, jacobian, evolute, involute
export normalnorm_slow, area_slow, surfacearea_slow, weingarten_slow, gausssign_slow
export gaussintrinsic, gaussextrinsic, gaussintrinsicnorm, gaussextrinsicnorm
export gaussintrinsic_slow, gausseintrinsicnorm_slow, curvatures, meancurvature
export gaussextrinsic_slow, gaussextrinsicnorm_slow, principals, principalaxes
export tangent_slow, normal_slow, unittangent_slow, unitnormal_slow, jacobian_slow
export ruledsurface, linedsurface, scrollsurface, revolve, revolvesphere, revolvesector
export sector, sectordet, sectorintegral, sectorintegrate, linkintegral, linknumber
export sector_slow, sectordet_slow, sectorintegral_slow, sectorintegrate_slow
export indexintegral, indexintegrate, indexintegral_slow, indexintegrate_slow
export unitjacobian, unitjacobian_slow, principal, principalbasetype, principalfibertype
export PrincipalFiber, LocalPrincipal, principalbase, principalfiber, principalnorm
export TangentBundle, Pullback, NormalBundle, principalbundle, principalaction
export UnitTangentBundle, UnitNormalBundle, UnitPullback, PathNormalBundle, unitcone
export unitcircle, unithelix, unitsphere, unitdisk, unitball, unitpipe, unitcylinder
export unitconic, conoid, rightconoid, sectorize, cylinderize, sphereradius, ballradius
export riemannsphere, riemannline

principal(x::LocalPrincipal) = principalfiber(x)
principalnorm(x::LocalPrincipal) = Real(abs(det(principalfiber(x))))
principalbase(x::LocalPrincipal) = base(x)
principalfiber(x::LocalPrincipal) = fiber(x)
principalbasetype(x::LocalPrincipal) = basetype(x)
principalfibertype(x::LocalPrincipal) = fibertype(x)

Base.inv(T::LocalPrincipal) = LocalPrincipal(principalbase(T),inv(principalfiber(T)))

(P::LocalPrincipal)(f) = principalaction(P,f)
(P::LocalPrincipal)() = principalaction(P)
(P::PrincipalFiber)(f) = principalaction(P,f)
(P::PrincipalFiber)() = principalaction(P)

principalaction(P::LocalPrincipal,f::TensorField) = select_action(P,f(principalbase(P)))
principalaction(P::LocalPrincipal,f::Function) = select_action(P,f(principalbase(P)))
principalaction(P::LocalPrincipal,f::Real) = select_action(P,f)
principalaction(P::LocalPrincipal,f::Complex) = select_action(P,f)
principalaction(P::LocalPrincipal,f::TensorAlgebra) = select_action(P,f)
principalaciton(P::LocalPrincipal) = principal(P)

principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap,<:AbstractCurve} where {M,G,N},f::TensorField{B,<:Chain,1,<:Interval} where B) = principalnorm(P)*pre_action(P,f)
#principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap,<:AbstractCurve} where {M,G,N},f::Function) = principalnorm(P)*f.(base(P))
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap,<:AbstractCurve} where {M,G,N},f::Real) = principalnorm(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap,<:AbstractCurve} where {M,G,N},f::Complex) = principalnorm(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap,<:AbstractCurve} where {M,G,N},f::TensorGraded{0}) = principalnorm(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::TensorField{B,<:AbstractReal} where B) = principalnorm(P)*pre_action(P,f)
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::TensorField) = principal(P)⋅pre_action(P,f)
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::Function) = principal(P)⋅f.(base(P))
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::Real) = principal(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::Complex) = principal(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::TensorGraded{0}) = principal(P)*f
principalaction(P::PrincipalFiber{M,G,N,<:IntervalMap} where {M,G,N},f::TensorAlgebra) = principal(P)⋅f
principalaction(P::PrincipalFiber,f::TensorField) = select_action(P,pre_action(P,f))
principalaction(P::PrincipalFiber,f::Function) = select_action(P,f.(base(P)))
principalaction(P::PrincipalFiber,f::Real) = select_action(P,f)
principalaction(P::PrincipalFiber,f::Complex) = select_action(P,f)
principalaction(P::PrincipalFiber,f::TensorAlgebra) = select_action(P,f)
principalaction(P::PrincipalFiber) = principal(P)
principalaction(P::FrameBundle) = principal(P)
function principalaction(P::FrameBundle,g)
    f = pre_action2(P,g)
    if isinduced(P)
        f
    else
        p = principal(P)
        TensorField(base(f),fiber(f).*sqrt.(abs.(Real.(det.(fiber(p))))))
    end
end
principalaction(t::TensorField) = isextrinsic(t)||!isinduced(t) ? principalaction(base(t),TensorField(principalbundle(t),fiber(t))) : t

principalaction(t,d,f) = principalaction(PrincipalFiber(t,_outermorphism(d)),f)
principalaction(t,d,f::Real) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::Complex) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::TensorGraded{0}) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::TensorGraded{1}) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::TensorGraded{G}) where G = principalaction(PrincipalFiber(t,compound(d,Val(G))),f)
principalaction(t,d,f::ScalarField) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::GradedField{0}) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::GradedField{1}) = principalaction(PrincipalFiber(t,d),f)
principalaction(t,d,f::GradedField{G}) where G = principalaction(PrincipalFiber(t,compound(d,Val(G))),f)

select_action(P,f) = principal(P)⋅f
select_action(P,f::Real) = principalnorm(P)*f
select_action(P,f::Complex) = principalnorm(P)*f
select_action(P,f::TensorGraded{0}) = principalnorm(P)*f
select_action(P,f::ScalarField) = principalnorm(P)*f
pre_action(P,f) = coordinates(f) == coordinates(P) ? f : f.(base(P))
pre_action2(P,f) = points(f) == points(P) ? f : f.(base(P))

principal(P::PrincipalFiber) = fiber(P)
principal(P::FrameBundle) = metrictensorfield(P)
function principalnorm(P::PrincipalFiber)
    p = principal(P)
    TensorField(base(p),Real.(abs.(det.(fiber(p)))))
end
function principalnorm(P::FrameBundle)
    p = principal(P)
    TensorField(base(p),sqrt.(abs.(Real.(det.(fiber(p))))))
end
principalbundle(P::PrincipalFiber) = base(base(P))
principalbundle(P::FrameBundle) = P#GridBundle(PointArray(0,points(P)),immersion(P))
principalbundle(P::TensorField) = principalbundle(base(P))
principalbase(P::PrincipalFiber) = fiber(base(P))
principalfiber(P::PrincipalFiber) = fiber(principal(P))
principalfiber(P::FrameBundle) = fiber(principal(P))
principalbasetype(P::PrincipalFiber) = fibertype(base(P))
principalfibertype(P::PrincipalFiber) = fibertype(principal(P))
principalfibertype(P::FrameBundle) = fibertype(principal(P))
localfiber(P::PrincipalFiber,x::GradedVector) = PrincipalFiber(base(P)(x),fiber(P)(x))

Base.inv(P::PrincipalFiber) = PrincipalFiber(base(P),inv(principal(P)))

_outermorphism(x::VectorField) = x
_outermorphism(x::ScalarField) = x
_outermorphism(x) = Outermorphism(x)

UnitTangentBundle(t,d=unitjacobian(t)) = PrincipalFiber(t,_outermorphism(d))
TangentBundle(t,d=jacobian(t)) = PrincipalFiber(t,_outermorphism(d))
PathTangentBundle(t,i::AbstractCurve) = PathTangentBundle(TangentBundle(t),i)
PathTangentBundle(P::PrincipalFiber,i::AbstractCurve) = PrincipalFiber(base(P).(i), _outermorphism(_tangent(fiber(P).(i))))
UnitPullback(t,d=unitjacobian(t)) = PrincipalFiber(t,_outermorphism(inv(d)))
Pullback(t,d=jacobian(t)) = PrincipalFiber(t,_outermorphism(inv(d)))
UnitNormalBundle(t,d=normalunitframe(t)) = PrincipalFiber(t,_outermorphism(d))
NormalBundle(t,d=normalframe(t)) = PrincipalFiber(t,_outermorphism(d))
PathNormalBundle(t,i::AbstractCurve) = PathNormalBundle(NormalBundle(t),i)
PathNormalBundle(P::PrincipalFiber,i::AbstractCurve) = PrincipalFiber(base(P).(i), _outermorphism(_normalframe(fiber(P).(i))))

unittangentaction(f,t,d=unitjacobian(t)) = principalaction(t,d,f)
tangentaction(f,t,d=jacobian(t)) = principalaction(t,d,f)
unitpullbackaction(f,t,d=unitjacobian(t)) = principalaction(t,d,f)
pullbackaction(f,t,d=jacobian(t)) = principalaction(t,d,f)
unitnormalaction(f,t,d=normalunitframe(t)) = principalaction(t,d,f)
normalaction(f,t,d=normalframe(t)) = principalaction(t,d,f)

export transport, paralleltransport, retract
retract(f,x=-normal(f)) = inv(normalframe(f))⋅(x-f)
retract(P::PrincipalFiber,f=-normal(base(P))) = inv(P)(f-base(P))
transport(f,x) = normalunitframe(f)⋅x + f
transport(P::PrincipalFiber,f) = P(f) + base(P)
function paralleltransport(P::PrincipalFiber,f::IntervalMap)
    X = base(f)⊕principalbundle(P)
    out = Array{principalbasetype(P),2}(undef,length(P),length(f))
    for i ∈ 1:length(f)
        assign!(out,i,fiber(transport(P,fiber(f)[i])))
    end
    return TensorField(X,out)
end

for fun ∈ (:trapz,:cumtrapz)
    for typ ∈ (:TensorField,:Function,:AbstractFloat)
        @eval begin
            $fun(ϕ::TensorField,f::$typ) =  $fun(tangentaction(f,ϕ))
            $fun(ϕ::TensorField,f::$typ,Ω) =  $fun(Ω*tangentaction(f,ϕ))
        end
    end
end
for (flx,fun) ∈ ((:fluxintegral,:cumtrapz),(:fluxintegrate,:trapz))
    for typ ∈ (:TensorField,:Function,:AbstractFloat)
        @eval begin
            $flx(N::TensorField,f::$typ) =  $fun(normalaction(f,N))
            $flx(N::TensorField,f::$typ,Ω) =  $fun(Ω*normalaction(f,N))
        end
    end
end
const ⨍ = fluxintegral
export fluxintegral, fluxintegrate, ⨍

# use graph for IntervalMap? or RealFunction!
(::Laplacian)(f::ScalarField) = _laplacian(f)
_laplacian(f::ScalarField) = tr(jacobian(gradient(f)))
tangent(f::IntervalMap) = gradient(f)
tangent(f::ScalarField) = tangent(graph(f))
tangent(f::VectorField) = ∧(gradient(f))
normal(f::ScalarField) = ⋆tangent(f)
normal(f::VectorField) = ⋆tangent(f)
normalframe(f) = normal(f)
unittangent(f::ScalarField,n=tangent(f)) = unit(n)
unittangent(f::VectorField,n=tangent(f)) = unit(n)
unittangent(f::IntervalMap) = unitgradient(f)
unitnormal(f) = ⋆unittangent(f)
normalunitframe(f) = unitnormal(f)
normalnorm(f) = Real(abs(det(normalframe(f))))
tangentnorm(f) = Real(abs(det(jacobian(f))))
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
indexintegral(f::RealFunction) = integral(f)
indexintegral(f::TensorField) = integral(sectordet(f))/spherearea(mdims(fibertype(f)))
indexintegrate(f::RealFunction) = integrate(f)
indexintegrate(f::TensorField) = integrate(sectordet(f))/spherearea(mdims(fibertype(f)))
degreeintegrate(f,ω::TensorField) = integrate(f,ω)/integrate(ω)
degreeintegrate(f,ω::Real=1) = integrate(f,float(ω))/integrate(TensorField(f,ω))
sector_slow(f::RealFunction) = f
sector_slow(f::TensorField) = TensorOperator(sector(f,gradient_slow(f)))
sectordet_slow(f::RealFunction) = f
sectordet_slow(f::PlaneCurve) = f∧gradient_slow(f)
sectordet_slow(f::VectorField) = f∧(∧(gradient_slow(f)))
sectorintegral_slow(f::RealFunction) = integral(f)
sectorintegral_slow(f::TensorField) = integral(sectordet_slow(f))/mdims(fibertype(f))
sectorintegrate_slow(f::RealFunction) = integrate(f)
sectorintegrate_slow(f::TensorField) = integrate(sectordet_slow(f))/mdims(fibertype(f))
indexintegral_slow(f::RealFunction) = integral_slow(f)
indexintegral_slow(f::TensorField) = integral_slow(sectordet_slow(f))/spherearea(mdims(fibertype(f)))
indexintegrate_slow(f::RealFunction) = integrate_slow(f)
indexintegrate_slow(f::TensorField) = integrate_slow(sectordet_slow(f))/spherearea(mdims(fibertype(f)))
degreeintegrate_slow(f,ω::TensorField) = integrate_slow(f,ω)/integrate_slow(ω)
degreeintegrate_slow(f,ω::Real=1) = integrate_slow(f,float(ω))/integrate_slow(TensorField(f,ω))

area(f::VectorField) = integral(normalnorm(f))
surfacearea(f::VectorField) = integrate(normalnorm(f))
surfacearea(f::ElementBundle) = sum(fiber(volumes(f)))
surfacearea(f::ScalarMap) = surfacearea(base(graphbundle(f)))
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

sphereradius(n,a=1) = (a/spherearea(n))^inv(n-1)
ballradius(n,v=1) = (v/ballvolume(n))^inv(n)

spherearea(n,r) = spherearea(n)*r^(n-1)
spherearea(n::Int) = ballvolume(n)*n
ballvolume(n,r) = ballvolume(n)*r^n
function ballvolume(n::Int)
    k = n ÷ 2
    if iseven(n)
        π^k/factorial(k)
    else
        2factorial(k)*(4π)^k/factorial(n)
    end
end

#=function arcparameter(f::IntervalMap,d=centraldiff(points(f)),t=centraldiff(fiber(f),d))
    cumtrapz(TensorField(unitdomain(f)*sum(value.(abs.(diff(fiber(f))))), t./abs.(t)))
end
function arctime(f::IntervalMap,d=centraldiff(points(f)),t=centraldiff(fiber(f),immersion(f),d))
    al = cumtrapz(arclength(f))
    out = cumtrapz(TensorField(value.(fiber(al)), inv.(abs.(t))))
    TensorField(base(out), (fiber(out).*(base(f)[end]/value(fiber(out)[end]))))
end
function arctime(f::IntervalMap)
    al = arclength(f)
    ad = arcdomain(f)
    at = TensorField(value.(fiber(al)), collect(points(f)))
    l = length(ad)
    TensorField(ad, [i∈(1,l) ? points(f)[i] : at(ad[i]) for i ∈ 1:l])
end
function arctime(f::IntervalMap,d=centraldiff(points(f)),t=centraldiff(fiber(f),immersion(f),d))
    #base(out) → (fiber(out).*(points(f)[end]/value(fiber(out)[end])))
    al = arclength(f)
    ad = arcdomain(f)
    at = TensorField(value.(fiber(al)), inv.(abs.(t)))
    l = length(ad)
    cumtrapz(TensorField(ad, [i==1 ? at(ad[i]+eps()) : i==l ? at(ad[i]-eps()) : at(ad[i]) for i ∈ 1:l]))
end=#

function speed(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(base(f), Real.(abs.(t,refmetric(f))))
end
function normal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(base(f), centraldiff(t./s,immersion(f),d)./s)
end
function unitnormal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(base(f), unit.(centraldiff(unit.(t,refmetric(f)),immersion(f),d),refmetric(f)))
end

export curvature, radius, bending, osculatingplane, unitosculatingplane
export binormal, unitbinormal, torsion, frame, unitframe, normalframe, normalunitframe
export frenet, darboux, darbouxframe, darbouxunitframe, wronskian, unitwronskian
export bishoppolar, bishop, bishopframe, bishopunitframe, normalangle

function curvature(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(base(f), Real.(abs.(centraldiff(t./s,immersion(f),d),refmetric(f))./s))
end
function radius(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(base(f), Real.(s./abs.(centraldiff(t./s,immersion(f),d),refmetric(f))))
end
function _bending(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    TensorField(base(f), Real.(abs2.(centraldiff(t./s,immersion(f),d),refmetric(f))./s))/2
end
bending(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = integral(_bending(f,d,t))
bendingenergy(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = integrate(_bending(f,d,t))
function localevolute(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = abs.(t,refmetric(f))
    n = centraldiff(t./s,immersion(f),d)
    an2 = abs2.(n,refmetric(f))
    TensorField(base(f), (s./an2).*n)
end
evolute(f::AbstractCurve) = f+localevolute(f)
Grassmann.involute(f::AbstractCurve) = f-unittangent(f)*arclength(f)
function Grassmann.involute(f::AbstractCurve,l)
    s = arclength(f)
    f-unittangent(f)*(s-s(float(l)))
end
function osculatingplane(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(base(f), TensorOperator.(Chain.(t,fiber(normal(f,d,t)))))
end
function unitosculatingplane(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    T = unit.(t,refmetric(f))
    TensorField(base(f),TensorOperator.(Chain.(T,unit.(centraldiff(T,immersion(f),d),refmetric(f)))))
end
function binormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(base(f), .⋆(t.∧fiber(normal(f,d,t)),refmetric(f)))
end
function unitbinormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    T = unit.(t)
    TensorField(base(f), .⋆(T.∧(unit.(centraldiff(T,immersion(f),d),refmetric(f))),refmetric(f)))
end
function torsion(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,immersion(f),d)
    b = t.∧n
    TensorField(base(f), Real.((b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆b,refmetric(f))))
end
#=function curvaturetrivector(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,immersion(f),d),b=t.∧n)
    a=abs.(t,refmetric(f)); ut=t./a
    TensorField(base(f), (abs2.(centraldiff(ut,immersion(f),d)./a)).*(b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆b))
end=#
#torsion(f::TensorField,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,immersion(f),d),a=abs.(t)) = TensorField(base(f), abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),immersion(f),d))./a)
frame(f::AbstractCurve...) = TensorOperator.(Chain.(f...))
frame(f::PlaneCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d)) = osculatingplane(f,d,e1)
@generated function frame(f::AbstractCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(s = abs.(e1,refmetric(f))),
        [:($(syms[i]) = centraldiff(unit.($(syms[i-1]),refmetric(f)),immersion(f),d)./s) for i ∈ list(2,N-1)]...,
        :(TensorField(base(f),TensorOperator.(Chain.($(syms...),.⋆(.∧($(syms...)),refmetric(f)))))))
end
unitframe(f::AbstractCurve...) = frame(unit.(f)...)
unitframe(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = unitosculatingplane(f,d,t)
@generated function unitframe(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(e1 = unit.(t,refmetric(f))),
        [:($(syms[i]) = unit.(centraldiff($(syms[i-1]),immersion(f),d),refmetric(f))) for i ∈ list(2,N-1)]...,
        :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(f))),
        :(TensorField(base(f),TensorOperator.(Chain.($([:($(Symbol(:e,i))) for i ∈ list(1,N)]...))))))
end
darbouxframe(f::AbstractCurve...) = TensorOperator.(Chain.(f...))
darbouxframe(f::RealFunction...) = darbouxframe(Chain.(f...))
darbouxframe(f::RealFunction,d=nothing,e1=nothing) = f
@generated function darbouxframe(f::AbstractCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,
        [:($(syms[i]) = centraldiff($(syms[i-1]),immersion(f),d)) for i ∈ list(2,N-1)]...,
        :(TensorField(base(f),TensorOperator.(Chain.(fiber(f),$(syms...))))))
end
darbouxunitframe(f::AbstractCurve...) = darbouxframe(unit.(f)...)
darbouxunitframe(f::RealFunction...) = darbouxunitframe(Chain.(f...))
darbouxunitframe(f::RealFunction,d=nothing,e1=nothing) = unit(f)
@generated function darbouxunitframe(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N))
    Expr(:block,:(df = fiber(darbouxframe(f,d,t))),:(e1 = unit.(fiber(f),refmetric(f))),
        [:($(syms[i]) = unit.(getindex.(df,$i),refmetric(f))) for i ∈ list(2,N)]...,
        :(TensorField(base(f),TensorOperator.(Chain.($(syms...))))))
end
wronskian(f::RealFunction...) = wronskian(Chain.(f...))
wronskian(f::AbstractCurve...) = det.(TensorOperator.(Chain.(f...)))
wronskian(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = det(darbouxframe(f,d,t))
unitwronskian(f::RealFunction...) = unitwronskian(Chain.(f...))
unitwronskian(f::AbstractCurve...) = wronskian(unit.(f)...)
unitwronskian(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = det(darbouxunitframe(f,d,t))
normalframe(f::PlaneCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d)) = normal(f,d,e1)
normalframe(f::AbstractCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d)) = _normalframe(TensorField(base(f),e1),d,e1)
#=@generated function normalframe(f::AbstractCurve,d=centraldiffpoints(f),e1=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(s = abs.(e1,refmetric(f))),
        [:($(syms[i]) = centraldiff(unit.($(syms[i-1]),refmetric(f)),immersion(f),d)./s) for i ∈ list(2,N-1)]...,
        :(TensorField(base(f),TensorOperator.(Chain.($(syms[2:end]...),.⋆(.∧($(syms...)),refmetric(f)))))))
end=#
normalunitframe(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = unitnormal(f,d,t)
normalunitframe(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d)) = _normalunitframe(TensorField(base(f),t),d,t)
#=@generated function normalunitframe(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    N = mdims(fibertype(f))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(e1 = unit.(t,refmetric(f))),
        [:($(syms[i]) = unit.(centraldiff($(syms[i-1]),immersion(f),d),refmetric(f))) for i ∈ list(2,N-1)]...,
        :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(f))),
        :(TensorField(base(f),TensorOperator.(Chain.($([:($(Symbol(:e,i))) for i ∈ list(2,N)]...))))))
end=#
function _normal(t::IntervalMap,d=centraldiffpoints(t))
    s = fiber(abs(t))
    TensorField(base(t), centraldiff(fiber(t)./s,immersion(t),d)./s)
end
function _unitnormal(t::IntervalMap,d=centraldiffpoints(t))
    TensorField(base(t), unit.(centraldiff(unit.(fiber(t),refmetric(t)),immersion(t),d),refmetric(t)))
end
_normalframe(t::PlaneCurve,d=centraldiffpoints(t),e1=fiber(t)) = _normal(f,d)
@generated function _normalframe(t::AbstractCurve,d=centraldiffpoints(t),e1=fiber(t))
    N = mdims(fibertype(t))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(s = abs.(e1,refmetric(t))),
        [:($(syms[i]) = centraldiff(unit.($(syms[i-1]),refmetric(t)),immersion(t),d)./s) for i ∈ list(2,N-1)]...,
        :(TensorField(base(t),TensorOperator.(Chain.($(syms[2:end]...),.⋆(.∧($(syms...)),refmetric(t)))))))
end
_normalunitframe(t::PlaneCurve,d=centraldiffpoints(t),e1=fiber(t)) = _unitnormal(t,d)
@generated function _normalunitframe(T::AbstractCurve,d=centraldiffpoints(T),t=fiber(T))
    N = mdims(fibertype(T))
    syms = Symbol.(:e,list(1,N-1))
    Expr(:block,:(e1 = unit.(t,refmetric(T))),
        [:($(syms[i]) = unit.(centraldiff($(syms[i-1]),immersion(T),d),refmetric(T))) for i ∈ list(2,N-1)]...,
        :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(T))),
        :(TensorField(base(T),TensorOperator.(Chain.($([:($(Symbol(:e,i))) for i ∈ list(2,N)]...))))))
end

function cartan(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    κ = curvature(f,d,t)
    TensorField(base(f), Chain.(Chain.(0,κ),Chain.(.-κ,0)))
end
function cartan(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,immersion(f),d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,immersion(f),d),refmetric(f))./s)
    τ = Real.((b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆(b,refmetric(f)),refmetric(f)))
    TensorField(base(f),TensorOperator.(Chain.(Chain.(0,κ,0),Chain.(.-κ,0,τ),Chain.(0,.-τ,0))))
end
@generated function cartan(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    N = mdims(V)
    bas = Grassmann.bladeindex.(N,UInt.(Λ(V).b[list(2,N)].∧Λ(V).b[list(3,N+1)]))
    syms,curvs = Symbol.(:e,list(1,N)),Symbol.(:κ,list(1,N-1))
    vals = [:($(curvs[i]) = Real.(Grassmann.contraction_metric.($(syms[i+1]),centraldiff($(syms[i]),immersion(f),d),refmetric(f))./s)) for i ∈ list(1,N-1)]
    Expr(:block,:(s=abs.(t)),:(uf = fiber(unitframe(f,d,t))),
        [:($(syms[i]) = getindex.(uf,$i)) for i ∈ list(1,N)]...,vals...,
        :(TensorField(base(f),TensorOperator.(Chain.($([:(Chain.($([j ∈ (i-1,i+1) ? j==i-1 ? :(.-$(curvs[j])) : curvs[j-1] : 0 for j ∈ list(1,N)]...))) for i ∈ list(1,N)]...))))))
end # cartan(unitframe(f))
function frenet(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = Real.(abs.(t,refmetric(f)))
    T = t./s
    N = unit.(centraldiff(T,immersion(f),d),refmetric(f))
    TensorField(base(f), TensorOperator.(centraldiff(Chain.(T,N),immersion(f),d)./s))
end
function frenet(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s = Real.(abs.(t,refmetric(f)))
    T = t./s
    N = unit.(centraldiff(T,immersion(f),d),refmetric(f))
    TensorField(base(f), TensorOperator.(centraldiff(Chain.(T,N,.⋆(T.∧N,refmetric(f))),immersion(f),d)./s))
end
frenet(f::AbstractCurve) = (F = unitframe(f); F⋅cartan(F))

# curvature, torsion, etc... invariant vector
curvatures(f::PlaneCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::SpaceCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::AbstractCurve,i::Int,args...) = curvatures(f,Val(i),args...)
curvatures(f::PlaneCurve,d::AbstractVector=centraldiffpoints(f),t=centraldifffiber(f,d)) = curvatures(f,Val(1),d,t)
function curvatures(f::SpaceCurve,::Val{2},d=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    torsion(f,d,t)*Λ(V).b[7]
end
function curvatures(f::SpaceCurve,d::AbstractVector=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    s = abs.(t,refmetric(f))
    n = centraldiff(t./s,immersion(f),d)./s
    b = t.∧n
    TensorField(base(f), Chain{V,2}.(value.(abs.(n,refmetric(f))),0,getindex.((b.∧centraldiff(n,immersion(f),d))./abs.(.⋆(b,refmetric(f)),refmetric(f)).^2,1)))
end
@generated function curvatures(f::AbstractCurve,d::AbstractVector=centraldiffpoints(f),t=centraldifffiber(f,d))
    V = Manifold(fibertype(f))
    N = mdims(V)
    bas = Grassmann.bladeindex.(N,UInt.(Λ(V).b[list(2,N)].∧Λ(V).b[list(3,N+1)]))
    syms = Symbol.(:e,list(1,N))
    vals = [:(Real.(Grassmann.contraction_metric.($(syms[i+1]),centraldiff($(syms[i]),immersion(f),d),refmetric(f))./s)) for i ∈ list(1,N-1)]
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
         [:($(syms[i]) = centraldiff($(syms[i-1])./s,immersion(f),d)) for i ∈ list(2,j<N-1 ? j+1 : N-1)]...,
        j==N-1 ? :($(Symbol(:e,N)) = .⋆(.∧($(syms...)),refmetric(f))) : nothing,
        :(TensorField(base(f),Real.(Grassmann.contraction_metric.(unit.($(Symbol(:e,j+1)),refmetric(f)),centraldiff(unit.($(syms[j]),refmetric(f)),d),refmetric(f))./abs.(e1)).*$(Λ(V).b[j+1]∧Λ(V).b[j+2]))))
end
darboux(f::AbstractCurve) = compound(unitframe(f),Val(2))⋅curvatures(f)
darboux(f::AbstractCurve,j) = compound(unitframe(f),Val(2))⋅curvatures(f,j)

function bishopframe(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s,b = Real.(abs.(t,refmetric(f)))
    T = t./s
    N = centraldiff(T,immersion(f),d)./s
    B = .⋆(T.∧N,refmetric(f))
    τs = fiber(torsion(f,d,t)).*s
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    cθ,sθ = cos.(θ),sin.(θ)
    TensorField(base(f), TensorOperator.(Chain.(T,cθ.*N.-sθ.*B,sθ.*N+cθ.*B)))
end
function bishopunitframe(f::SpaceCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    s,b = Real.(abs.(t,refmetric(f)))
    T = unit.(t)
    N = unit.(centraldiff(t./s,immersion(f),d))
    B = .⋆(T.∧N,refmetric(f))
    τs = fiber(torsion(f,d,t)).*s
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    cθ,sθ = cos.(θ),sin.(θ)
    TensorField(base(f), TensorOperator.(Chain.(T,cθ.*N.-sθ.*B,sθ.*N+cθ.*B)))
end
function bishoppolar(f::AbstractCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,immersion(f),d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,immersion(f),d),refmetric(f))./s)
    τs = Real.(((b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆(b,refmetric(f)),refmetric(f))).*s)
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    TensorField(base(f), Chain.(κ,θ))
end
function bishop(f::AbstractCurve,θ0=0.0,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,immersion(f),d)
    s,b = abs.(t,refmetric(f)),t.∧n
    κ = Real.(abs.(centraldiff(t./s,immersion(f),d),refmetric(f))./s)
    τs = Real.(((b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆(b,refmetric(f)),refmetric(f))).*s)
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1]).+θ0
    pushfirst!(θ,θ0)
    TensorField(base(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))
end
#bishoppolar(f::TensorField) = TensorField(base(f), Chain.(value.(fiber(curvature(f))),getindex.(fiber(cumtrapz(torsion(f))),1)))
#bishop(f::TensorField,κ=value.(fiber(curvature(f))),θ=getindex.(fiber(cumtrapz(torsion(f))),1)) = TensorField(base(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))
function normalangle(f::AbstractCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    n = centraldiff(t,immersion(f),d)
    s,b = abs.(t,refmetric(f)),t.∧n
    τs = Real.(((b.∧centraldiff(n,immersion(f),d))./abs2.(.⋆(b,refmetric(f)),refmetric(f))).*s)
    θ = (diff(points(f))/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,0)
    TensorField(base(f), θ)
end

tangentangle(f::AbstractCurve) = integral(curvature(f))
totalcurvature(f::AbstractCurve) = integrate(curvature(f))
winding(f::AbstractCurve) = totalcurvature(f)/(2π)
export tangentangle, totalcurvature, winding
function planecurve(κ::RealFunction,φ::Real=0.0)
    int = iszero(φ) ? integral(κ) : integral(κ)+φ
    integral(Chain.(cos(int),sin(int)))
end

scrollsurface(f::AbstractCurve,g::AbstractCurve,n::Int=61) = linedsurface(f,g-f,n)
scrollsurface(f::AbstractCurve,g::AbstractCurve,t) = ruledsurface(f,g-f,t)

linedsurface(f::AbstractCurve,g::AbstractCurve=tangent(f),n::Int=61) = ruledsurface(f,g,OpenParameter(n))
ruledsurface(f::AbstractCurve,g::AbstractCurve,n::Int=61) = ruledsurface(f,g,TensorField(LinRange(-1,1,n)))
function ruledsurface(f::AbstractCurve,g::AbstractCurve,t)
    TensorField(base(f)×base(t),[fiber(f)[i]+fiber(t)[j]*fiber(g)[i] for i ∈ OneTo(length(f)), j ∈ OneTo(length(t))])
end

revolve22(f::Chain,g::Chain) = Chain(f[1]*g[1],f[1]*g[2],f[2])
revolve32(f::Chain,g::Chain) = Chain(f[1]*g[1],f[2]*g[2],f[3])
revolve23(f::Chain,g::Chain) = Chain(f[1]*g[1],f[1]*g[2],f[2]+g[3])
revolve33(f::Chain,g::Chain) = Chain(f[1]*g[1],f[2]*g[2],f[3]+g[3])
for (fun,prod) ∈ ((:revolve,:fiberproduct),(:revolvesphere,:fibersphere),(:revolvesector,:fibersector))
    @eval begin
        $fun(f::TensorField,n::Int=61) = $fun(f,unitcircle(n))
        $fun(f::RealFunction,n::Int=61) = $fun(f,unitcircle(n))
        $fun(f::TensorField,n::AbstractRange) = $fun(f,unitcircle(n))
        $fun(f::TensorField{B,<:Chain{V,G,T,2} where {V,G,T}} where B,g::PlaneCurve) = $prod(f,g,revolve22)
        $fun(f::TensorField{B,<:Chain{V,G,T,2} where {V,G,T}} where B,g::SpaceCurve) = $prod(f,g,revolve23)
        $fun(f::TensorField{B,<:Chain{V,G,T,3} where {V,G,T}} where B,g::PlaneCurve) = $prod(f,g,revolve32)
        $fun(f::TensorField{B,<:Chain{V,G,T,3} where {V,G,T}} where B,g::SpaceCurve) = $prod(f,g,revolve33)
        $fun(f::RealFunction,g...) = $fun(graph(f),g...)
        $fun(f,g::RealFunction) = $fun(f,unithelix(g))
        #$fun(f::RealFunction,g::RealFunction) = $fun(graph(f),unithelix(g)) # ambiguous
    end
end

linkmap(f::Chain,g::Chain) = g-f
linkmap(f::SpaceCurve,g::SpaceCurve) = fiberproduct(f,g,linkmap)
function link(tf::Chain,tg::Chain,f::Chain,g::Chain)
    gf = g-f; ngf = norm(gf)
    ∧(gf,tf,tg)/(ngf*ngf*ngf)
end
function link(f::SpaceCurve,g::SpaceCurve)
    tf,tg = fiber(tangent(f)),fiber(tangent(g))
    TensorField(base(f)×base(g),[link(tf[i],tg[j],fiber(f)[i],fiber(g)[j]) for i ∈ OneTo(length(f)), j ∈ OneTo(length(g))])
end
linkintegral(f::SpaceCurve,g::SpaceCurve) = integral(link(f,g))/4π
linknumber(f::SpaceCurve,g::SpaceCurve) = integrate(link(f,g))/4π

sectorize23(f::Chain,g::Chain) = Chain(f[1]*g[1],f[1]*g[2],f[2]*g[3])
sectorize33(f::Chain,g::Chain) = Chain(f[1]*g[1],f[2]*g[2],f[3]*g[3])
sectorize(f::TensorField) = sectorize(10,f)
sectorize(f::Int,g::TensorField) = sectorize(LinRange(0,1,f),g)
sectorize(f::AbstractRange,g::TensorField) = sectorize(TensorField(f),g)
sectorize(f::RealFunction,g::IntervalMap) = fibersector(f,g,*)
sectorize(f::RealFunction,g::TensorField) = fibersector(f,g,*)
sectorize(f::PlaneCurve,g::TensorField) = fibersector(f,g,sectorize23)
sectorize(f::SpaceCurve,g::TensorField) = fibersector(f,g,sectorize33)
cylinderize(f::TensorField,n::Int=30) = cylinderize(f,LinRange(0,1,n))
cylinderize(f::TensorField,t::AbstractRange) = cylinderize(f,TensorField(t))
cylinderize(f::TensorField,t::RealFunction) = sectorize(Chain.(t,0t+1),f)

unitcircle(n::Int=61) = unitcircle(SphereParameter(n))
unitcircle(t::AbstractRange) = unitcircle(TensorField(t))
unitcircle(t::RealFunction) = Chain.(cos.(t),sin.(t))

_helix(f,g) = Chain(g[1],g[2],f[1])
unithelix(n::Int=61) = unithelix(LinRange(0,2π,n))
unithelix(t::AbstractRange) = unithelix(TensorField(t))
unithelix(f::RealFunction,g::PlaneCurve=unitcircle(TensorField(base(f)))) = TensorField(base(f),_helix.(f,g))

unitsphere(n::Int=31,m=61) = unitsphere(LinRange(-π/2,π/2,n),m)
unitsphere(n,m) = revolvesphere(unitcircle(n),unitcircle(m))
unitdisk(n=61,r=20) = sectorize(r,unitcircle(n))
unitball(n=31,m=61,r=10) = sectorize(r,unitsphere(n,m))
riemannsphere(n::Int=31,m=61) = unitsphere(n,m)/2+Chain(0,0,1/2)
riemannline(x,n=11) = resample(TensorField(0:2,[Chain(x[1],x[2],0.0),Chain(x[1],x[2],x[1]^2+x[2]^2)/(x[1]^2+x[2]^2+1),Chain(0,0,1.0)]),n)

unitpipe(n::Int=20,m=61) = unitpipe(LinRange(-1,1,n),m)
unitpipe(t::AbstractRange,m=61) = unitpipe(TensorField(t),m)
unitpipe(t::RealFunction,m=61) = revolve(Chain.(0t+1,t),m)
unitcylinder(n=20,m=61,r=20) = cylinderize(unitpipe(n,m),r)

unitconic(n::Int=20,m=61) = unitconic(LinRange(0,1,n),m)
unitconic(t::AbstractRange,m=61) = revolvesector(TensorField(t),m)
unitcone(n::Int=20,m=61,r=20) = unitcone(LinRange(0,1,n),m,r)
unitcone(t::AbstractRange,m=61,r=20) = cylinderize(revolve(TensorField(t),m),r)
# cylinderize(unitconic())

conoid(f::SpaceCurve,n::Int=61) = revolve(TensorField(LinRange(-1,1,n),0),f)
conoid(f::RealFunction,n::Int) = conoid(unithelix(f),n)
conoid(f::RealFunction,g::PlaneCurve=unitcircle(TensorField(base(f))),n::Int=61) = conoid(unithelix(f,g),n)
rightconoid(f::SpaceCurve,n::Int=61) = revolve(TensorField(OpenParameter(n),0),f)
rightconoid(f::RealFunction,n::Int) = rightconoid(unithelix(f),n)
rightconoid(f::RealFunction,g::PlaneCurve=unitcircle(TensorField(base(f))),n::Int=61) = rightconoid(unithelix(f,g),n)

#???
function compare(f::TensorField)#???
    d = centraldiffpoints(f)
    t = centraldifffiber(f,d)
    n = centraldiff(t,immersion(f),d)
    s = abs.(t)
    TensorField(fiber(f), centraldiff(t./s,immersion(f),d).-n./s)
end #????

function frame(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(fiber(f),immersion(f),d),n=centraldiff(t,immersion(f),d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    TensorField(base(f), TensorOperator.(Chain.(fiber(Ψu),fiber(⋆(ξ3∧Ψu)),fiber(ξ3))))
end
function unitframe(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(fiber(f),immersion(f),d),n=centraldiff(t,immersion(f),d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    ξ2 = ⋆(ξ3∧Ψu)
    TensorField(base(f), TensorOperator.(Chain.(fiber(Ψu/abs(Ψu)),fiber(ξ2/abs(ξ2)),fiber(ξ3/abs(ξ3)))))
end

export surfacemetric, surfacemetricdiag, surfaceframe
export intrinsicmetric, intrinsicmetricdiag, intrinsicframe, shape
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

function EFG(V,dfdx,dfdy,dfdz)
    xx,xy,yy,xz,yz,zz = fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdx⋅dfdy)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdx⋅dfdz)),fiber(Real(dfdy⋅dfdz)),fiber(Real(dfdz⋅dfdz))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz),Chain{V}.(xy,yy,yz),Chain{V}.(xz,yz,zz)))
end
function EG(V,dfdx,dfdy,dfdz)
    DiagonalOperator.(Chain{V}.(fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdz⋅dfdz))))
end
function LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2)
    xx,xy,yy,xz,yz,zz = fiber(Real(ddfdx2⋅n)),fiber(Real(ddfdxdy⋅n)),fiber(Real(ddfdy2⋅n)),fiber(Real(ddfdxdz⋅n)),fiber(Real(ddfdydz⋅n)),fiber(Real(ddfdz2⋅n))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz),Chain{V}.(xy,yy,yz),Chain{V}.(xz,yz,zz)))
end

function EFG(V,dfdx,dfdy,dfdz,dfdw)
    xx,xy,yy,xz,yz,zz,xw,yw,zw,ww = fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdx⋅dfdy)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdx⋅dfdz)),fiber(Real(dfdy⋅dfdz)),fiber(Real(dfdz⋅dfdz)),fiber(Real(dfdx⋅dfdw)),fiber(Real(dfdy⋅dfdw)),fiber(Real(dfdz⋅dfdw)),fiber(Real(dfdw⋅dfdw))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz,xw),Chain{V}.(xy,yy,yz,yw),Chain{V}.(xz,yz,zz,zw),Chain{V}.(xw,yw,zw,ww)))
end
function EG(V,dfdx,dfdy,dfdz,dfdw)
    DiagonalOperator.(Chain{V}.(fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdz⋅dfdz)),fiber(Real(dfdw⋅dfdw))))
end
function LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2)
    xx,xy,yy,xz,yz,zz,xw,yw,zw,ww = fiber(Real(ddfdx2⋅n)),fiber(Real(ddfdxdy⋅n)),fiber(Real(ddfdy2⋅n)),fiber(Real(ddfdxdz⋅n)),fiber(Real(ddfdydz⋅n)),fiber(Real(ddfdz2⋅n)),fiber(Real(ddfdxdw⋅n)),fiber(Real(ddfdydw⋅n)),fiber(Real(ddfdzdw⋅n)),fiber(Real(ddfdw2⋅n))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz,xw),Chain{V}.(xy,yy,yz,yw),Chain{V}.(xz,yz,zz,zw),Chain{V}.(xw,yw,zw,ww)))
end

function EFG(V,dfdx,dfdy,dfdz,dfdw,dfdv)
    xx,xy,yy,xz,yz,zz,xw,yw,zw,ww,xv,yv,zv,wv,vv = fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdx⋅dfdy)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdx⋅dfdz)),fiber(Real(dfdy⋅dfdz)),fiber(Real(dfdz⋅dfdz)),fiber(Real(dfdx⋅dfdw)),fiber(Real(dfdy⋅dfdw)),fiber(Real(dfdz⋅dfdw)),fiber(Real(dfdw⋅dfdw)),fiber(Real(dfdx⋅dfdv)),fiber(Real(dfdy⋅dfdv)),fiber(Real(dfdz⋅dfdv)),fiber(Real(dfdw⋅dfdv)),fiber(Real(dfdv⋅dfdv))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz,xw,xv),Chain{V}.(xy,yy,yz,yw,yv),Chain{V}.(xz,yz,zz,zw,zv),Chain{V}.(xw,yw,zw,ww,wv),Chain{V}.(xv,yv,zv,wv,vv)))
end
function EG(V,dfdx,dfdy,dfdz,dfdw,dfdv)
    DiagonalOperator.(Chain{V}.(fiber(Real(dfdx⋅dfdx)),fiber(Real(dfdy⋅dfdy)),fiber(Real(dfdz⋅dfdz)),fiber(Real(dfdw⋅dfdw)),fiber(Real(dfdv⋅dfdv))))
end
function LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2,ddfdxdv,ddfdydv,ddfdzdv,ddfdwdv,ddfdv2)
    xx,xy,yy,xz,yz,zz,xw,yw,zw,ww,xv,yv,zv,wv,vv = fiber(Real(ddfdx2⋅n)),fiber(Real(ddfdxdy⋅n)),fiber(Real(ddfdy2⋅n)),fiber(Real(ddfdxdz⋅n)),fiber(Real(ddfdydz⋅n)),fiber(Real(ddfdz2⋅n)),fiber(Real(ddfdxdw⋅n)),fiber(Real(ddfdydw⋅n)),fiber(Real(ddfdzdw⋅n)),fiber(Real(ddfdw2⋅n)),fiber(Real(ddfdxdv⋅n)),fiber(Real(ddfdydv⋅n)),fiber(Real(ddfdzdv⋅n)),fiber(Real(ddfdwdv⋅n)),fiber(Real(ddfdv2⋅n))
    TensorOperator.(Chain{V}.(Chain{V}.(xx,xy,xz,xw,xv),Chain{V}.(xy,yy,yz,yw,yv),Chain{V}.(xz,yz,zz,zw,zv),Chain{V}.(xw,yw,zw,ww,wv),Chain{V}.(xv,yv,zv,wv,vv)))
end

firstform(dom::ScalarField,f::Function) = firstform(TensorField(dom,f))
function firstform(t::ScalarField,g=gradient(t),V=Submanifold(ndims(t)))
    if ndims(t) == 1
        return g
    elseif ndims(t) == 2
        dfdx,dfdy = getindex.(g,1),getindex.(g,2)
        E,F,G = fiber(dfdx*dfdx),fiber(dfdx*dfdy),fiber(dfdy*dfdy)
        TensorField(base(t),TensorOperator.(Chain{V}.(Chain{V}.(E.+1,F),Chain{V}.(F,G.+1))))
    elseif ndims(t) == 3
        dfdx,dfdy,dfdz = getindex.(g,1),getindex.(g,2),getindex.(g,3)
        xx,xy,yy,xz,yz,zz = fiber(dfdx*dfdx),fiber(dfdx*dfdy),fiber(dfdy*dfdy),fiber(dfdx*dfdz),fiber(dfdy*dfdz),fiber(dfdz*dfdz)
        TensorField(base(t),TensorOperator.(Chain{V}.(Chain{V}.(xx.+1,xy,xz),Chain{V}.(xy,yy.+1,yz),Chain{V}.(xz,yz,zz.+1))))
    elseif ndims(t) == 4
        dfdx,dfdy,dfdz,dfdw = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)
        xx,xy,yy,xz,yz,zz,xw,yw,zw,ww = fiber(dfdx*dfdx),fiber(dfdx*dfdy),fiber(dfdy*dfdy),fiber(dfdx*dfdz),fiber(dfdy*dfdz),fiber(dfdz*dfdz),fiber(dfdx*dfdw),fiber(dfdy*dfdw),fiber(dfdz*dfdw),fiber(dfdw*dfdw)
        TensorField(base(t),TensorOperator.(Chain{V}.(Chain{V}.(xx.+1,xy,xz,xw),Chain{V}.(xy,yy.+1,yz,yw),Chain{V}.(xz,yz,zz.+1,zw),Chain{V}.(xw,yw,zw,ww.+1))))
    elseif ndims(t) == 5
        dfdx,dfdy,dfdz,dfdw,dfdv = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)
        xx,xy,yy,xz,yz,zz,xw,yw,zw,ww,xv,yv,zv,wv,vv = fiber(dfdx*dfdx),fiber(dfdx*dfdy),fiber(dfdy*dfdy),fiber(dfdx*dfdz),fiber(dfdy*dfdz),fiber(dfdz*dfdz),fiber(dfdx*dfdw),fiber(dfdy*dfdw),fiber(dfdz*dfdw),fiber(dfdw*dfdw),fiber(dfdx*dfdv),fiber(dfdy*dfdv),fiber(dfdz*dfdv),fiber(dfdw*dfdv),fiber(dfdv*dfdv)
        TensorField(base(t),TensorOperator.(Chain{V}.(Chain{V}.(xx.+1,xy,xz,xw,xv),Chain{V}.(xy,yy.+1,yz,yw,yv),Chain{V}.(xz,yz,zz.+1,zw,zv),Chain{V}.(xw,yw,zw,ww.+1,wv),Chain{V}.(xv,yv,zv,wv,vv.+1))))
    end
end

firstformdiag(dom::ScalarField,f::Function) = firstformdiag(TensorField(dom,f))
function firstformdiag(t::ScalarField,g=gradient(t),V=Submanifold(ndims(t)))
    if ndims(t) == 1
        return g
    elseif ndims(t) == 2
        dfdx,dfdy = getindex.(g,1),getindex.(g,2)
        E1,G1 = fiber(1+dfdx*dfdx),fiber(1+dfdy*dfdy)
        TensorField(base(t),DiagonalOperator.(Chain{V}.(E1,G1)))
    elseif ndims(t) == 3
        dfdx,dfdy,dfdz = getindex.(g,1),getindex.(g,2),getindex.(g,3)
        xx1,yy1,zz1 = fiber(1+dfdx*dfdx),fiber(1+dfdy*dfdy),fiber(1+dfdz*dfdz)
        TensorField(base(t),DiagonalOperator.(Chain{V}.(xx1,yy1,zz1)))
    elseif ndims(t) == 4
        dfdx,dfdy,dfdz,dfdw = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)
        xx1,yy1,zz1,ww1 = fiber(1+dfdx*dfdx),fiber(1+dfdy*dfdy),fiber(1+dfdz*dfdz),fiber(1+dfdw*dfdw)
        TensorField(base(t),DiagonalOperator.(Chain{V}.(xx1,yy1,zz1,ww1)))
    elseif ndims(t) == 5
        dfdx,dfdy,dfdz,dfdw,dfdv = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)
        xx1,yy1,zz1,ww1,vv1 = fiber(1+dfdx*dfdx),fiber(1+dfdy*dfdy),fiber(1+dfdz*dfdz),fiber(1+dfdw*dfdw),fiber(1+dfdv*dfdv)
        TensorField(base(t),DiagonalOperator.(Chain{V}.(xx1,yy1,zz1,ww1,vv1)))
    end
end

firstform(dom,f::Function) = firstform(TensorField(dom,f))
function firstform(t,g=gradient(t),V=Submanifold(ndims(t)))
    if ndims(t) == 1
        return Real.(abs.(g,refmetric(t)))
    elseif ndims(t) == 2
        TensorField(base(t),EFG(V,getindex.(g,1),getindex.(g,2)))
    elseif ndims(t) == 3
        TensorField(base(t),EFG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3)))
    elseif ndims(t) == 4
        TensorField(base(t),EFG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)))
    elseif ndims(t) == 5
        TensorField(base(t),EFG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)))
    end
end

firstformdiag(dom,f::Function) = firstformdiag(TensorField(dom,f))
function firstformdiag(t,g=gradien(t),V=Submanifold(ndims(t)))
    if ndims(t) == 1
        return Real.(abs.(g,refmetric(t)))
    elseif ndims(t) == 2
        TensorField(base(t),EG(V,getindex.(g,1),getindex.(g,2)))
    elseif ndims(t) == 3
        TensorField(base(t),EG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3)))
    elseif ndims(t) == 4
        TensorField(base(t),EG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)))
    elseif ndims(t) == 5
        TensorField(base(t),EG(V,getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)))
    end
end

secondform(dom,f::Function) = secondform(TensorField(dom,f))
function secondform(t,g=gradient(t),V=Submanifold(ndims(t)))
    n = ⋆unittangent(t,∧(g))
    if ndims(t) == 2
        dfdx,dfdy = getindex.(g,1),getindex.(g,2)
        ddfdx,ddfdy2 = gradient(dfdx),gradient(dfdy,Val(2))
        ddfdx2,ddfdxdy = getindex.(ddfdx,1),getindex.(ddfdx,2)
        TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2))
    elseif ndims(t) == 3
        dfdx,dfdy,dfdz = getindex.(g,1),getindex.(g,2),getindex.(g,3)
        ddfdx,ddfdy2,ddfdydz = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3))
        ddfdx2,ddfdxdy,ddfdxdz = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3)
        ddfdz2 = gradient(dfdz,Val(3))
        TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2))
    elseif ndims(t) == 4
        dfdx,dfdy,dfdz,dfdw = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)
        ddfdx,ddfdy2,ddfdydz,ddfdydw = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3)),gradient(dfdy,Val(4))
        ddfdx2,ddfdxdy,ddfdxdz,ddfdxdw = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3),getindex.(ddfdx,4)
        ddfdz2,ddfdzdw = gradient(dfdz,Val(3)),gradient(dfdz,Val(4))
        ddfdw2 = gradient(dfdw,Val(4))
        TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2))
    elseif ndims(t) == 5
        dfdx,dfdy,dfdz,dfdw,dfdv = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)
        ddfdx,ddfdy2,ddfdydz,ddfdydw,ddfdydv = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3)),gradient(dfdy,Val(4)),gradient(dfdy,Val(5))
        ddfdx2,ddfdxdy,ddfdxdz,ddfdxdw,ddfdxdv = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3),getindex.(ddfdx,4),getindex.(ddfdx,5)
        ddfdz2,ddfdzdw,ddfdzdv = gradient(dfdz,Val(3)),gradient(dfdz,Val(4)),gradient(dfdz,Val(5))
        ddfdw2,ddfdwdv = gradient(dfdw,Val(4)),gradient(dfdv,Val(5))
        ddfdv2 = gradient(dfdv,Val(5))
        TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2,ddfdxdv,ddfdydv,ddfdzdv,ddfdwdv,ddfdv2))
    end
end

firstsecondform(dom,f::Function) = firstsecondform(TensorField(dom,f))
function firstsecondform(t,g=gradient(t),V=Submanifold(ndims(t)))
    n = ⋆unittangent(t,∧(g))
    if ndims(t) == 2
        dfdx,dfdy = getindex.(g,1),getindex.(g,2)
        ddfdx,ddfdy2 = gradient(dfdx),gradient(dfdy,Val(2))
        ddfdx2,ddfdxdy = getindex.(ddfdx,1),getindex.(ddfdx,2)
        return (TensorField(base(t),EFG(V,dfdx,dfdy)),
            TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2)))
    elseif ndims(t) == 3
        dfdx,dfdy,dfdz = getindex.(g,1),getindex.(g,2),getindex.(g,3)
        ddfdx,ddfdy2,ddfdydz = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3))
        ddfdx2,ddfdxdy,ddfdxdz = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3)
        ddfdz2 = gradient(dfdz,Val(3))
        return (TensorField(base(t),EFG(V,dfdx,dfdy,dfdz)),
            TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2)))
    elseif ndims(t) == 4
        dfdx,dfdy,dfdz,dfdw = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)
        ddfdx,ddfdy2,ddfdydz,ddfdydw = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3)),gradient(dfdy,Val(4))
        ddfdx2,ddfdxdy,ddfdxdz,ddfdxdw = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3),getindex.(ddfdx,4)
        ddfdz2,ddfdzdw = gradient(dfdz,Val(3)),gradient(dfdz,Val(4))
        ddfdw2 = gradient(dfdw,Val(4))
        return (TensorField(base(t),EFG(V,dfdx,dfdy,dfdz,dfdw)),
            TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2)))
    elseif ndims(t) == 5
        dfdx,dfdy,dfdz,dfdw,dfdv = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)
        ddfdx,ddfdy2,ddfdydz,ddfdydw,ddfdydv = gradient(dfdx),gradient(dfdy,Val(2)),gradient(dfdy,Val(3)),gradient(dfdy,Val(4)),gradient(dfdy,Val(5))
        ddfdx2,ddfdxdy,ddfdxdz,ddfdxdw,ddfdxdv = getindex.(ddfdx,1),getindex.(ddfdx,2),getindex.(ddfdx,3),getindex.(ddfdx,4),getindex.(ddfdx,5)
        ddfdz2,ddfdzdw,ddfdzdv = gradient(dfdz,Val(3)),gradient(dfdz,Val(4)),gradient(dfdz,Val(5))
        ddfdw2,ddfdwdv = gradient(dfdw,Val(4)),gradient(dfdv,Val(5))
        ddfdv2 = gradient(dfdv,Val(5))
        return (TensorField(base(t),EFG(V,dfdx,dfdy,dfdz,dfdw,dfdv)),
            TensorField(base(t),LMN(V,n,ddfdx2,ddfdxdy,ddfdy2,ddfdxdz,ddfdydz,ddfdz2,ddfdxdw,ddfdydw,ddfdzdw,ddfdw2,ddfdxdv,ddfdydv,ddfdzdv,ddfdwdv,ddfdv2)))
    end
end

thirdform(dom,f::Function) = thirdform(TensorField(dom,f))
thirdform(t,V=Submanifold(ndims(t))) = firstform(t,gradient(unitnormal(t)),V)

shape(dom,f::Function) = shape(TensorField(dom,f))
function shape(t,g=gradient(t),V=Submanifold(ndims(t)))
    EFG,LMN = firstsecondform(t,g,V)
    return inv(EFG)⋅LMN
end

intrinsicmetric(dom,f::Function) = intrinsicmetric(TensorField(dom,f))
function intrinsicmetric(t::DiagonalField)
    GridBundle(PointArray(points(t),fiber(outermorphism(t))),immersion(t))
end
function intrinsicmetric(t::EndomorphismField)
    GridBundle(PointArray(points(t),fiber(Outermorphism(t))),immersion(t))
end
function intrinsicmetric(t,g=gradient(t))
    V = if ndims(t) == 2
        Submanifold(MetricTensor([1 1; 1 1]))
    elseif ndims(t) == 3
        Submanifold(MetricTensor([1 1 1; 1 1 1; 1 1 1]))
    elseif ndims(t) == 4
        Submanifold(MetricTensor([1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1]))
    elseif ndims(t) == 5
        Submanifold(MetricTensor([1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1]))
    end
    EFG = Outermorphism.(fiber(firstform(t,g,V)))
    GridBundle(PointArray(points(t),EFG),immersion(t))
end

intrinsicmetricdiag(dom,f::Function) = intrinsicmetricdiag(TensorField(dom,f))
intrinsicmetricdiag(t::DiagonalField) = intrinsicmetric(t)
function intrinsicmetricdiag(t,g=gradient(t))
    V = if ndims(t) == 2
        Submanifold(DiagonalForm(Values(1,1)))
    elseif ndims(t) == 3
        Submanifold(DiagonalForm(Values(1,1,1)))
    elseif ndims(t) == 4
        Submanifold(DiagonalForm(Values(1,1,1,1)))
    elseif ndims(t) == 5
        Submanifold(DiagonalForm(Values(1,1,1,1,1)))
    end
    EG = outermorphism.(fiber(firstformdiag(t,g,V)))
    GridBundle(PointArray(points(t),EG),immersion(t))
end
const surfacemetric, surfacemetricdiag = intrinsicmetric, intrinsicmetricdiag

intrinsicframe(t::DiagonalField) = intrinsicframediag(t)
intrinsicframe(t::FrameBundle) = intrinsicframe(metrictensorfield(t))
function intrinsicframe(t::EndomorphismField)
    if  ndims(t) == 2
        intrinsicframe(base(t),getindex.(t,1,1),getindex.(t,1,2),getindex.(t,2,2))
    elseif ndims(t) == 3
        intrinsicframe(base(t),getindex.(t,1,1),getindex.(t,1,2),getindex.(t,2,2),getindex.(t,1,3),getindex.(t,2,3),getindex.(3,3))
    elseif ndims(t) == 4
        intrinsicframe(base(t),getindex.(t,1,1),getindex.(t,1,2),getindex.(t,2,2),getindex.(t,1,3),getindex.(t,2,3),getindex.(3,3),getindex.(t,1,4),getindex.(t,2,4),getindex.(t,3,4),getindex.(t,4,4))
    elseif ndims(t) == 5
        intrinsicframe(base(t),getindex.(t,1,1),getindex.(t,1,2),getindex.(t,2,2),getindex.(t,1,3),getindex.(t,2,3),getindex.(3,3),getindex.(t,1,4),getindex.(t,2,4),getindex.(t,3,4),getindex.(t,4,4),getindex.(t,1,5),getindex.(t,2,5),getindex.(t,3,5),getindex.(t,4,5),getindex.(t,5,5))
    end
end
function intrinsicframe(t,g=gradient(t))
    if ndims(t) == 2
        dx,dy = getindex.(g,1),getindex.(g,2)
        intrinsicframe(base(t),fiber(Real(dx⋅dx)),fiber(Real(dx⋅dy)),fiber(Real(dy⋅dy)))
    elseif ndims(t) == 3
        dx,dy,dz = getindex.(g,1),getindex.(g,2),getindex.(g,3)
        intrinsicframe(base(t),fiber(Real(dx⋅dx)),fiber(Real(dx⋅dy)),fiber(Real(dy⋅dy)),fiber(Real(dx⋅dz)),fiber(Real(dy⋅dz)),fiber(Real(dz⋅dz)))
    elseif ndims(t) == 4
        dx,dy,dz,dw = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4)
        intrinsicframe(base(t),fiber(Real(dx⋅dx)),fiber(Real(dx⋅dy)),fiber(Real(dy⋅dy)),fiber(Real(dx⋅dz)),fiber(Real(dy⋅dz)),fiber(Real(dz⋅dz)),fiber(Real(dx⋅dw)),fiber(Real(dy⋅dw)),fiber(Real(dz⋅dw)),fiber(Real(dw⋅dw)))
    elseif ndims(t) == 5
        dx,dy,dz,dw,dv = getindex.(g,1),getindex.(g,2),getindex.(g,3),getindex.(g,4),getindex.(g,5)
        intrinsicframe(base(t),fiber(Real(dx⋅dx)),fiber(Real(dx⋅dy)),fiber(Real(dy⋅dy)),fiber(Real(dx⋅dz)),fiber(Real(dy⋅dz)),fiber(Real(dz⋅dz)),fiber(Real(dx⋅dw)),fiber(Real(dy⋅dw)),fiber(Real(dz⋅dw)),fiber(Real(dw⋅dw)),fiber(Real(dx⋅dv)),fiber(Real(dy⋅dv)),fiber(Real(dz⋅dv)),fiber(Real(dw⋅dv)),fiber(Real(dv⋅dv)))
    end
end
function intrinsicframe(b,E,F,G)
    F2 = F.*F; mag,sig = sqrt.((E.*E).+F2), sign.(F2.-(E.*G))
    TensorField(b,TensorOperator.(Chain.(Chain.(E,F)./mag,Chain.(F,.-E)./(sig.*mag))))
end
#function intrinsicframe(b,xx,xy,yy,xz,yz,zz)

intrinsicframediag(t) = intrinsicframediag(firstformdiag(t))
intrinsicframediag(t::FrameBundle) = intrinsicframediag(metrictensorfield(t))
function intrinsicframediag(t::DiagonalField)
    if ndims(t) == 2
        E,G = getindex.(value.(fiber(g)),1),getindex.(value.(fiber(g)),2)
        mag,sig = sqrt.(E.*E), sign.(.-(E.*G))
        TensorField(base(t),DiagonalOperator.(Chain.(E./mag,.-E./(sig.*mag))))
    end
end
const surfaceframe, surfaceframediag = intrinsicframe, intrinsicframediag

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

