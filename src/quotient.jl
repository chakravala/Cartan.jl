
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

# QuotientTopology

"""
    QuotientTopology{N} <: ImmersedTopology{N,N}

Generalizing upon the `ProductTopology`, the `QuotientTopology` defines a quotient identification across the boundary fluxes of the region, from which the differential topology induced compact local substructure is derived.
```Julia
isopen(t) # true if open topology
iscompact(t) # true if compact topology
```
Common instances include `OpenTopology`, `CompactTopology`, `RibbonTopology`, `MobiusTopology`, `WingTopology`, `MirrorTopology`, `ClampedTopology`, `TorusTopology`, `HopfTopology`, `KleinTopology`, `ConeTopology`, `PolarTopology`, `SphereTopology`, `GeographicTopology`.
"""
struct QuotientTopology{N,L,M,O,LA<:ImmersedTopology{L,L}} <: ImmersedTopology{N,N}
    p::Values{O,Int}
    q::Values{O,LA}
    r::Values{M,Int}
    s::Values{N,Int}
    c::Values{M,Int}
    #t::Values{O,Int}
    #QuotientTopology(p::Values{O,Int},q::Values{O,LA},r::Values{M,Int},n::Values{N,Int}) where {O,L,LA<:ImmersedTopology{L,L},M,N} = QuotientTopology{N,L,M,O,LA}(p,q,r,n)
    QuotientTopology(p::Values{O,Int},q::Values{O,LA},r::Values{M,Int},s::Values{N,Int},c::Values{M,Int}=zeros(Values{M,Int})) where {O,L,LA<:ImmersedTopology{L,L},M,N} = new{N,L,M,O,LA}(p,q,r,s,c)
end

#QuotientTopology(p::Values{O,Int},q::Values{O,LA},r::Values{M,Int},n::Values{N,Int}) where {O,L,LA<:ImmersedTopology{L,L},M,N} = QuotientTopology{N,L,M,O,LA}(p,q,r,n,invert_q(Val(O),r))

invert_q(s::Int,i::Int) = iszero(s) ? () : (i,)
invert_q(::Val{M},s::Values{M,Int}) where M = s
invert_q(::Val{0},s::Values{M,Int}) where M = Values{0,Int}()
@generated invert_q(::Val{O},s::Values{M,Int}) where {O,M} = Expr(:call,:(Values{O,Int}),[:(invert_q((@inbounds s[$i]),$i)...) for i ∈ list(1,M)]...)

const OpenTopology{N,L,M,LA} = QuotientTopology{N,L,M,0,LA}
const CompactTopology{N,L,M,LA} = QuotientTopology{N,L,M,M,LA}
QuotientTopology(n::ProductTopology) = OpenTopology(n.v)
OpenTopology(n::ProductTopology) = OpenTopology(n.v)
OpenTopology(n::QuotientTopology) = OpenTopology(size(n))
OpenTopology(n::Values{N,Int}) where N = QuotientTopology(Values{0,Int}(),Values{0,Array{Values{N-1,Int},N-1}}(),zeros(Values{2N,Int}),n)
RibbonTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1),Values(ProductTopology(n[2]),ProductTopology(n[2])),Values(1,2,0,0),n)
MobiusTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1)),Values(1,2,0,0),n)
WingTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1)),Values(1,2,0,0),n)
MirrorTopology(n::Values{1,Int}) = QuotientTopology(Values((1,)),Array{Values{0,Int},0}.(Values((undef,))),Values(1,0),n)
MirrorTopology(n::Values{2,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2]),)),Values(1,0,0,0),n)
MirrorTopology(n::Values{3,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3]),)),Values(1,0,0,0,0,0),n)
MirrorTopology(n::Values{4,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3],n[4]),)),Values(1,0,0,0,0,0,0,0),n)
MirrorTopology(n::Values{5,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3],n[4],n[5]),)),Values(1,0,0,0,0,0,0,0,0,0),n)
ClampedTopology(n::Values{1,Int}) = QuotientTopology(Values(1,2),Array{Values{0,Int},0}.(Values(undef,undef)),Values(1,2),n)
ClampedTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,3,4),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
ClampedTopology(n::Values{3,Int}) = QuotientTopology(Values(1,2,3,4,5,6),Values(ProductTopology(n[2],n[3]),ProductTopology(n[2],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
ClampedTopology(n::Values{4,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8),Values(ProductTopology(n[2],n[3],n[4]),ProductTopology(n[2],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
ClampedTopology(n::Values{5,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8,9,10),Values(ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
TorusTopology(n::Values{1,Int}) = QuotientTopology(Values(2,1),Array{Values{0,Int},0}.(Values(undef,undef)),Values(1,2),n)
TorusTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
TorusTopology(n::Values{3,Int}) = QuotientTopology(Values(2,1,4,3,6,5),Values(ProductTopology(n[2],n[3]),ProductTopology(n[2],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
TorusTopology(n::Values{4,Int}) = QuotientTopology(Values(2,1,4,3,6,5,8,7),Values(ProductTopology(n[2],n[3],n[4]),ProductTopology(n[2],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
TorusTopology(n::Values{5,Int}) = QuotientTopology(Values(2,1,4,3,6,5,8,7,10,9),Values(ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
HopfTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
HopfTopology(n::Values{3,Int}) = QuotientTopology(Values(2,1,4,3,6,5),Values(ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
KleinTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1),ProductTopology(1:1:n[1]),ProductTopology(1:1:n[1])),Values(1,2,3,4),n)
ConeTopology(n::Values{2,Int}) = QuotientTopology(Values(1,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,0,2,3),n)
PolarTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
SphereTopology(n::Values{1,Int}) = TorusTopology(n)
SphereTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n,Values(1,1,0,0))
SphereTopology(n::Values{3,Int}) = QuotientTopology(Values(1,2,3,4,6,5),Values(ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
SphereTopology(n::Values{4,Int}) = QuotientTopology(Values(1,2,3,4,5,6,8,7),Values(ProductTopology(OneTo(n[2]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[2]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[2]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[2]),CrossRange(n[4])),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
SphereTopology(n::Values{5,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8,10,9),Values(ProductTopology(OneTo(n[2]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[2]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[3]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[3]),CrossRange(n[5])),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
GeographicTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,3,4),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(CrossRange(n[1])),ProductTopology(CrossRange(n[1]))),Values(1,2,3,4),n)

OpenParameter(n::ProductTopology) = OpenParameter(n.v)
OpenParameter(n::Values{1,Int}) = OpenTopology(PointArray(0,LinRange(0,1,n[1])))
OpenParameter(n::Values{2,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2]))
OpenParameter(n::Values{3,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3]))
OpenParameter(n::Values{4,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4]))
OpenParameter(n::Values{5,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4])⊕LinRange(0,1,n[5]))
RibbonParameter(n::Values{2,Int}) = RibbonTopology(LinRange(0,2π,n[1])⊕LinRange(-1,1,n[2]))
MobiusParameter(n::Values{2,Int}) = MobiusTopology(LinRange(0,2π,n[1])⊕LinRange(-1,1,n[2]))
WingParameter(n::Values{2,Int}) = WingParameter(LinRange(0,1,n[1])⊕LinRange(-1,1,n[2]))
MirrorParameter(n::Values{1,Int}) = MirrorTopology(PointArray(0,LinRange(0,2π,n[1])))
MirrorParameter(n::Values{2,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2]))
MirrorParameter(n::Values{3,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3]))
MirrorParameter(n::Values{4,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4]))
MirrorParameter(n::Values{5,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4])⊕LinRange(0,1,n[5]))
ClampedParameter(n::Values{1,Int}) = ClampedTopology(PointArray(0,LinRange(0,2π,n[1])))
ClampedParameter(n::Values{2,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
ClampedParameter(n::Values{3,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3]))
ClampedParameter(n::Values{4,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4]))
ClampedParameter(n::Values{5,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4])⊕LinRange(0,2π,n[5]))
TorusParameter(n::Values{1,Int}) = TorusTopology(PointArray(0,LinRange(0,2π,n[1])))
TorusParameter(n::Values{2,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
TorusParameter(n::Values{3,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3]))
TorusParameter(n::Values{4,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4]))
TorusParameter(n::Values{5,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4])⊕LinRange(0,2π,n[5]))
HopfParameter(n::Values{2,Int}) = HopfTopology(LinRange(0,2π,n[2])⊕LinRange(0,4π,n[3]))
HopfParameter(n::Values{3,Int}) = HopfTopology(LinRange(7π/16/n[1],7π/16,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,4π,n[3]))
KleinParameter(n::Values{2,Int}) = KleinTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
ConeParameter(n::Values{2,Int}) = ConeTopology(LinRange(0,1,n[1])⊕LinRange(0,2π,n[2]))
PolarParameter(n::Values{2,Int}) = PolarTopology(LinRange(0,1,n[1])⊕LinRange(0,2π,n[2]))
SphereParameter(n::Values{1,Int}) = TorusParameter(n)
SphereParameter(n::Values{2,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,2π,n[2]))
SphereParameter(n::Values{3,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,2π,n[3]))
SphereParameter(n::Values{4,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,π,n[3])⊕LinRange(0,2π,n[4]))
SphereParameter(n::Values{5,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,π,n[3])⊕LinRange(0,π,n[4])⊕LinRange(0,2π,n[5]))
GeographicParameter(n::Values{2,Int}) = GeographicTopology(LinRange(-π,π,n[1])⊕LinRange(-π/2,π/2,n[2]))

for fun ∈ (:Open,:Ribbon,:Mobius,:Wing,:Mirror,:Clamped,:Torus,:Hopf,:Klein,:Cone,:Polar,:Sphere,:Geographic)
    for typ ∈ (Symbol(fun,:Topology),Symbol(fun,:Parameter))
        @eval begin
            export $typ
            $typ(p::ProductSpace) = $typ(PointArray(p))
            $typ(p::Values{N,<:AbstractVector} where N) = $typ(ProductSpace(p))
            $typ(p::T...) where T<:AbstractVector = $typ(ProductSpace(Values(p)))
            $typ(n::NTuple) = $typ(Values(n))
        end
    end
end
for mod ∈ (:Topology,:Parameter)
    for fun ∈ (:Hopf,)
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n::Int...) = $typ(Values(n...))
                $typ() = $typ(Values(7,60,61))
            end
        end
    end
    for fun ∈ (:Open,:Mirror,:Clamped,:Torus)
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ() = $typ(60,60)
                $typ(n::Int...) = $typ(Values(n...))
            end
        end
    end
    for (fun,n,m) ∈ ((:Ribbon,60,20),(:Wing,60,20),(:Mobius,60,20),(:Klein,60,60),(:Cone,30,:(2n+1)),(:Polar,30,:(2n)),(:Sphere,30,:(2n+1)),(:Geographic,61,:(n÷2)))
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n=$n,m=$m) = $typ(Values(n,m))
            end
        end
    end
end

isopen(t::QuotientTopology) = false
isopen(t::OpenTopology) = true
iscompact(t::QuotientTopology) = false
iscompact(t::CompactTopology) = true
_to_axis(f::Int) = (iseven(f) ? f : f+1)÷2

zeroprodtop(r,n) = iszero(r) ? () : (ProductTopology(n),)
LinearAlgebra.cross(m::OpenTopology,n::OpenTopology) = OpenTopology(m.s...,n.s...)
LinearAlgebra.cross(m::OpenTopology{1},n::OpenTopology{1}) = OpenTopology(m.s...,n.s...)
function LinearAlgebra.cross(m::QuotientTopology{1},n::QuotientTopology{1})
    M,N = m.s[1],n.s[1]
    QuotientTopology(Values(m.p...,(n.p.+2)...),
        Values((zeroprodtop(m.r[1],N)...,zeroprodtop(m.r[2],N)...,zeroprodtop(n.r[1],M)...,zeroprodtop(n.r[2],M)...)),
        Values(m.r...,iszero(n.r[1]) ? 0 : n.r[1]+length(m.p),iszero(n.r[2]) ? 0 : n.r[2]+length(m.p)),Values(M,N))
end
LinearAlgebra.cross(m::OpenTopology,n::Int) = OpenTopology(m.s...,n)
LinearAlgebra.cross(m::OpenTopology{1},n::Int) = OpenTopology(m.s...,n)
function LinearAlgebra.cross(m::QuotientTopology{1},n::Int)
    QuotientTopology(m.p,
        Values((zeroprodtop(m.r[1],n)...,zeroprodtop(m.r[2],n)...)),
        Values(m.r...,0,0),Values(m.s...,n))
end
function LinearAlgebra.cross(m::QuotientTopology,n::Int)
    QuotientTopology(m.p,m.q .× n,Values(m.r...,0,0),Values(m.s...,n))
end

getlocate(i) = Values((i,))
getlocate(a,i) = Values((i,))
getlocate(a,i,j) = isone(a) ? Values(i,j) : Values(j,i)
getlocate(a,i,j,k) = isone(a) ? Values(i,j,k) : (a==2) ? Values(j,i,k) : Values(j,k,i)
getlocate(a,i,j,k,l) = isone(a) ? Values(i,j,k,l) : (a==2) ? Values(j,i,k,l) : (a==3) ? Values(j,k,i,l) : Values(j,k,l,i)
getlocate(a,i,j,k,l,o) = isone(a) ? Values(i,j,k,l,o) : (a==2) ? Values(j,i,k,l,o) : (a==3) ? Values(j,k,i,l,o) : (a==4) ? Values(j,k,l,i,o) : Values(j,k,l,o,i)

function locate_fast(pr1::Int,s,i::Int)
    if isodd(pr1)
        return abs(i-1)+1
    else
        return @inbounds s[_to_axis(pr1)]-abs(i-1)
    end
end
function locate_fast(pr2::Int,n::Int,s,i::Int)
    if iseven(pr2)
        return @inbounds s[_to_axis(pr2)]+n-i
    else
        return (i+1)-n
    end
end
function locate(pr1::Int,a::Int,s,i::Int)
    if isodd(pr1)
        return abs(i-1)+1
    else
        return @inbounds s[a]-abs(i-1)
    end
end
function locate(pr2::Int,a::Int,n::Int,s,i::Int)
    if iseven(pr2)
        return @inbounds s[a]+n-i
    else
        return (i+1)-n
    end
end

function location(p,q,r::Int,s::Tuple,i::Int,jk::Int...)
    pr = @inbounds p[r]
    a = _to_axis(pr)
    getlocate(a,locate(pr,a,s,i),(@inbounds q[r])[jk...]...)
end
function location(p,q,r::Int,n::Int,s::Tuple,i::Int,jk::Int...)
    pr = @inbounds p[r]
    a = _to_axis(pr)
    getlocate(a,locate(pr,a,n,s,i),(@inbounds q[r])[jk...]...)
end

@generated function resize(m::OpenTopology{N},i) where N
    Expr(:call,:QuotientTopology,:(m.p),:(m.q),:(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end
@generated function resize(m::QuotientTopology{N,L,M,O},i) where {N,L,M,O}
    Expr(:call,:QuotientTopology,:(m.p),
         Expr(:call,:Values,Expr(:tuple,[:((@inbounds m.r[$j])∉(@inbounds m.r[2N-1],@inbounds m.r[2N]) ? (@inbounds resize(m.q[$j],i)) : (@inbounds m.q[$j])) for j ∈ list(1,O)]...)),
         :(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end
@generated function resize(m::QuotientTopology{N,L,O,O},i) where {N,L,O}
    Expr(:call,:QuotientTopology,:(m.p),
         Expr(:call,:Values,[j∉(O-1,O) ? :(@inbounds resize(m.q[$j],i)) : :(@inbounds m.q[$j]) for j ∈ list(1,O)]...),
         :(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end

resample(m::QuotientTopology{N,L,M,0},i::NTuple{N}) where {N,L,M} = QuotientTopology(m.p,m.q,m.r,Values(i))
@generated function resample(m::QuotientTopology{N,L,O,O},i::NTuple{N}) where {N,L,O}
    Expr(:block,:(perms = $(reverse(Values{L}.(Values{N}(Grassmann.combo(N,L)))))),
        Expr(:call,:QuotientTopology,:(m.p),
            Expr(:call,:Values,Expr(:tuple,[:(@inbounds resample(m.q[$j],i[perms[$((j+1)÷2)]])) for j ∈ list(1,O)]...)),
            :(m.r),Expr(:call,:Values,:i)))
end
@generated function resample(m::QuotientTopology{N,L,M,O},i::NTuple{N}) where {N,L,M,O}
    Expr(:block,:(t = invert_q($(Val(O)),m.r)),
        :(perms = $(reverse(Values{L}.(Values{N}(Grassmann.combo(N,L)))))),
        Expr(:call,:QuotientTopology,:(m.p),
            Expr(:call,:Values,Expr(:tuple,[:(@inbounds resample(m.q[$j],i[perms[t[$((j+1)÷2)]]])) for j ∈ list(1,O)]...)),
            :(m.r),Expr(:call,:Values,:i)))
end

@pure Base.eltype(::Type{<:QuotientTopology{N}}) where N = Values{N,Int}
Base.size(m::QuotientTopology) = m.s.v
Base.iterate(t::QuotientTopology) = (getindex(t,1),1)
Base.iterate(t::QuotientTopology,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

function Base.getindex(m::QuotientTopology{N},i::Vararg{Int,N}) where N
    N > 5 ? Values{N,Int}(i) : getindex(m,Val(0),i...)
end
function Base.getindex(m::QuotientTopology{1},i::Int)
    s = size(m)
    n = @inbounds s[1]
    ii = if (i > 1 && i < n)
        i
    elseif i < 2
        r = @inbounds m.r[1]
        iszero(r) ? i : locate_fast((@inbounds m.p[r]),s,i)
    else
        r = @inbounds m.r[2]
        iszero(r) ? i : locate_fast((@inbounds m.p[r]),n,s,i)
    end
    return Values{1,Int}((ii,))
end

bounds(i,n,::Val{N},::Val{M}) where {N,M} = (i > (N∈(0,M) ? 1 : 0) && (N∈(0,M) ? (<) : (≤))(i,n))

Base.getindex(m::QuotientTopology{N},::Val,i::Vararg{Int,N}) where N = getindex(m,i...)
Base.getindex(m::QuotientTopology{1},::Val,i::Int) = getindex(m,i)
function Base.getindex(m::QuotientTopology{2},N::Val,i::Int,j::Int)
    s = size(m)
    n1,n2 = @inbounds (s[1],s[2])
    isi,isj = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)))
    if isj && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j))
        end
    elseif isi && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i))
        end
    end
    return Values(i,j)
end
function Base.getindex(m::QuotientTopology{3},N::Val,i::Int,j::Int,k::Int)
    s = size(m)
    n1,n2,n3 = @inbounds (s[1],s[2],s[3])
    isi,isj,isk = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)))
    if isj && isk && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k))
        end
    elseif isi && isk && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k))
        end
    elseif isi && isj && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j))
        end
    end
    return Values(i,j,k)
end
function Base.getindex(m::QuotientTopology{4},N::Val,i::Int,j::Int,k::Int,l::Int)
    s = size(m)
    n1,n2,n3,n4 = @inbounds (s[1],s[2],s[3],s[4])
    isi,isj,isk,isl = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)),bounds(l,n4,N,Val(4)))
    if isj && isk && isl && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k,l))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k,l))
        end
    elseif isi && isk && isl && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k,l))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k,l))
        end
    elseif isi && isj && isl && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j,l))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j,l))
        end
    elseif isi && isj && isk && !isl
        if l < 2
            r = @inbounds m.r[7]
            !iszero(r) && (return location(m.p,m.q,r,s,l,i,j,k))
        else
            r = @inbounds m.r[8]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,l,i,j,k))
        end
    end
    return Values(i,j,k,l)
end
function Base.getindex(m::QuotientTopology{5},N::Val,i::Int,j::Int,k::Int,l::Int,o::Int)
    s = size(m)
    n1,n2,n3,n4,n5 = @inbounds (s[1],s[2],s[3],s[4],s[5])
    isi,isj,isk,isl,iso = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)),bounds(l,n4,N,Val(4)),bounds(o,n5,N,Val(5)))
    if isj && isk && isl && iso && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k,l,o))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k,l,o))
        end
    elseif isi && isk && isl && iso && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k,l,o))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k,l,o))
        end
    elseif isi && isj && isl && iso && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j,l,o))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j,l,o))
        end
    elseif isi && isj && isk && iso && !isl
        if l < 2
            r = @inbounds m.r[7]
            !iszero(r) && (return location(m.p,m.q,r,s,l,i,j,k,o))
        else
            r = @inbounds m.r[8]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,l,i,j,k,o))
        end
    elseif isi && isj && isk && isl && !iso
        if o < 2
            r = @inbounds m.r[9]
            !iszero(r) && (return location(m.p,m.q,r,s,o,i,j,k,l))
        else
            r = @inbounds m.r[10]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,o,i,j,k,l))
        end
    end
    return Values(i,j,k,l,o)
end

function findface(m,r,i,vals)
    ri = r[i]
    if iszero(ri) || m.p[ri] ∉ r || m.q[ri][vals...] ≠ vals
        0
    else
        m.p[ri]≠(@inbounds r[1]) ? 2 : 1
    end
end
function findface(m,r,i,vals,ex...)
    ri = r[i]
    if iszero(ri) || m.p[ri] ∉ r || exclude(m.q[ri],ex...)[vals...] ≠ vals
        0
    else
        pri = m.p[ri]
        findfirst(x->x==pri,r)
    end
end

subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N = OpenTopology(size(m)[N+1])
subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M} = OpenTopology(size(m)[N+1],size(m)[N+M+2])
subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L} = OpenTopology(size(m)[N+1],size(m)[N+M+2],size(m)[N+M+L+3])
subtopology(m::QuotientTopology{1},i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N = m
subtopology(m::QuotientTopology{2},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M} = m
subtopology(m::QuotientTopology{3},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L} = m
subtopology(m::QuotientTopology{4},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,o::NTuple{Y,Int} where Y) where {N,M,L,O} = m
subtopology(m::QuotientTopology{5},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,o::NTuple{Y,Int},::Colon,z::NTuple{Z,Int} where Z) where {N,M,L,O,Y} = m
function subtopology(m::QuotientTopology{M},::Val{N}) where {M,N}
    r1,r2 = m.r[2N-1],m.r[2N]
    r1z,r2z,n = iszero(r1),iszero(r2),size(m)[N]
    if r1z
        if r2z
            OpenTopology(n)
        else
            QuotientTopology(Values((isodd(m.p[r2]) ? 1 : 2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(0,1),Values((n,)))
        end
    elseif r2z && !r1z
        QuotientTopology(Values((isodd(m.p[r1]) ? 1 : 2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(1,0),Values((n,)))
    else
        QuotientTopology(Values(isodd(m.p[r1]) ? 1 : 2,isodd(m.p[r2]) ? 1 : 2),Array{Values{0,Int},0}.(Values((undef,undef))),Values(1,2),Values((n,)))
    end
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N
    N1 = N+1
    r,vals = m.r[Values(2N1-1,2N1)],Values(i...,j...)
    p1,p2 = findface(m,r,1,vals),findface(m,r,2,vals)
    p1z,p2z,n = iszero(p1),iszero(p2),size(m)[N1]
    if p1z
        if p2z
            OpenTopology(n)
        else
            QuotientTopology(Values((2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(0,1),Values((n,)))
        end
    elseif p2z && !p1z
        MirrorTopology(n)
    else
        QuotientTopology(Values(p1,p2),Array{Values{0,Int},0}.(Values((undef,undef))),Values(1,2),Values((n,)))
    end
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M}
    N1,M1,s = N+1,N+M+2,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1)],Values(i...,j...,k...)
    (p1,p2,p3,p4) = (
        findface(m,r,1,vals,Val(M1-1)),findface(m,r,2,vals,Val(M1-1)),
        findface(m,r,3,vals,Val(N1)),findface(m,r,4,vals,Val(N1)))
    p1z,p2z,p3z,p4z = iszero.((p1,p2,p3,p4))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : Values{4}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...)
    b = if iszero(pz)
        Values{0,Array{Values{1,Int},1}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[M1-1]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[M1-1]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[N1]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[N1]),))...)
    end
    c = Values{4,Int}(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1]))
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L}
    N1,M1,L1,s = N+1,N+M+2,N+M+L+3,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1,2L1-1,2L1)],Values(i...,j...,k...,l...)
    (p1,p2,p3,p4,p5,p6) = (
        findface(m,r,1,vals,Val(M1-1),Val(L1-1)),findface(m,r,2,vals,Val(M1-1),Val(L1-1)),
        findface(m,r,3,vals,Val(N1),Val(L1-1)),findface(m,r,4,vals,Val(N1),Val(L1-1)),
        findface(m,r,5,vals,Val(N1),Val(M1)),findface(m,r,6,vals,Val(N1),Val(M1)))
    p1z,p2z,p3z,p4z,p5z,p6z = iszero.((p1,p2,p3,p4,p5,p6))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : pz==4 ? Values{4} : pz==5 ? Values{5} : Values{6}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...,(p5z ? () : (p5,))...,(p6z ? () : (p6,))...)
    b = if iszero(pz)
        Values{0,Array{Values{2,Int},2}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[Values(M1-1,L1-1)]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[Values(M1-1,L1-1)]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[Values(N1,L1-1)]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[Values(N1,L1-1)]),))...,
        (p5z ? () : (ProductTopology(m.q[r[5]].v[Values(N1,M1)]),))...,
        (p6z ? () : (ProductTopology(m.q[r[6]].v[Values(N1,M1)]),))...)
    end
    c = Values(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z),p5z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),p6z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z),))...,(p5z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),))...,(p6z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1],s[L1]))
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,h::NTuple{H,Int} where H) where {N,M,L,O}
    N1,M1,L1,O1,s = N+1,N+M+2,N+M+L+3,N+M+L+O+4,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1,2L1-1,2L1,2O1-1,2O1)],Values(i...,j...,k...,l...,h...)
    (p1,p2,p3,p4,p5,p6,p7,p8) = (
        findface(m,r,1,vals,Val(M1-1),Val(L1-1),Val(O1-1)),findface(m,r,2,vals,Val(M1-1),Val(L1-1),Val(O1-1)),
        findface(m,r,3,vals,Val(N1),Val(L1-1),Val(O1-1)),findface(m,r,4,vals,Val(N1),Val(L1-1),Val(O1-1)),
        findface(m,r,5,vals,Val(N1),Val(M1),Val(O1-1)),findface(m,r,6,vals,Val(N1),Val(M1),Val(O1-1)),
        findface(m,r,7,vals,Val(N1),Val(M1),Val(L1)),findface(m,r,8,vals,Val(N1),Val(M1),Val(L1)))
    p1z,p2z,p3z,p4z,p5z,p6z,p7z,p8z = iszero.((p1,p2,p3,p4,p5,p6,p7,p8))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z)+!(p8z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : pz==4 ? Values{4} : pz==5 ? Values{5} : pz==6 ? Values{6} : pz==7 ? Values{7} : Values{8}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...,(p5z ? () : (p5,))...,(p6z ? () : (p6,))...,(p7z ? () : (p7,))...,(p8z ? () : (p8,))...)
    b = if iszero(pz)
        Values{0,Array{Values{3,Int},3}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[Values(M1-1,L1-1,O1-1)]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[Values(M1-1,L1-1,O1-1)]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[Values(N1,L1-1,O1-1)]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[Values(N1,L1-1,O1-1)]),))...,
        (p5z ? () : (ProductTopology(m.q[r[5]].v[Values(N1,M1,O1-1)]),))...,
       (p6z ? () : (ProductTopology(m.q[r[6]].v[Values(N1,M1,O1-1)]),))...,
        (p7z ? () : (ProductTopology(m.q[r[7]].v[Values(N1,M1,L1)]),))...,
        (p8z ? () : (ProductTopology(m.q[r[8]].v[Values(N1,M1,L1)]),))...)
    end
    c = Values(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z),p5z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),p6z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z),p7z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z),p8z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z),))...,(p5z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),))...,(p6z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z),))...,(p7z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z),))...,(p8z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1],s[L1],s[O1]))
end

# 1
(m::QuotientTopology)(c::Colon,i::Int...) = subtopology(m,(),c,i)
(m::QuotientTopology)(i::Int,c::Colon,j::Int...) = subtopology(m,(i,),c,j)
(m::QuotientTopology)(i::Int,j::Int,c::Colon,k::Int...) = subtopology(m,(i,j),c,k)
(m::QuotientTopology)(i::Int,j::Int,k::Int,c::Colon,l::Int...) = subtopology(m,(i,j,k),c,l)
(m::QuotientTopology)(i::Int,j::Int,k::Int,l::Int,c::Colon,h::Int...) = subtopology(m,(i,j,k,l),c,h)

# 2 - 0
(m::QuotientTopology)(c::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,i::Int,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(i,j),c,k)
(m::QuotientTopology)(c::Colon,i::Int,j::Int,k::Int,::Colon,l::Int...) = subtopology(m,(),c,(i,j,k),c,l)
# 2 - 1
(m::QuotientTopology)(i::Int,c::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(i,),c,(j,),c,k)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,k::Int,::Colon,l::Int...) = subtopology(m,(i,),c,(j,k),c,l)
# 2 - 2
(m::QuotientTopology)(i::Int,j::Int,c::Colon,::Colon,k::Int...) = subtopology(m,(i,j),c,(),c,k)
(m::QuotientTopology)(i::Int,j::Int,c::Colon,k::Int,::Colon,l::Int...) = subtopology(m,(i,j),c,(k,),c,l)
# 2 - 3
(m::QuotientTopology)(i::Int,j::Int,k::Int,c::Colon,::Colon,l::Int...) = subtopology(m,(i,j,k),c,(),c,l)

# 3 - 0 - 0
(m::QuotientTopology)(c::Colon,::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(),c,(i,j),c,k)
# 3 - 0 - 1
(m::QuotientTopology)(c::Colon,i::Int,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,(),c,j)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(i,),c,(j,),c,k)
# 3 - 0 - 2
(m::QuotientTopology)(c::Colon,i::Int,j::Int,::Colon,::Colon,k::Int...) = subtopology(m,(),c,(i,j),c,(),c,k)
# 3 - 1
(m::QuotientTopology)(i::Int,c::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,::Colon,::Colon,k::Int...) = subtopology(m,(i,),c,(j,),c,(),c,k)
(m::QuotientTopology)(i::Int,c::Colon,::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(i,),c,(),c,(j,),c,k)
# 3 - 2
(m::QuotientTopology)(i::Int,j::Int,c::Colon,::Colon,::Colon,k::Int...) = subtopology(m,(i,j),c,(),c,(),c,k)

# 4
(m::QuotientTopology)(c::Colon,::Colon,::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,::Colon,::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(),c,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(),c,(i,),c,(),c,j)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,(),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,(),c,(),c,j)

# Multilinear topology

export MultilinearTopology, linearelement, linearelements, elementfun, elementfuns

abstract type MultilinearTopology{N} <: ImmersedTopology{N,1} end

function linearelement(l::AbstractVector,i)
    Values(l[i],l[i+1])
end
function linearelement(l::AbstractArray{T,2} where T,i,j)
    i1,j1 = i+1,j+1
    Values(l[i,j],l[i1,j],l[i1,j1],l[i,j1])
end
function linearelement(l::AbstractArray{T,3} where T,i,j,k)
    i1,j1,k1 = i+1,j+1,k+1
    Values(l[i,j,k],l[i1,j,k],l[i1,j1,k],l[i,j1,k],
           l[i,j,k1],l[i1,j,k1],l[i1,j1,k1],l[i,j1,k1])
end
function linearelement(l::AbstractArray{T,4} where T,i,j,k,w)
    i1,j1,k1,w1 = i+1,j+1,k+1,w+1
    Values(l[i,j,k,w],l[i1,j,k,w],l[i1,j1,k,w],l[i,j1,k,w],
           l[i,j,k1,w],l[i1,j,k1,w],l[i1,j1,k1,w],l[i,j1,k1,w],
           l[i,j1,k1,w1],l[i1,j1,k1,w1],l[i1,j,k1,w1],l[i,j,k1,w1],
           l[i,j1,k,w1],l[i1,j1,k,w1],l[i1,j,k,w1],l[i,j,k,w1])
end
function linearelement(l::AbstractArray{T,5} where T,i,j,k,w,v)
    i1,j1,k1,w1,v1 = i+1,j+1,k+1,w+1,v+1
    Values(l[i,j,k,w,v],l[i1,j,k,w,v],l[i1,j1,k,w,v],l[i,j1,k,w,v],
           l[i,j,k1,w,v],l[i1,j,k1,w,v],l[i1,j1,k1,w,v],l[i,j1,k1,w,v],
           l[i,j1,k1,w1,v],l[i1,j1,k1,w1,v],l[i1,j,k1,w1,v],l[i,j,k1,w1,v],
           l[i,j1,k,w1,v],l[i1,j1,k,w1,v],l[i1,j,k,w1,v],l[i,j,k,w1,v],
           l[i,j,k,w,v1],l[i1,j,k,w,v1],l[i1,j1,k,w,v1],l[i,j1,k,w,v1],
           l[i,j,k1,w,v1],l[i1,j,k1,w,v1],l[i1,j1,k1,w,v1],l[i,j1,k1,w,v1],
           l[i,j1,k1,w1,v1],l[i1,j1,k1,w1,v1],l[i1,j,k1,w1,v1],l[i,j,k1,w1,v1],
           l[i,j1,k,w1,v1],l[i1,j1,k,w1,v1],l[i1,j,k,w1,v1],l[i,j,k,w1,v1])
end

linearelements(l::AbstractVector) = linearelement.(l,length(l)-1)
linearelements(l::AbstractArray{T,2} where T,s=size(l)) = [linearelement(l,i,j) for i ∈ OneTo(s[1]-1), j ∈ OneTo(s[2]-1)]
linearelements(l::AbstractArray{T,3} where T,s=size(l)) = [linearelement(l,i,j,k) for i ∈ OneTo(s[1]-1), j ∈ OneTo(s[2]-1), k ∈ OneTo(s[3]-1)]
linearelements(l::AbstractArray{T,4} where T,s=size(l)) = [linearelement(l,i,j,k,w) for i ∈ OneTo(s[1]-1), j ∈ OneTo(s[2]-1), k ∈ OneTo(s[3]-1), w ∈ OneTo(s[3]-1)]
linearelements(l::AbstractArray{T,5} where T,s=size(l)) = [linearelement(l,i,j,k,w,v) for i ∈ OneTo(s[1]-1), j ∈ OneTo(s[2]-1), k ∈ OneTo(s[3]-1), w ∈ OneTo(s[4]-1), v ∈ OneTo(s[5]-1)]

linearelements(m::QuotientTopology) = linearelements(LinearIndices(m),m)
linearelements(l::LinearIndices,m::QuotientTopology) = linearelements(elementfuns(l,m))
linearelement(m::QuotientTopology,ij...) = linearelement(LinearIndices(m),m,ij...)
function linearelement(l::LinearIndices{1},m::QuotientTopology{1},i)
    Values(elementfun(l,m,i),elementfun(l,m,i+1))
end
function linearelement(l::LinearIndices{2},m::QuotientTopology{2},i,j)
    i1,j1 = i+1,j+1
    Values(elementfun(l,m,i,j),elementfun(l,m,i1,j),elementfun(l,m,i1,j1),elementfun(l,m,i,j1))
end
function linearelement(l::LinearIndices{3},m::QuotientTopology{3},i,j,k)
    i1,j1,k1 = i+1,j+1,k+1
    Values(elementfun(l,m,i,j,k),elementfun(l,m,i1,j,k),elementfun(l,m,i1,j1,k),elementfun(l,m,i,j1,k),
           elementfun(l,m,i,j,k1),elementfun(l,m,i1,j,k1),elementfun(l,m,i1,j1,k1),elementfun(l,m,i,j1,k1))
end
function linearelement(l::LinearIndices{4},m::QuotientTopology{4},i,j,k,w)
    i1,j1,k1,w1 = i+1,j+1,k+1,w+1
    Values(elementfun(l,m,i,j,k,w),elementfun(l,m,i1,j,k,w),elementfun(l,m,i1,j1,k,w),elementfun(l,m,i,j1,k,w),
           elementfun(l,m,i,j,k1,w),elementfun(l,m,i1,j,k1,w),elementfun(l,m,i1,j1,k1,w),elementfun(l,m,i,j1,k1,w),
           elementfun(l,m,i,j1,k1,w1),elementfun(l,m,i1,j1,k1,w1),elementfun(l,m,i1,j,k1,w1),elementfun(l,m,i,j,k1,w1),
           elementfun(l,m,i,j1,k,w1),elementfun(l,m,i1,j1,k,w1),elementfun(l,m,i1,j,k,w1),elementfun(l,m,i,j,k,w1))
end
function linearelement(l::LinearIndices{5},m::QuotientTopology{5},i,j,k,w,v)
    i1,j1,k1,w1,v1 = i+1,j+1,k+1,w+1,v+1
    Values(elementfun(l,m,i,j,k,w,v),elementfun(l,m,i1,j,k,w,v),elementfun(l,m,i1,j1,k,w,v),elementfun(l,m,i,j1,k,w,v),
           elementfun(l,m,i,j,k1,w,v),elementfun(l,m,i1,j,k1,w,v),elementfun(l,m,i1,j1,k1,w,v),elementfun(l,m,i,j1,k1,w,v),
           elementfun(l,m,i,j1,k1,w1,v),elementfun(l,m,i1,j1,k1,w1,v),elementfun(l,m,i1,j,k1,w1,v),elementfun(l,m,i,j,k1,w1,v),
           elementfun(l,m,i,j1,k,w1,v),elementfun(l,m,i1,j1,k,w1,v),elementfun(l,m,i1,j,k,w1,v),elementfun(l,m,i,j,k,w1,v),
           elementfun(l,m,i,j,k,w,v1),elementfun(l,m,i1,j,k,w,v1),elementfun(l,m,i1,j1,k,w,v1),elementfun(l,m,i,j1,k,w,v1),
           elementfun(l,m,i,j,k1,w,v1),elementfun(l,m,i1,j,k1,w,v1),elementfun(l,m,i1,j1,k1,w,v1),elementfun(l,m,i,j1,k1,w,v1),
           elementfun(l,m,i,j1,k1,w1,v1),elementfun(l,m,i1,j1,k1,w1,v1),elementfun(l,m,i1,j,k1,w1,v1),elementfun(l,m,i,j,k1,w1,v1),
           elementfun(l,m,i,j1,k,w1,v1),elementfun(l,m,i1,j1,k,w1,v1),elementfun(l,m,i1,j,k,w1,v1),elementfun(l,m,i,j,k,w1,v1))
end

elementfuns(m::QuotientTopology,s::Tuple=size(m)) = elementfuns(LinearIndices(m),m,s)
elementfuns(l::LinearIndices,m::QuotientTopology,s::Tuple=size(m)) = l
elementfuns(l::LinearIndices{1},m::OpenTopology{1},s::Tuple=size(m)) = l
elementfuns(l::LinearIndices{2},m::OpenTopology{2},s::Tuple=size(m)) = l
elementfuns(l::LinearIndices{3},m::OpenTopology{3},s::Tuple=size(m)) = l
elementfuns(l::LinearIndices{4},m::OpenTopology{4},s::Tuple=size(m)) = l
elementfuns(l::LinearIndices{5},m::OpenTopology{5},s::Tuple=size(m)) = l
function elementfuns(l::LinearIndices{1},m::QuotientTopology{1},s::Tuple=size(m))
    Int[elementfun(l,m,i) for i ∈ OneTo(s[1])]
end
function elementfuns(l::LinearIndices{2},m::QuotientTopology{2},s::Tuple=size(m))
    out = Int[elementfun(l,m,i,j) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2])]
    isone(m.c[1]) && (out[1,:] .= 1)
    isone(m.c[2]) && (out[end,:] .= out[end,1])
    isone(m.c[3]) && (out[:,1] .= 1)
    isone(m.c[4]) && (out[:,end] .= out[1,end])
    return out
end
function elementfuns(l::LinearIndices{3},m::QuotientTopology{3},s::Tuple=size(m))
    out = Int[elementfun(l,m,i,j,k) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2]), k ∈ OneTo(s[3])]
    isone(m.c[1]) && (out[1,:,:] .= 1)
    isone(m.c[2]) && (out[end,:,:] .= out[end,1,1])
    isone(m.c[3]) && (out[:,1,:] .= 1)
    isone(m.c[4]) && (out[:,end,:] .= out[1,end,1])
    isone(m.c[5]) && (out[:,:,1] .= 1)
    isone(m.c[6]) && (out[:,:,end] .= out[1,1,end])
    return out
end
function elementfuns(l::LinearIndices{4},m::QuotientTopology{4},s::Tuple=size(m))
    out = Int[elementfun(l,m,i,j,k,w) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2]), k ∈ OneTo(s[3]), w ∈ OneTo(s[4])]
    isone(m.c[1]) && (out[1,:,:,:] .= 1)
    isone(m.c[2]) && (out[end,:,:,:] .= out[end,1,1,1])
    isone(m.c[3]) && (out[:,1,:,:] .= 1)
    isone(m.c[4]) && (out[:,end,:,:] .= out[1,end,1,1])
    isone(m.c[5]) && (out[:,:,1,:] .= 1)
    isone(m.c[6]) && (out[:,:,end,:] .= out[1,1,end,1])
    isone(m.c[7]) && (out[:,:,:,1] .= 1)
    isone(m.c[8]) && (out[:,:,:,end] .= out[1,1,1,end])
    return out
end
function elementfuns(l::LinearIndices{5},m::QuotientTopology{5},s::Tuple=size(m))
    out = Int[elementfun(l,m,i,j,k,w,v) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2]), k ∈ OneTo(s[3]), w ∈ OneTo(s[4]), v ∈ OneTo(s[5])]
    isone(m.c[1]) && (out[1,:,:,:,:] .= 1)
    isone(m.c[2]) && (out[end,:,:,:,:] .= out[end,1,1,1,1])
    isone(m.c[3]) && (out[:,1,:,:,:] .= 1)
    isone(m.c[4]) && (out[:,end,:,:,:] .= out[1,end,1,1,1])
    isone(m.c[5]) && (out[:,:,1,:,:] .= 1)
    isone(m.c[6]) && (out[:,:,end,:,:] .= out[1,1,end,1,1])
    isone(m.c[7]) && (out[:,:,:,1,:] .= 1)
    isone(m.c[8]) && (out[:,:,:,end,:] .= out[1,1,1,end,1])
    isone(m.c[9]) && (out[:,:,:,:,1] .= 1)
    isone(m.c[10]) && (out[:,:,:,:,end] .= out[1,1,1,1,end])
    return out
end

elementfun(m::QuotientTopology,ij...) = elementfun(LinearIndices(m),m,ij...)
function elementfun(l::LinearIndices,m::QuotientTopology,ij...)
    out1 = getlinear(l,m,Val(0),ij...)
    out2 = getindex(l,ij...)
    out1<out2 ? out1 : out2
end

mycollect2(m::QuotientTopology{2},N=Val(0),s=size(m)) = [getindex(m,N,i,j) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2])]
mycollect(m::QuotientTopology{2},N=Val(0),s=size(m)) = [getlinear(m,N,i,j) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2])]

#elementfun.(Ref(li),mycollect(box))
#elementfun.(Ref(li),collect(box))

#bounds(i,n) = (i > 0) && (i ≤ n)

getlinear(l,m::QuotientTopology,N::Val,ij...) = getindex(l,ij...)
getlinear(l,m::QuotientTopology{1},::Val,i::Int) = getindex(m,i)
function getlinear(l,m::QuotientTopology{2},N::Val{n},i::Int,j::Int) where n
    s = size(m)
    n1,n2 = @inbounds (s[1],s[2])
    isi,isj = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)))
    if isj && !isi
        if i < 2
            #return Values(i,j)
            r = @inbounds m.r[1]
            if !iszero(r)
                p = @inbounds m.p[r]
                isodd(p) && (return getindex(l,location(m.p,m.q,r,s,i,j)...))
            end
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return getindex(l,location(m.p,m.q,r,n1,s,i,j)...))
        end
    elseif isi && !isj
        if j < 2
            #return Values(i,j)
            r = @inbounds m.r[3]
            if !iszero(r)
                p = @inbounds m.p[r]
                isodd(p) && (return getindex(l,location(m.p,m.q,r,s,j,i)...))
            end
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return getindex(l,location(m.p,m.q,r,n2,s,j,i)...))
        end
    elseif !isi && !isj && iszero(n)
        out1 = getindex(m,Val(1),i,j)
        out2 = getindex(m,Val(2),i,j)
        return getindex(l,min.(out1,out2)...)
    end
    return getindex(l,Values(i,j)...)
end

# BilinearTopology

export BilinearTopology, elementsplit, elementquad, elementtri

struct BilinearTopology{Q<:QuotientTopology{2},P,V} <: MultilinearTopology{4}
    m::Q
    q::Vector{Values{4,Int}} # quad
    t::Vector{Values{3,Int}} # tri
    i::P # vertices
    #p::Int # nodes
    v::V # verticesinv
    iq::Vector{Int} # quad id
    it::Vector{Int} # tri id
    s::Vector{Pair{Int,Int}} # elementsplit
end

MultilinearTopology(m::QuotientTopology{2}) = BilinearTopology(m)
function BilinearTopology(m::QuotientTopology{2})
    efs = elementfuns(m)
    q,t,iq,it = detect_tri(vec(linearelements(efs)))
    i = vertices(efs)
    BilinearTopology(m,q,t,i,to_verticesinv(efs),iq,it,elementsplit(iq,it))
end

QuotientTopology(t::BilinearTopology) = t.m
vertices(t::BilinearTopology) = t.i
verticesinv(t::BilinearTopology) = t.v
nodes(t::BilinearTopology) = length(verticesinv(t))
elementsplit(t::BilinearTopology) = t.s
elementquad(t::BilinearTopology) = t.iq
elementtri(t::BilinearTopology) = t.it

function detect_tri(quad::Vector{Values{4,Int}})
    tri = Vector{Values{3,Int}}()
    iq = collect(OneTo(length(quad)))
    it = Vector{Int}()
    i = 1
    while i ≤ length(quad)
        q = quad[i]
        if @inbounds q[1]==q[2]
            push!(it,iq[i])
            deleteat!(iq,i)
            deleteat!(quad,i)
            push!(tri,@inbounds Values(q[2],q[3],q[4]))
        elseif @inbounds q[2]==q[3]
            push!(it,iq[i])
            deleteat!(iq,i)
            deleteat!(quad,i)
            push!(tri,@inbounds Values(q[1],q[2],q[4]))
        elseif @inbounds q[3]==q[4]
            push!(it,iq[i])
            deleteat!(iq,i)
            deleteat!(quad,i)
            push!(tri,@inbounds Values(q[1],q[2],q[3]))
        elseif @inbounds q[4]==q[1]
            push!(it,iq[i])
            deleteat!(iq,i)
            deleteat!(quad,i)
            push!(tri,@inbounds Values(q[1],q[2],q[3]))
        else
            i += 1
        end
    end
    return quad,tri,iq,it
end
function elementsplit(iq,it)
    nq,nt = length(iq),length(it)
    out = Vector{Pair{Int,Int}}(undef,nq+nt)
    for j ∈ OneTo(nq)
        i = iq[j]
        out[i] = 4 => j
    end
    for j ∈ OneTo(nt)
        i = it[j]
        out[i] = 3 => j
    end
    return out
end

to_verticesinv(m) = unique(vec(m))
to_verticesinv(m::LinearIndices) = OneTo(length(m))
verticesinv(m::OpenTopology) = OneTo(length(m))
verticesinv(m::QuotientTopology) = unique(vec(elementfuns(m)))
function duplicates(m::QuotientTopology)
    li = LinearIndices(m)
    setdiff(vec(li),vec(elementfuns(li,m)))
end
function duplicatemap(m::QuotientTopology)
    li = LinearIndices(m)
    els = vec(elementfuns(li,m))
    out = setdiff(OneTo(length(li)),els)
    out.=>els[out]
end
function uniquemap(m::QuotientTopology)
    els = vec(elementfuns(m))
    out = unique(els)
    out.=>OneTo(length(out))
end
vertices(m::QuotientTopology) = vertices(elementfuns(m))
vertices(elm::LinearIndices) = elm
function vertices(elm::Array{Int})
    els = vec(elm)
    dup = setdiff(OneTo(length(els)),els)
    unq = unique(els)
    out = zeros(Int,size(elm))
    out[unq] .= OneTo(length(unq))
    out[dup] .= out[els[dup]]
    return out
end

