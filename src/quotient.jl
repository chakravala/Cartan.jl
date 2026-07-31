
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

import MeshTopology: QuotientTopology, invert_q, CompactTopology

OpenParameter(n::ProductTopology) = OpenParameter(n.v)
OpenParameter(n::Values{1,Int}) = OpenTopology(PointArray(0,LinRange(0,1,n[1])))
OpenParameter(n::Values{2,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2]))
OpenParameter(n::Values{3,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3]))
OpenParameter(n::Values{4,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4]))
OpenParameter(n::Values{5,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4])⊕LinRange(0,1,n[5]))
CylinderParameter(n::Values{2,Int}) = CylinderTopology(LinRange(-π,π,n[1])⊕LinRange(-1,1,n[2]))
MobiusParameter(n::Values{2,Int}) = MobiusTopology(LinRange(-π,π,n[1])⊕LinRange(-1,1,n[2]))
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
TubeParameter(n::Values{2,Int}) = TubeTopology(LinRange(-1,1,n[1])⊕LinRange(-π,π,n[2]))
TubeParameter(n::Values{3,Int}) = TubeTopology(LinRange(0,1,n[1])⊕LinRange(-1,1,n[2])⊕LinRange(-π,π,n[3]))
BallParameter(n::Values{1,Int}) = BallTopology(PointArray(0,LinRange(-1,1,n[1])))
BallParameter(n::Values{2,Int}) = BallTopology(LinRange(0,1,n[1])⊕LinRange(-π,π,n[2]))
BallParameter(n::Values{3,Int}) = BallTopology(LinRange(0,1,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π,π,n[3]))
BallParameter(n::Values{4,Int}) = BallTopology(LinRange(0,1,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π/2,π/2,n[3])⊕LinRange(-π,π,n[4]))
BallParameter(n::Values{5,Int}) = BallTopology(LinRange(0,1,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π/2,π/2,n[3])⊕LinRange(-π/2,π/2,n[4])⊕LinRange(-π,π,n[5]))
SphereParameter(n::Values{1,Int}) = SphereTopology(PointArray(0,LinRange(-π,π,n[1])))
SphereParameter(n::Values{2,Int}) = SphereTopology(LinRange(-π/2,π/2,n[1])⊕LinRange(-π,π,n[2]))
SphereParameter(n::Values{3,Int}) = SphereTopology(LinRange(-π/2,π/2,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π,π,n[3]))
SphereParameter(n::Values{4,Int}) = SphereTopology(LinRange(-π/2,π/2,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π/2,π/2,n[3])⊕LinRange(-π,π,n[4]))
SphereParameter(n::Values{5,Int}) = SphereTopology(LinRange(-π/2,π/2,n[1])⊕LinRange(-π/2,π/2,n[2])⊕LinRange(-π/2,π/2,n[3])⊕LinRange(-π/2,π/2,n[4])⊕LinRange(-π,π,n[5]))
GeographicParameter(n::Values{2,Int}) = GeographicTopology(LinRange(-π,π,n[1])⊕LinRange(-π/2,π/2,n[2]))

for fun ∈ (:Open,:Cylinder,:Mobius,:Wing,:Mirror,:Clamped,:Torus,:Hopf,:Klein,:Cone,:Tube,:Ball,:Sphere,:Geographic)
    @eval import MeshTopology: $(Symbol(fun,:Topology))
    for typ ∈ (Symbol(fun,:Parameter),)#Symbol(fun,:Topology))
        @eval begin
            export $typ
            $typ(p::ProductSpace) = $typ(PointArray(p))
            #$typ(p::Values{N,<:AbstractVector} where N) = $typ(ProductSpace(p))
            #$typ(p::T...) where T<:AbstractVector = $typ(ProductSpace(Values(p)))
            #$typ(n::NTuple) = $typ(Values(n))
        end
    end
end
for mod ∈ (:Parameter,)#:Topology)
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
                $typ() = $typ(61,61)
                $typ(n::Int...) = $typ(Values(n...))
            end
        end
    end
    for fun ∈ (:Tube,:Ball,:Sphere)
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n::Int...) = $typ(Values(n...))
            end
        end
    end
    for (fun,n,m) ∈ ((:Cylinder,61,20),(:Wing,61,20),(:Mobius,61,20),(:Klein,61,61),(:Cone,31,:(2n+1)),(:Geographic,61,:(n÷2)))
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n=$n,m=$m) = $typ(Values(n,m))
            end
        end
    end
end
#TubeTopology() = TubeTopology(20,61)
TubeParameter() = TubeParameter(20,61)
#BallTopology() = TubeTopology(20,61)
BallParameter() = TubeParameter(20,61)
#SphereTopology() = TubeTopology(31,61)
SphereParameter() = TubeParameter(31,61)

import MeshTopology: PolarTopology, RevolvedTopology
const PolarParameter = BallParameter
const RevolvedParameter = TubeParameter
export PolarTopology, PolarParameter, RevolvedTopology, RevolvedParameter

import MeshTopology: isopen, iscompact, _to_axis, zerotuple, zeroprodtop, resize, bounds
import MeshTopology: cross_sphere, cross_sector, getlocate, locate_fast, locate, location
import MeshTopology: findface, subtopology, MultilinearTopology, linearelement
import MeshTopology: linearelements, elementfuns, mycollect, mycollect2, getlinear

# Multilinear topology

export MultilinearTopology, linearelement, linearelements, elementfun, elementfuns

# BilinearTopology

export BilinearTopology, elementsplit, elementquad, elementtri

import MeshTopology: BilinearTopology, elementsplit, elementquad, elementtri, detect_tri
import MeshTopology: to_verticesinv, duplicates, duplicatemap, uniquemap

