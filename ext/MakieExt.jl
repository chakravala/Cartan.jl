module MakieExt

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

using Grassmann, Cartan
isdefined(Cartan, :Requires) ? (import Cartan: Makie) : (using Makie)

import Cartan: linegraph, linegraph!
funsym(sym) = String(sym)[end] == '!' ? sym : Symbol(sym,:!)

import Cartan: polytransform, unorientedpoly, orientedpoly
import Cartan: argarrows, argarrows2, argarrows3, gridargs
import Cartan: tangentbundle, tangentbundle!, normalbundle, normalbundle!
import Cartan: planesbundle, planesbundle!, spacesbundle, spacesbundle!
import Cartan: arrowsbundle, arrowsbundle!, scaledarrows, scaledarrows!
import Cartan: scaledfield, scaledfield!, scaledbundle, scaledbundle!
import Cartan: scaledplanes, scaledplanes!, scaledspaces, scaledspaces!
import Cartan: planes, planes!, spaces, spaces!, linegraph

for fun ∈ (:linegraph,:scaledarrows,:planes,:scaledplanes,:spaces,:scaledspaces,:scaledfield,:scaledbundle,:arrowsbundle,:planesbundle,:spacesbundle,:tangentbundle,:normalbundle)
    @eval begin
        function Cartan.$fun(t::Components;args...)
            display(Makie.$fun(t[1];args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i];args...)
            end
        end
        function Cartan.$(Symbol(fun,:!))(t::Components;args...)
            display(Makie.$(Symbol(fun,:!))(t[1];args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i];args...)
            end
        end
    end
end
for fun ∈ (:wireframe,:mesh,:lines,:linesegments,:streamplot,:volume,:contour,:contourf,:contour3d,:heatmap,:voxels,:volumeslices,:surface,:scatter,:text,:arrows)
    @eval begin
        function Makie.$fun(t::Components;args...)
            display(Makie.$fun(t[1];args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i];args...)
            end
        end
        function Makie.$(Symbol(fun,:!))(t::Components;args...)
            display(Makie.$(Symbol(fun,:!))(t[1];args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i];args...)
            end
        end
    end
end
for fun ∈ (:mesh,:lines)
    @eval begin
        function Makie.$fun(t::Components,f;args...)
            display(Makie.$fun(t[1],f;args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i],f;args...)
            end
        end
        function Makie.$(Symbol(fun,:!))(t::Components,f;args...)
            display(Makie.$(Symbol(fun,:!))(t[1],f;args...))
            for i ∈ 2:length(t)
                Makie.$(Symbol(fun,:!))(t[i],f;args...)
            end
        end
    end
end
for fun ∈ (:(Makie.scatter),:(Makie.scatter!),:(Makie.wireframe),:(Makie.wireframe!),:linegraph,:linegraph!)
    @eval $fun(t::TensorField{B,<:AbstractComplex} where B;args...) = $fun(vectorize(t);args...)
end
for fun ∈ (:(Makie.arrows),:(Makie.arrows!),:scaledarrows,:scaledarrows!)
    @eval begin
        $fun(t::TensorField{B,<:AbstractComplex} where B;args...) = $fun(vectorize(t);args...)
        $fun(M::TensorField{B,<:AbstractComplex} where B,t::TensorField{B,<:AbstractComplex} where B;args...) = $fun(vectorize(M),vectorize(t);args...)
    end
end
for fun ∈ (:mesh,:streamplot)
    @eval begin
        Makie.$fun(t::TensorField{B,<:AbstractComplex,2} where B;args...) = Makie.$fun(vectorize(t);args...)
        Makie.$(Symbol(fun,:!))(t::TensorField{B,<:AbstractComplex,2} where B;args...) = Makie.$(Symbol(fun,:!))(vectorize(t);args...)
        Makie.$fun(M::TensorField{B,<:AbstractComplex,2} where B,t::TensorField{B,<:AbstractComplex,2} where B;args...) = Makie.$fun(vectorize(M),Real(angle(t));args...)
        Makie.$(Symbol(fun,:!))(M::TensorField{B,<:AbstractComplex,2} where B,t::TensorField{B,<:AbstractComplex,2} where B;args...) = Makie.$(Symbol(fun,:!))(vectorize(M),Real(angle(t)),args...)
    end
end
Makie.mesh(t::TensorField{B,<:AbstractComplex,2} where B,f::Function;args...) = Makie.mesh(vectorize(t),vectorize(f(t));args...)
Makie.mesh!(t::TensorField{B,<:AbstractComplex,2} where B,f::Function;args...) = Makie.mesh!(vectorize(t),vectorize(f(t));args...)

for lines ∈ (:lines,:lines!,:linesegments,:linesegments!)
    @eval begin
        Makie.$lines(t::RectangleMap;args...) = Makie.$lines(boundarycomponents(t);args...)
        Makie.$lines(t::HyperrectangleMap;args...) = Makie.$lines(boundarycomponents(t);args...)
        Makie.$lines(t::ScalarMap;args...) = Makie.$lines(TensorField(GridBundle{1}(base(t)),fiber(t));args...)
        Makie.$lines(t::SpaceCurve,f::RealFunction;args...) = Makie.$lines(vec(fiber(t));color=Real.(vec(fiber(f))),args...)
        Makie.$lines(t::PlaneCurve,f::RealFunction=speed;args...) = Makie.$lines(vec(fiber(t));color=Real.(vec(fiber(f))),args...)
        Makie.$lines(t::RealFunction,f::RealFunction=speed;args...) = Makie.$lines(Real.(points(t)),Real.(fiber(t));color=Real.(vec(fiber(f))),args...)
        Makie.$lines(t::AbstractCurve,f::Function=speed;args...) = Makie.$lines(t,f(t);args...)
        Makie.$lines(t::ComplexMap{B,F,1},f::Function=speed;args...) where {B<:Coordinate{<:AbstractReal},F} = Makie.$lines(t,f(t);args...)
        Makie.$lines(t::ComplexMap{B,F,1},f::RealFunction;args...) where {B<:Coordinate{<:AbstractReal},F} = Makie.$lines(Cartan.realvalue.(fiber(t)),Cartan.imagvalue.(fiber(t));color=Real.(vec(fiber(f))),args...)
        #Makie.$lines(t::TensorField{B,F<:AbstractReal,N,<:SimplexBundle};args...) = Makie.$lines(TensorField(GridBundle{1}(base(t)),fiber(t));args...)
    end
end
#Makie.lines(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
#Makie.lines!(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
for fun ∈ (:linegraph,:linegraph!)
    @eval begin
        $fun(t::SurfaceGrid;args...) = $fun(graph(t);args...)
        $fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = $fun(TensorField(GridBundle(base(t)),fiber(t)))
    end
end
function linegraph(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
    x,y = Real.(points(t)),value.(fiber(t))
    display(Makie.lines(x,Real.(getindex.(y,1));args...))
    for i ∈ 2:Grassmann.binomial(mdims(fiber(t)),grade(t))
        Makie.lines!(x,Real.(getindex.(y,i));args...)
    end
end
function linegraph!(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
    x,y = Real.(points(t)),value.(fiber(t))
    display(Makie.lines!(x,Real.(getindex.(y,1));args...))
    for i ∈ 2:Grassmann.binomial(mdims(fiber(t)),grade(t))
        Makie.lines!(x,Real.(getindex.(y,i));args...)
    end
end
function Makie.lines(t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
    display(Makie.lines(getindex.(t,1),f;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.lines!(getindex.(t,i),f;args...)
    end
end
function Makie.lines!(t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
    display(Makie.lines!(getindex.(t,1),f;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.lines!(getindex.(t,i),f;args...)
    end
end
function Makie.lines(M::AbstractCurve,t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
    display(Makie.lines(M+getindex.(t,1),f;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.lines!(M+getindex.(t,i),f;args...)
    end
end
function Makie.lines!(M::AbstractCurve,t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
    display(Makie.lines!(M+getindex.(t,1),f;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.lines!(M+getindex.(t,i),f;args...)
    end
end

for (fun,pla) ∈ ((:tangentbundle,:planesbundle),(:tangentbundle!,:planesbundle!))
    @eval $fun(M::TensorField{B,<:Chain{V,1} where V,2} where B,t=jacobian(M);args...) = $pla(M,t;args...)
end
for (fun,arr) ∈ ((:tangentbundle,:arrowsbundle),(:tangentbundle!,:arrowsbundle!))
    @eval $fun(M::TensorField{B,<:Chain{V,1} where V,1} where B,t::VectorField=gradient(M);args...) = $arr(M,t;args...)
end
for (fun,pla) ∈ ((:normalbundle,:planesbundle),(:normalbundle!,:planesbundle!))
    @eval $fun(M::TensorField{B,<:Chain{V,1} where V,2} where B,t=normalframe(M);args...) = $pla(M,t;args...)
end
for (fun,arr) ∈ ((:normalbundle,:arrowsbundle),(:normalbundle!,:arrowsbundle!))
    @eval $fun(M::TensorField{B,<:Chain{V,1} where V,1} where B,t::VectorField=normalframe(M);args...) = $arr(M,t;args...)
end

for (fun,arr) ∈ ((:planesbundle,:arrows),(:planesbundle!,:arrows!))
    @eval function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;poly=false,args...)
        M,t,kwargs = gridargs(M,t,args)
        s = Cartan.spacing(M)/minimum(value(sum(map.(norm,fiber(value(t))))/length(t)))
        display(Makie.$arr(M,TensorField(base(t),sum.(value.(value.(fiber(t)))));argarrows(t,s/2)...))
        if poly
        v = vec(fiber(t)*(s/2))
        Makie.poly!(unorientedpoly.(vec(fiber(M)),getindex.(v,1),getindex.(v,2));kwargs...)
        else
        for ij ∈ ProductTopology(size(M)...)
            v = fiber(t)[ij...]*(s/2)
            Makie.mesh!(Cartan.unorientedplane(fiber(M)[ij...],v[1],v[2]);kwargs...)
        end
        end
    end
end

for (fun,arr) ∈ ((:spacesbundle,:arrows),(:spacesbundle!,:arrows!))
    @eval function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;poly=false,args...)
        M,t,kwargs = gridargs(b,f,args)
        s = Cartan.spacing(M)/minimum(value(sum(map.(norm,fiber(value(t))))/length(t)))
        display(Makie.$arr(M,TensorField(base(t),sum.(value.(value.(fiber(t)))));argarrows(t,s/2)...))
        if poly
        v = vec(fiber(t)*(s/2))
        v1,v2,v3 = getindex.(v,1),getindex.(v,2),getindex.(v,3)
        Makie.poly!(unorientedpoly.(vec(fiber(M)),v1,v2);kwargs...)
        Makie.poly!(unorientedpoly.(vec(fiber(M)),v1,v3);kwargs...)
        Makie.poly!(unorientedpoly.(vec(fiber(M)),v2,v3);kwargs...)
        else
        for ij ∈ ProductTopology(size(M)...)
            p,v = fiber(M)[ij...],fiber(t)[ij...]*(s/2)
            Makie.mesh!(Cartan.unorientedplane(p,v[1],v[2]);kwargs...)
            Makie.mesh!(Cartan.unorientedplane(p,v[1],v[3]);kwargs...)
            Makie.mesh!(Cartan.unorientedplane(p,v[2],v[3]);kwargs...)
        end
        end
    end
end

for (fun,sca) ∈ ((:arrowsbundle,:scatter),(:arrowsbundle!,:scatter!))
    @eval begin
        function $fun(b::VectorField,f::VectorField;args...)
            M,t,kwargs = gridargs(b,f,args)
            s = Cartan.spacing(M)/(sum(fiber(norm(t)))/length(t))
            display(Makie.$sca(vec(fiber(M))))
            Makie.arrows!(M,t;argarrows(t,s/2)...,kwargs...)
            Makie.arrows!(M,-t;argarrows(t,s/2)...,kwargs...)
        end
        function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;args...)
            M,t,kwargs = gridargs(b,f,args)
            s = Cartan.spacing(M)/minimum(value(sum(map.(norm,fiber(value(t))))/length(t)))
            display(Makie.$sca(vec(fiber(M))))
            Makie.arrows!(M,t;argarrows(t,s/2)...,kwargs...)
            Makie.arrows!(M,-t;argarrows(t,s/2)...,kwargs...)
        end
    end
end

for (fun,arr,pln,spa) ∈ ((:scaledfield,:scaledarrows,:scaledplanes,:scaledspaces),(:scaledfield!,:scaledarrows!,:scaledplanes!,:scaledspaces!),(:scaledbundle,:arrowsbundle,:planesbundle,:spacesbundle),(:scaledbundle!,:arrowsbundle!,:planesbundle!,:spacesbundle!))
    @eval begin
        $fun(M::VectorField,t::VectorField;args...) = $arr(M,t;args...)
        function $fun(M::VectorField,t::TensorField{B,<:TensorOperator} where B;args...)
            N = mdims(fibertype(t))
            N==1 ? $arr(M,t;args...) : N==2 ? $pln(M,t;args...) : $spa(M,t;args...)
        end
    end
end

for (fun,pla) ∈ ((:scaledplanes,:planes),(:scaledplanes!,:planes!),(:scaledspaces,:spaces),(:scaledspaces!,:spaces!))
    @eval function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;args...)
        M,t,kwargs = gridargs(b,f,args)
        s = Cartan.spacing(M)/minimum(value(sum(map.(norm,fiber(value(t))))/length(t)))
        $pla(M,t;lengthscale=s/2,kwargs...)
    end
end

for (fun,arr) ∈ ((:planes,:arrows),(:planes!,:arrows!))
    @eval function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;lengthscale=1,poly=false,args...)
        M,t,kwargs = gridargs(b,f,args)
        display(Makie.$arr(M,TensorField(base(t),sum.(value.(value.(fiber(t)))));argarrows(t,lengthscale)...))
        if poly
        v = vec(fiber(t)*lengthscale)
        Makie.poly!(orientedpoly.(vec(fiber(M)),getindex.(v,1),getindex.(v,2));kwargs...)
        else
        for ij ∈ ProductTopology(size(M)...)
            v = fiber(t)[ij...]*lengthscale
            Makie.mesh!(Cartan.orientedplane(fiber(M)[ij...],v[1],v[2]);kwargs...)
        end
        end
    end
end

for (fun,arr) ∈ ((:spaces,:arrows),(:spaces!,:arrows!))
    @eval function $fun(b::VectorField,f::TensorField{B,<:TensorOperator} where B;lengthscale=1,poly=false,args...)
        M,t,kwargs = gridargs(b,f,args)
        display(Makie.$arr(M,TensorField(base(t),sum.(value.(value.(fiber(t)))));argarrows(t,lengthscale)...,kwargs...))
        if poly
        v = vec(fiber(t)*lengthscale)
        v1,v2,v3 = getindex.(v,1),getindex.(v,2),getindex.(v,3)
        Makie.poly!(orientedpoly.(vec(fiber(M)),v1,v2))
        Makie.poly!(orientedpoly.(vec(fiber(M)),v1,v3))
        Makie.poly!(orientedpoly.(vec(fiber(M)),v2,v3))
        else
        for ij ∈ ProductTopology(size(M)...)
            p,v = fiber(M)[ij...],fiber(t)[ij...]*lengthscale
            Makie.mesh!(Cartan.orientedplane(p,v[1],v[2]))
            Makie.mesh!(Cartan.orientedplane(p,v[1],v[3]))
            Makie.mesh!(Cartan.orientedplane(p,v[2],v[3]))
        end
        end
    end
end

for fun ∈ (:arrows,:arrows!)
    @eval begin
        $(Symbol(:scaled,fun))(t::VectorField;args...) = $(Symbol(:scaled,fun))(TensorField(base(t)),t;args...)
        $(Symbol(:scaled,fun))(t::TensorField{B,<:TensorOperator,N,<:GridBundle} where {B,N};args...) = $(Symbol(:scaled,fun))(TensorField(base(t)),t)
        function $(Symbol(:scaled,fun))(b::VectorField,f::VectorField;args...)
            M,t,kwargs = gridargs(b,f,args)
            s = Cartan.spacing(M)/(sum(fiber(norm(t)))/length(t))
            Makie.$fun(M,t;argarrows(t,s/3,s/17)...,kwargs...)
        end
        function $(Symbol(:scaled,fun))(b::VectorField,f::TensorField{B,<:TensorOperator} where B;args...)
            M,t,kwargs = gridargs(b,f,args)
            s = Cartan.spacing(M)/maximum(value(sum(map.(norm,fiber(value(t))))/length(t)))
            Makie.$fun(M,t;argarrows(t,s/3,s/17)...,kwargs...)
        end
    end
end
@eval begin
    function Makie.arrows(b::VectorField,f::TensorField{B,<:TensorOperator,N,<:GridBundle} where B;args...) where N
        M,t,kwargs = gridargs(b,f,args)
        Makie.arrows(TensorField(fiber(M),fiber(t));kwargs...)
    end
    function Makie.arrows!(b::VectorField,f::TensorField{B,<:TensorOperator,N,<:GridBundle} where B;args...) where N
        M,t,kwargs = gridargs(b,f,args)
        Makie.arrows!(TensorField(fiber(M),fiber(t));kwargs...)
    end
end
function Makie.arrows(t::VectorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}};args...)
    display(Makie.arrows(getindex.(t,1);args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.arrows!(getindex.(t,i);args...)
    end
end
function Makie.arrows!(t::VectorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}};args...)
    display(Makie.arrows!(getindex.(t,1);args...))
    for i ∈ 2:mdims(eltype(fber(t)))
        Makie.arrows!(getindex.(t,i);args...)
    end
end
for (fun,fun!) ∈ ((:arrows,:arrows!),(:streamplot,:streamplot!))
    @eval begin
        function Makie.$fun(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridBundle};args...) where N
            display(Makie.$fun(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(fiber(t)))
                Makie.$fun!(getindex.(t,i);args...)
            end
        end
        function Makie.$fun!(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridBundle};args...) where N
            display(Makie.$fun!(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(fiber(t)))
                Makie.$fun!(getindex.(t,i);args...)
            end
        end
    end
end
function Makie.streamplot(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}},m::U;args...) where U<:Union{<:VectorField,<:Function}
    display(Makie.streamplot(getindex.(t,1),m;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.streamplot!(getindex.(t,i),m;args...)
    end
end
function Makie.streamplot!(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}},m::U;args...) where U<:Union{<:VectorField,<:Function}
    display(Makie.streamplot!(getindex.(t,1),m;args...))
    for i ∈ 2:mdims(eltype(fiber(t)))
        Makie.streamplot!(getindex.(t,i),m;args...)
    end
end

for fun ∈ (:volume,:volume!,:contour,:contour!,:voxels,:voxels!)
    @eval function Makie.$fun(t::VolumeGrid;args...)
        p = points(t).v
        Makie.$fun(Makie.:..(p[1][1],p[1][end]),Makie.:..(p[2][1],p[2][end]),Makie.:..(p[3][1],p[3][end]),Real.(fiber(t));args...)
    end
end
for fun ∈ (:volumeslices,:volumeslices!)
    @eval Makie.$fun(t::VolumeGrid;args...) = Makie.$fun(points(t).v...,Real.(fiber(t));args...)
end
for fun ∈ (:surface,:surface!)
    @eval begin
        Makie.$fun(t::SurfaceGrid,f::Function=gradient_fast;args...) = Makie.$fun(points(t).v...,Real.(fiber(t));color=Real.(abs.(fiber(f(Real(t))))),args...)
        Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(radius.(fiber(t)));color=Real.(angle.(fiber(t))),colormap=:twilight,args...)
        Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
        function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G,f::Function=gradient_fast;args...) where {B,F<:Chain}
            x,y = points(t),value.(fiber(t))
            yi = Real.(getindex.(y,1))
            display(Makie.$fun(x.v...,yi;color=Real.(abs.(fiber(f(x→yi)))),args...))
            for i ∈ 2:Grassmann.binomial(mdims(eltype(fiber(t))),grade(t))
                yi = Real.(getindex.(y,i))
                Makie.$(funsym(fun))(x.v...,yi;color=Real.(abs.(fiber(f(x→yi)))),args...)
            end
        end
    end
end
for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!)
    @eval begin
        Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(radius.(fiber(t)));args...)
    end
end
for fun ∈ (:heatmap,:heatmap!)
    @eval begin
        Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(angle.(fiber(t)));colormap=:twilight,args...)
    end
end
for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!)
    @eval begin
        Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(points(t).v...,Real.(fiber(t));args...)
        Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
        function Makie.$fun(t::TensorField{B,F,2,<:RealSpace{2}};args...) where {B,G,F<:Chain{V,G} where V}
            x,y = points(t),value.(fiber(t))
            display(Makie.$fun(x.v...,Real.(getindex.(y,1));args...))
            for i ∈ 2:Grassmann.binomial(mdims(eltype(fiber(t))),G)
                Makie.$(funsym(fun))(x.v...,Real.(getindex.(y,i));args...)
            end
        end
    end
end
for fun ∈ (:wireframe,:wireframe!)
    @eval begin
        Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(graph(t);gridargs(t,Makie.$fun,args)...)
        Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t));args...)
    end
end

import Cartan: point2chain, makietransform, streamargs
quatf3vec(q) = q * Makie.Vec3f(0,0,1)
chain3vec(x) = Makie.Vec3(x[1],x[2],x[3])
chain3quatf(x) = Makie.to_rotation(chain3vec(x))

function makietransform(M,st::Makie.FigureAxisPlot,N::Val)
    fig,ax,pl = st
    maketransform(M,pl,N)
    return st
end
function makietransform(M,pl,::Val{N}) where N
    pl.transformation.transform_func[] = Makie.PointTrans{N}() do p
        return Makie.Point(M(p))
    end
    return pl
end

for fun ∈ (:streamplot,:streamplot!)
    @eval begin
        #Makie.$fun(f::Function,t::Rectangle;args...) = Makie.$fun(f,t.v...;args...)
        #Makie.$fun(f::Function,t::Hyperrectangle;args...) = Makie.$fun(f,t.v...;args...)
        Makie.$fun(m::ScalarField{<:Coordinate{<:Chain},<:AbstractReal,N,<:RealSpace} where N;args...) = Makie.$fun(gradient_fast(m);args...)
        Makie.$fun(m::ScalarMap,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
        Makie.$fun(m::VectorField{R,F,1,<:SimplexBundle} where {R,F},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
        Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealSpace} where {F,N};args...) = Makie.$fun(m,points(m).v...;args...)
        Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealSpace} where {F,N},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),dims...;streamargs(m,args)...)
        Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t));args...)
        function Makie.$fun(M::VectorField,m::VectorField{<:Coordinate{<:Chain{V}},<:Chain,2,<:RealSpace{2}};args...) where V
            dim = mdims(fibertype(M)) ≠ 2
            kwargs = streamargs(dim,args)
            st = if dim
                w,gs = Cartan.widths(points(m)),kwargs[:gridsize]
                scale = 0.2sqrt(surfacearea(M)/prod(w))
                Makie.$fun(p->(z=m(p);Makie.Point(z[1],z[2],0)),points(m).v...,Makie.ClosedInterval(-1e-15,1e-15);arrow_size=scale*minimum(w)/minimum((gs[1],gs[2])),kwargs...)
            else
                Makie.$fun(m;kwargs...)
            end
            $(fun≠:streamplot ? :pl : :((fig,ax,pl))) = st
            if dim
                makietransform(M,st,Val(3))
                jac,arr = jacobian(M),pl.plots[2]
                arr.rotation[] = chain3quatf.(jac.(point2chain.(arr.args.value[][1],V)).⋅point2chain.(quatf3vec(arr.rotation[]),V))
            else
                xs = getindex.(fiber(M),1); ys = getindex.(fiber(M),2)
                $(fun≠:streamplot ? nothing : :(ax.limits = ((minimum(xs),maximum(xs)),(minimum(ys),maximum(ys)))))
                makietransform(M,st,Val(2))
            end
            return st
        end
        function Makie.$fun(M::VectorField,m::VectorField{<:Coordinate{<:Chain{V}},<:Chain,3,<:RealSpace{3}};args...) where V
            kwargs = streamargs(args)
            st = Makie.$fun(p->(z=m(Chain{V}(p[1],p[2],p[3]));Makie.Point(z[1],z[2],z[3])),points(m).v...;kwargs...)
            makietransform(M,st,Val(3))
            #xs = getindex.(fiber(M),1); ys = getindex.(fiber(M),2); zs = getindex.(fiber(M),3)
            #ax.limits = ((minimum(xs),maximum(xs)),(minimum(ys),maximum(ys)),(minimum(zs),maximum(zs)))
        end
    end
end

for (fun,fun2,fun3) ∈ ((:arrows,:arrows2d,:arrows3d),(:arrows!,:arrows2d!,:arrows3d!))
    @eval begin
        function Makie.$fun(t::VectorField;args...)
            mdims(fibertype(t))≠3 ? Makie.$fun2(t;args...) : Makie.$fun3(t;args...)
        end
        function Makie.$fun(b::VectorField,f::VectorField;args...)
            M,t,kwargs = gridargs(b,f,args)
            mdims(fibertype(M))≠3 ? Makie.$fun2(M,t;kwargs...) : Makie.$fun3(M,t;kwargs...)
        end
    end
end
for fun ∈ (:arrows2d,:arrows3d,:arrows2d!,:arrows3d!)
    @eval begin
        #Makie.$fun(t::ScalarField{<:Coordinate{<:Chain},F,2,<:RealSpace{2}} where F;args...) = Makie.$fun(vec(Makie.Point.(fiber(graph(Real(t))))),vec(Makie.Point.(fiber(normal(Real(t)))));args...)
        function Makie.$fun(f::VectorField{<:Coordinate{<:Chain{W,L,F,2} where {W,L,F}},<:Chain{V,G,T,2} where {V,G,T},2,<:AlignedRegion{2}};args...)
            t,kwargs = gridargs(f,args)
            Makie.$fun(points(t).v...,getindex.(fiber(t),1),getindex.(fiber(t),2);kwargs...)
        end
        function Makie.$fun(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}};args...)
            t,kwargs = gridargs(f,args)
            Makie.$fun(Makie.Point.(vec(points(t))),Makie.Point.(vec(fiber(t)));kwargs...)
        end
        function Makie.$fun(f::VectorField{<:Coordinate{<:Chain},F,N,<:GridBundle} where {F,N};args...)
            t,kwargs = gridargs(f,args)
            Makie.$fun(vec(Makie.Point.(points(t))),vec(Makie.Point.(fiber(t)));kwargs...)
        end
        function Makie.$fun(b::VectorField,f::VectorField;args...)
            M,t,kwargs = gridargs(b,f,args)
            Makie.$fun(vec(Makie.Point.(fiber(M))),vec(Makie.Point.(fiber(t)));kwargs...)
        end
        #Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
        #Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
    end
end

Makie.arrows(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.arrows(Makie.Point.(↓(Manifold(base(t))).(points(t))),Makie.Point.(fiber(t));args...)
Makie.arrows!(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.arrows!(Makie.Point.(↓(Manifold(base(t))).(points(t))),Makie.Point.(fiber(t));args...)

Makie.convert_arguments(P::Makie.PointBased, a::SimplexBundle) = Makie.convert_arguments(P, Vector(points(a)))
Makie.convert_single_argument(a::LocalFiber) = convert_arguments(P,Point(a))

#Makie.scatter(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.scatter(submesh(base(t))[:,1],fiber(t);args...)
#Makie.scatter!(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.scatter!(submesh(base(t))[:,1],fiber(t);args...)
Makie.scatter(p::RealFunction;args...) = Makie.scatter(points(p),fiber(p);args...)
Makie.scatter!(p::RealFunction;args...) = Makie.scatter!(points(p),fiber(p);args...)
Makie.scatter(p::TensorField;args...) = Makie.scatter(vec(fiber(p));args...)
Makie.scatter!(p::TensorField;args...) = Makie.scatter!(vec(fiber(p));args...)
Makie.scatter(p::SimplexBundle;args...) = Makie.scatter(submesh(p);args...)
Makie.scatter!(p::SimplexBundle;args...) = Makie.scatter!(submesh(p);args...)
Makie.scatter(p::FaceBundle;args...) = Makie.scatter(submesh(fiber(means(p)));args...)
Makie.scatter!(p::FaceBundle;args...) = Makie.scatter!(submesh(fiber(means(p)));args...)

Makie.text(p::SimplexBundle;args...) = Makie.text(submesh(p);text=string.(vertices(p)),args...)
Makie.text!(p::SimplexBundle;args...) = Makie.text!(submesh(p);text=string.(vertices(p)),args...)
Makie.text(p::FaceBundle;args...) = Makie.text(submesh(fiber(means(p)));text=string.(subelements(p)),args...)
Makie.text!(p::FaceBundle;args...) = Makie.text!(submesh(fiber(means(p)));text=string.(subelements(p)),args...)

Makie.lines(p::SimplexBundle;args...) = Makie.lines(Vector(points(p));args...)
Makie.lines!(p::SimplexBundle;args...) = Makie.lines!(Vector(points(p));args...)
#Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(Makie.Point.(p);args...)
#Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(Makie.Point.(p);args...)
#Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
#Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
#Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
#Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)

for (fun,lin,disp) ∈ ((:linegraph,:(Makie.lines),:display),(:linegraph!,:(Makie.lines!),:identity))
    @eval begin
        function $fun(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::Function=speed;args...)
            if haskey(args,:gridsize)
                wargs = Dict(args)
                delete!(wargs,:gridsize)
                kwargs = (;wargs...)
                n = args[:gridsize]
                out = variation!(M,$lin,Makie.lines!,n[1],f;kwargs...)
                Cartan._alteration(out,M,0.0,Makie.lines!,Makie.lines!,n[2],Val(false),f;kwargs...)
            #=elseif haskey(args,:arcgridsize)
                wargs = Dict(args)
                delete!(wargs,:arcgridsize)
                kwargs = (;wargs...)
                n = args[:arcgridsize]
                return fun(arcresample(t,args[:arcgridsize]);kwargs...)=#
            else
                out = variation!(M,$lin,Makie.lines!,f;args...)
                Cartan._alteration(out,M,0.0,Makie.lines!,Makie.lines!,Val(false),f;args...)
            end
        end
        function $fun(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::TensorField;args...)
            rf = fiber(Real(f))
            minmax = (minimum(rf),maximum(rf))
            if haskey(args,:gridsize)
                wargs = Dict(args)
                delete!(wargs,:gridsize)
                kwargs = (;wargs...)
                n = args[:gridsize]
                out = variation!(M,$lin,Makie.lines!,n[1],f;colorrange=minmax,kwargs...)
                Cartan._alteration(out,M,0.0,Makie.lines!,Makie.lines!,n[2],Val(false),f;colorrange=minmax,kwargs...)
            #=elseif haskey(args,:arcgridsize)
                wargs = Dict(args)
                delete!(wargs,:arcgridsize)
                kwargs = (;wargs...)
                n = args[:arcgridsize]
                return fun(arcresample(t,args[:arcgridsize]);kwargs...)=#
            else
                out = variation!(M,$lin,Makie.lines!,f;colorrange=minmax,args...)
                Cartan._alteration(out,M,0.0,Makie.lines!,Makie.lines!,Val(false),f;colorrange=minmax,args...)
            end
        end
        function $fun(v::TensorField{B,<:Chain,3,<:GridBundle} where B,f::Function=speed;args...)
            if haskey(args,:gridsize)
                wargs = Dict(args)
                delete!(wargs,:gridsize)
                kwargs = (;wargs...)
                n = args[:gridsize]
                x = resample(points(v).v[1],n[1])
                y = resample(points(v).v[2],n[3])
                z = resample(points(v).v[3],n[3])
                xyz = Values(x,y,z)
                $disp($lin(Cartan.leaf2(v,float(y[1]),float(z[1]),1),f;kwargs...))
                c = (2,3),(1,3),(1,2)
                for k ∈ (1,2,3)
                    xk = xyz[c[k][1]]
                    for i ∈ 1:length(xk)
                        xi,yk = xk[i],xyz[c[k][2]]
                        for j ∈ 1:length(yk)
                            Makie.lines!(Cartan.leaf2(v,float(xi),float(yk[j]),k),f;kwargs...)
                        end
                    end
                end
            #=elseif haskey(args,:arcgridsize)
                wargs = Dict(args)
                delete!(wargs,:arcgridsize)
                kwargs = (;wargs...)
                n = args[:arcgridsize]
                return fun(arcresample(t,args[:arcgridsize]);kwargs...)=#
            else
                $disp($lin(Cartan.leaf2(v,1,1,1),f;args...))
                c = (2,3),(1,3),(1,2)
                for k ∈ (1,2,3)
                    for i ∈ 1:length(points(v).v[c[k][1]])
                        for j ∈ 1:length(points(v).v[c[k][2]])
                            Makie.lines!(Cartan.leaf2(v,i,j,k),f;args...)
                        end
                    end
                end
            end
        end
        function $fun(v::TensorField{B,<:Chain,3,<:GridBundle} where B,f::TensorField;args...)
            rf = fiber(Real(f))
            minmax = (minimum(rf),maximum(rf))
            if haskey(args,:gridsize)
                wargs = Dict(args)
                delete!(wargs,:gridsize)
                kwargs = (;wargs...)
                n = args[:gridsize]
                x = resample(points(v).v[1],n[1])
                y = resample(points(v).v[2],n[3])
                z = resample(points(v).v[3],n[3])
                xyz = Values(x,y,z)
                $disp($lin(Cartan.leaf2(v,float(y[1]),float(z[1]),1),Cartan.leaf2(f,float(y[1]),float(z[1]),1);colorrange=minmax,kwargs...))
                c = (2,3),(1,3),(1,2)
                for k ∈ (1,2,3)
                    xk = xyz[c[k][1]]
                    for i ∈ 1:length(xk)
                        xi,yk = float(xk[i]),xyz[c[k][2]]
                        for j ∈ 1:length(yk)
                            Makie.lines!(Cartan.leaf2(v,xi,float(yk[j]),k),Cartan.leaf2(f,xi,float(yk[j]),k);colorrange=minmax,kwargs...)
                        end
                    end
                end
            #=elseif haskey(args,:arcgridsize)
                wargs = Dict(args)
                delete!(wargs,:arcgridsize)
                kwargs = (;wargs...)
                n = args[:arcgridsize]
                return fun(arcresample(t,args[:arcgridsize]);kwargs...)=#
            else
                $disp($lin(Cartan.leaf2(v,1,1,1),Cartan.leaf2(f,1,1,1);colorrange=minmax,args...))
                c = (2,3),(1,3),(1,2)
                for k ∈ (1,2,3)
                    for i ∈ 1:length(points(v).v[c[k][1]])
                        for j ∈ 1:length(points(v).v[c[k][2]])
                            Makie.lines!(Cartan.leaf2(v,i,j,k),Cartan.leaf2(v,i,j,k);colorrange=minmax,args...)
                        end
                    end
                end
            end
        end
    end
end

function Makie.linesegments(e::SimplexBundle;args...)
    sdims(immersion(e)) ≠ 2 && (return Makie.linesegments(edges(e)))
    Makie.linesegments(Grassmann.pointpair.(e[immersion(e)],↓(Manifold(e)));args...)
end
function Makie.linesegments!(e::SimplexBundle;args...)
    sdims(immersion(e)) ≠ 2 && (return Makie.linesegments!(edges(e)))
    Makie.linesegments!(Grassmann.pointpair.(e[immersion(e)],↓(Manifold(e)));args...)
end

#Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(base(t));color=Real.(fiber(t)),args...)
#Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(base(t));color=Real.(fiber(t)),args...)
Makie.wireframe(t::SimplexBundle;args...) = Makie.linesegments(edges(t);args...)
Makie.wireframe!(t::SimplexBundle;args...) = Makie.linesegments!(edges(t);args...)
for fun ∈ (:wireframe,:wireframe!)
    @eval Makie.$fun(M::GridBundle;args...) = Makie.$fun(Makie.GeometryBasics.Mesh(M);args...)
end
for fun ∈ (:mesh,:mesh!)
    @eval Makie.$fun(M::GridBundle;args...) = Makie.$fun(Makie.GeometryBasics.Mesh(M);shading=mdims(M)≠2,backlight=1,args...)
end

Makie.wireframe(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...) = Makie.wireframe(GridBundle(fiber(M));args...)
Makie.wireframe!(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...) = Makie.wireframe!(GridBundle(fiber(M));args...)
Makie.wireframe(M::TensorField{B,<:Chain,3,<:GridBundle} where B;args...) = Makie.wireframe(boundarycomponents(M);args...)
Makie.wireframe!(M::TensorField{B,<:Chain,3,<:GridBundle} where B;args...) = Makie.wireframe!(boundarycomponents(M);args...)
function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...)
    Makie.mesh(GridBundle(fiber(M));args...)
end
function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...)
    Makie.mesh!(GridBundle(fiber(M));args...)
end
function Makie.mesh(M::TensorField{B,<:Chain,3,<:GridBundle} where B;args...)
    Makie.mesh(boundarycomponents(M);args...)
end
function Makie.mesh!(M::TensorField{B,<:Chain,3,<:GridBundle} where B;args...)
    Makie.mesh!(boundarycomponents(M);args...)
end
function Makie.mesh(M::TensorField{B,<:AbstractReal,2,<:GridBundle} where B;args...)
    Makie.mesh(Makie.Mesh(base(M));color=vec(fiber(Real(M))),args...)
end
function Makie.mesh!(M::TensorField{B,<:AbstractReal,2,<:GridBundle} where B;args...)
    Makie.mesh!(Makie.Mesh(base(M));color=vec(fiber(Real(M))),args...)
end
function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::TensorField;args...)
    Makie.mesh(GridBundle(fiber(M));color=vec(fiber(Real(f))),args...)
end
function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::TensorField;args...)
    Makie.mesh!(GridBundle(fiber(M));color=vec(fiber(Real(f))),args...)
end
Makie.mesh(M::TensorField,f::Function;args...) = ndims(M)≠2 ? Makie.mesh(boundarycomponents(M),f) : Makie.mesh(M,f(M);args...)
Makie.mesh!(M::TensorField,f::Function;args...) = ndims(M)≠2 ? Makie.mesh(boundarycomponents(M),f) : Makie.mesh!(M,f(M);args...)
Makie.mesh(M::TensorField{B,F,N,<:FaceBundle} where {B,F,N};args...) = Makie.mesh(interp(M);args...)
Makie.mesh!(M::TensorField{B,F,N,<:FaceBundle} where {B,F,N};args...) = Makie.mesh!(interp(M);args...)
Makie.mesh(t::ScalarMap;args...) = Makie.mesh(base(t);color=Real.(fiber(t)),args...)
Makie.mesh!(t::ScalarMap;args...) = Makie.mesh!(base(t);color=Real.(fiber(t)),args...)
function Makie.mesh(M::SimplexBundle;args...)
    if mdims(M) == 2
        sm = submesh(M)[:,1]
        Makie.lines(sm,args[:color])
        Makie.plot!(sm,args[:color])
    else
        Makie.mesh(submesh(M),Cartan.array(immersion(M));args...)
    end
end
function Makie.mesh!(M::SimplexBundle;args...)
    if mdims(M) == 2
        sm = submesh(M)[:,1]
        Makie.lines!(sm,args[:color])
        Makie.plot!(sm,args[:color])
    else
        Makie.mesh!(submesh(M),Cartan.array(immersion(M));args...)
    end
end

function Makie.surface(M::ScalarMap,f::Function=laplacian;args...)
    fM = f(M)
    col = isdiscontinuous(M) && !isdiscontinuous(fM) ? discontinuous(fM,base(M)) : fM
    Makie.mesh(hcat(submesh(base(M)),Real.(fiber(M))),Cartan.array(immersion(M));color=fiber(col),args...)
end
function Makie.surface!(M::ScalarMap,f::Function=laplacian;args...)
    fM = f(M)
    col = isdiscontinuous(M) && !isdiscontinuous(fM) ? discontinuous(fM,base(M)) : fM
    Makie.mesh!(hcat(submesh(base(M)),Real.(fiber(M))),Cartan.array(immersion(M));color=fiber(col),args...)
end
function Makie.surface(M::TensorField{B,<:AbstractReal,1,<:FaceBundle} where B,f::Function=laplacian;args...)
    Makie.surface(interp(M),f;args...)
end
function Makie.surface!(M::TensorField{B,<:AbstractReal,1,<:FaceBundle} where B,f::Function=laplacian;args...)
    Makie.surface!(interp(M),f;args...)
end

end # module
