module Gtk4MakieExt

#   This file is part of Cartan.jl
#   It is licensed under the GPL license
#   Cartan Copyright (C) 2026 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

using Cartan
isdefined(Cartan, :Requires) ? (import Cartan: Gtk4Makie) : (using Gtk4Makie)
using GtkObservables
import Gtk4Makie: Gtk4

slidertoggles(x::Real) = slidertoggles(x=>range(min(2x,iszero(x) ? -1 : 0),max(2x,iszero(x) ? 1 : 0),101))
slidertoggles(x::AbstractRange) = slider(float(x))
slidertoggles(x::Pair{<:Real,<:AbstractRange}) = slider(float(last(x)),value=first(x),snap=true)
slidertoggles(x::Pair{String}) = slidertoggles(last(x))
slidertoggles(x::Pair{String,Bool}) = togglebutton(last(x),label=first(x))
slidertoggles(x::Bool) = togglebutton(last(x),label="True")

boxstring(x::Pair{String}) = first(x)
boxstring(x::Pair{String,Bool}) = ""
boxstring(x::Pair{String,<:Real}) = first(x)*" = "*string(last(x))
boxstring(x::Pair{String,<:AbstractRange}) = first(x)*" ∈ ["*string(first(last(x)))*", "*string(last(last(x)))*"]"
boxstring(x::Pair{String,Pair{<:Real,<:AbstractRange}}) = first(x)*" ∈ ["*string(first(last(last(x))))*", "*string(last(last(last(x))))*"]"
boxstring(x::Pair{<:Real,<:AbstractRange}) = string(first(x))*" ∈ ["*string(first(last(x)))*", "*string(last(last(x)))*"]"
boxstring(x) = ""

function sliderbox(str::NTuple,args...)
    bx = Gtk4.GtkBox(:v)
    for i ∈ 1:length(args)
        !isempty(str[i]) && push!(bx,Gtk4.GtkLabel(str[i];halign=Gtk4.Align_START,margin_start=10))
        push!(bx,args[i])
        if typeof(args[i]) <: GtkObservables.Slider
            push!(bx,textbox(typeof(args[i][]); observable=observable(args[i])))
        end
    end
    return bx
end

getaxis(x::Makie.FigureAxisPlot) = getaxis(x.axis)
getaxis(x::Axis) = x
getaxis(x::LScene) = x#.scene[1]

mylimits!(ax::Axis) = autolimits!(ax)
mylimits!(ax::LScene) = (ax.show_axis[] = false)

function plotbuttons(params,obj,osl)
    ax = getaxis(obj)
    live = togglebutton(true; label="Live")
    reset = togglebutton(false; label="Reset")
    apply = button("Apply")
    on(apply) do _
        params[] = getindex.(osl)
        reset[] && mylimits!(ax)
    end
    on(live) do is_live
        if is_live
            params[] = getindex.(osl)
            reset[] && mylimits!(ax)
        end
    end
    for x in osl
        on(x) do _
            if live[]
                params[] = getindex.(osl)
                reset[] && mylimits!(ax)
            end
        end
    end
    bx = Gtk4.GtkBox(:h)
    push!(bx,live)
    push!(bx,apply)
    push!(bx,reset)
    return bx
end

splitobservables(x::Observable) = (x,)
splitobservables(x::Observable{<:Tuple}) = ([(@lift $x[i]) for i ∈ 1:length(x[])]...,)

Cartan.gtkplot(plt::Function,args...;kwargs...) = Cartan.gtkplot(:h,plt,args...;kwargs...)
function Cartan.gtkplot(vh::Symbol,plt::Function,fun::Function,args...;kwargs...)
    sl = slidertoggles.(args)
    osl = observable.(sl)
    for s ∈ sl
        typeof(s) <: GtkObservables.Slider && (widget(s).draw_value = false)
    end
    win = Gtk4.GtkWindow(string(plt)*": "*string(fun))
    win[] = p = Gtk4.GtkPaned(vh;position=10,wide_handle=true,vexpand=true,resize_end_child=true,resize_start_child=false,shrink_start_child=false)
    params = Observable(getindex.(osl))
    y = @lift fun($params...)
    obj = plt(splitobservables(y)...;kwargs...)
    p[1] = sliderbox(boxstring.(args),sl...)
    p[2] = Gtk4Makie.GtkMakieWidget()
    push!(p[1],plotbuttons(params,obj,osl))
    push!(p[2],obj)
    return win
end

Cartan.gtkplot(obj::Gtk4Makie.Makie.FigureAxisPlot,str=string(typeof(obj.plot))) = Cartan.gtkplot(:v,obj,str)
Cartan.gtkplot(vh::Symbol,plt::Function,args...;kwargs...) = Cartan.gtkplot(vh,plt(args...;kwargs...),string(plt))
function Cartan.gtkplot(vh::Symbol,obj::Gtk4Makie.Makie.FigureAxisPlot,str=string(typeof(obj.plot)))
    win = Gtk4.GtkWindow(str)
    win[] = p = Gtk4.GtkPaned(vh;wide_handle=true,vexpand=true,resize_end_child=true,resize_start_child=false,shrink_start_child=false)
    p[1] = sliderbox(())
    p[2] = Gtk4Makie.GtkMakieWidget()
    push!(p[2],obj)
    return win
end

end # module
