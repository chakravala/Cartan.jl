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

ranges(x::Real) = ranges(x=>range(min(2x,iszero(x) ? -1 : 0),max(2x,iszero(x) ? 1 : 0),101))
ranges(x::AbstractRange) = float(x)
ranges(x::Pair{<:Real,<:AbstractRange}) = float(last(x))
ranges(x::Pair{String}) = ranges(last(x))
ranges(x::Pair{String,Bool}) = ranges(last(x))
ranges(x::Bool) = 0:1

slidertoggles(x::Real) = slidertoggles(x=>range(min(2x,iszero(x) ? -1 : 0),max(2x,iszero(x) ? 1 : 0),101))
slidertoggles(x::AbstractRange) = slider(float(x),snap=true)
slidertoggles(x::Pair{<:Real,<:AbstractRange}) = slider(float(last(x)),value=first(x),snap=true)
slidertoggles(x::Pair{String}) = slidertoggles(last(x))
slidertoggles(x::Pair{String,Bool}) = togglebutton(last(x),label=first(x))
slidertoggles(x::Bool) = togglebutton(last(x),label="True")

function no_slider_value!(sl)
    for s ∈ sl
        typeof(s) <: GtkObservables.Slider && (widget(s).draw_value = false)
    end
    return sl
end

boxstring(x::Pair{String}) = first(x)
boxstring(x::Pair{String,Bool}) = ""
boxstring(x::Pair{String,<:Real}) = first(x)*" = "*string(last(x))
boxstring(x::Pair{String,<:AbstractRange}) = first(x)*" ∈ ["*string(first(last(x)))*", "*string(last(last(x)))*"]"
boxstring(x::Pair{String,Pair{<:Real,<:AbstractRange}}) = first(x)*" ∈ ["*string(first(last(last(x))))*", "*string(last(last(last(x))))*"]"
boxstring(x::Pair{<:Real,<:AbstractRange}) = string(first(x))*" ∈ ["*string(first(last(x)))*", "*string(last(last(x)))*"]"
boxstring(x) = ""

function sliderbox(str::NTuple,args)
    bx = Gtk4.GtkBox(:v)
    isempty(args) && (return bx)
    for i ∈ 1:length(args)
        !isempty(str[i]) && push!(bx,Gtk4.GtkLabel("$i) "*str[i];halign=Gtk4.Align_START,margin_start=10))
        push!(bx,args[i])
        if typeof(args[i]) <: GtkObservables.Slider
            push!(bx,textbox(typeof(args[i][]); observable=observable(args[i])))
        end
    end
    scroll = Gtk4.GtkScrolledWindow()
    scroll.vexpand = true
    scroll.hexpand = true
    scroll.hscrollbar_policy = Gtk4.PolicyType_NEVER
    scroll.vscrollbar_policy = Gtk4.PolicyType_AUTOMATIC
    scroll.child = bx
    return scroll
end

makielimits!(ax::Gtk4Makie.Makie.Axis) = Gtk4Makie.Makie.autolimits!(ax)
makielimits!(ax::Gtk4Makie.Makie.LScene) = nothing #(ax.show_axis[] = false)

function plotbuttons(update,obj,osl)
    ax = obj.axis
    live = togglebutton(true; label="Live")
    reset = togglebutton(false; label="Reset")
    apply = button("Apply")
    on(apply) do _
        update()
        reset[] && makielimits!(ax)
    end
    on(live) do is_live
        if is_live
            update()
            reset[] && makielimits!(ax)
        end
    end
    for x in osl
        on(x) do _
            if live[]
                update()
                reset[] && makielimits!(ax)
            end
        end
    end
    bx = Gtk4.GtkBox(:h)
    push!(bx,live)
    push!(bx,apply)
    push!(bx,reset)
    return bx
end

function playbuttons(update,r,osl,k0=1)
    k = Observable(k0)
    dt = Observable(0.1)
    N = Observable(50)
    play = togglebutton(false; label="Play")
    tb1 = textbox(typeof(k[]); observable=observable(k))
    tb2 = textbox(typeof(dt[]); observable=observable(dt))
    tb3 = textbox(typeof(N[]); observable=observable(N))
    tb1.widget.width_chars = 2
    tb2.widget.width_chars = 4
    tb3.widget.width_chars = 4
    tb1.widget.width_request = 2
    tb2.widget.width_request = 4
    tb3.widget.width_request = 4
    i = Ref(0)
    on(play) do is_play
        is_play || return
        ok,n = k[],N[]
        ri = range(r[ok][1],r[ok][end],n)
        @async while play[]
            i[] = i[] ≥ n ? 1 : i[]+1
            osl[ok][] = ri[i[]]
            update()
            sleep(dt[])
        end
    end
    bx = Gtk4.GtkBox(:h)
    push!(bx,tb1)
    push!(bx,play)
    push!(bx,tb2)
    push!(bx,tb3)
    return bx
end

splitobservables(x::Observable) = (x,)
splitobservables(x::Observable{<:Tuple}) = ([(@lift $x[i]) for i ∈ 1:length(x[])]...,)

Cartan.gtkplot(plt::Function,args...;kwargs...) = Cartan.gtkplot(:h,plt,args...;kwargs...)
Cartan.gtkplot(plt::Function,fun::Function,arg;kwargs...) = Cartan.gtkplot(:v,plt,fun,arg;kwargs...)
Cartan.gtkplot(vh::Symbol,plt::Function,fun::Function,args...;kwargs...) = Cartan.gtkplot(vh,plt,fun,args;kwargs...)
function Cartan.gtkplot(vh::Symbol,plt::Function,fun::Function,args::Tuple;play=false,kwargs...)
    sl = no_slider_value!(slidertoggles.(args))
    osl = observable.(sl)
    params = Observable(getindex.(osl))
    update() = (params[] = getindex.(osl))
    y = @lift fun($params...)
    obj = plt(splitobservables(y)...;kwargs...)
    bx = sliderbox(boxstring.(args),sl)
    push!(bx.child.child,plotbuttons(update,obj,osl))
    play && push!(bx.child.child,playbuttons(update,ranges.(args),osl))
    Cartan.gtkplot(vh,obj,string(plt)*": "*string(fun),bx)
end

Cartan.gtkplot(vh::Symbol,plt::Function,fun1::Function,args1::Tuple,fun2::Function,args2...;kwargs...) = Cartan.gtkplot(vh,plt,fun1,args1,fun2,args2;kwargs...)
function Cartan.gtkplot(vh::Symbol,plt::Function,fun1::Function,args1::Tuple,fun2::Function,args2::Tuple;play=false,kwargs...)
    args = (args1...,args2...)
    sl1 = no_slider_value!(slidertoggles.(args1))
    sl2 = no_slider_value!(slidertoggles.(args2))
    osl1 = observable.(sl1)
    osl2 = observable.(sl2)
    osl = (osl1...,osl2...)
    params1 = Observable(getindex.(osl1))
    params2 = Observable(getindex.(osl2))
    function update()
        gosl1 = getindex.(osl1)
        if params1[] ≠ gosl1
            params1[] = gosl1
        end
        gosl2 = getindex.(osl2)
        if params2[] ≠ gosl2
            params2[] = gosl2
        end
    end
    y1 = @lift fun1($params1...)
    y2 = @lift fun2($y1,$params2...)
    obj = plt(splitobservables(y2)...;kwargs...)
    bx = sliderbox(boxstring.(args),(sl1...,sl2...))
    push!(bx.child.child,plotbuttons(update,obj,osl))
    play && push!(bx.child.child,playbuttons(update,ranges.(args),(osl1...,osl2...),length(sl1)+1))
    Cartan.gtkplot(vh,obj,string(plt)*": "*string(fun1)*", "*string(fun2),bx)
end

Cartan.gtkplot(vh::Symbol,plt::Function,fun1::Function,args1::Tuple,fun2::Function,args2::Tuple,fun3::Function,args3...;kwargs...) = Cartan.gtkplot(vh,plt,fun1,args1,fun2,args2,fun3,args3;kwargs...)
function Cartan.gtkplot(vh::Symbol,plt::Function,fun1::Function,args1::Tuple,fun2::Function,args2::Tuple,fun3::Function,args3::Tuple;play=false,kwargs...)
    args = (args1...,args2...,args3...)
    sl1 = no_slider_value!(slidertoggles.(args1))
    sl2 = no_slider_value!(slidertoggles.(args2))
    sl3 = no_slider_value!(slidertoggles.(args3))
    osl1 = observable.(sl1)
    osl2 = observable.(sl2)
    osl3 = observable.(sl3)
    osl = (osl1...,osl2...,osl3...)
    params1 = Observable(getindex.(osl1))
    params2 = Observable(getindex.(osl2))
    params3 = Observable(getindex.(osl3))
    function update()
        gosl1 = getindex.(osl1)
        if params1[] ≠ gosl1
            params1[] = gosl1
        end
        gosl2 = getindex.(osl2)
        if params2[] ≠ gosl2
            params2[] = gosl2
        end
        gosl3 = getindex.(osl3)
        if params3[] ≠ gosl3
            params3[] = gosl3
        end
    end
    y1 = @lift fun1($params1...)
    y2 = @lift fun2($y1,$params2...)
    y3 = @lift fun3($y2,$params3...)
    obj = plt(splitobservables(y3)...;kwargs...)
    bx = sliderbox(boxstring.(args),(sl1...,sl2...,sl3...))
    push!(bx.child.child,plotbuttons(update,obj,osl))
    play && push!(bx.child.child,playbuttons(update,ranges.(args),(osl1...,osl2...,osl3...),length(sl1)+length(sl2)+1))
    Cartan.gtkplot(vh,obj,string(plt)*": "*string(fun1)*", "*string(fun2)*", "*string(fun3),bx)
end

# Gtk4Makie only

Cartan.gtkplot(obj::Gtk4Makie.Makie.FigureAxisPlot,str=string(typeof(obj.plot))) = Cartan.gtkplot(:v,obj,str)
Cartan.gtkplot(vh::Symbol,plt::Function,args...;kwargs...) = Cartan.gtkplot(vh,plt(args...;kwargs...),string(plt))
function Cartan.gtkplot(vh::Symbol,obj::Gtk4Makie.Makie.FigureAxisPlot,str=string(typeof(obj.plot)),bx=Gtk4.GtkBox(:v))
    p = Gtk4.GtkPaned(vh;wide_handle=true,vexpand=true,resize_end_child=true,resize_start_child=false,shrink_start_child=false)
    p[1] = bx
    p[2] = Gtk4Makie.GtkMakieWidget()
    push!(p[2],obj)
    gtkwin(p,str)
end

function gtkwin(obj::Gtk4Makie.GtkWidget,str::String)
    win = Gtk4.GtkWindow(str)
    win[] = obj
    return win
end

Base.display(obj::Gtk4Makie.Makie.FigureAxisPlot) = gtkplot(obj)

end # module
