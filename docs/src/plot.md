# Cartan.jl plotting

```@contents
Pages = ["index.md","fiber.md","videos.md","library.md","plot.md"]
```

## Makie.jl

[https://docs.makie.org/stable/reference](https://docs.makie.org/stable/reference)

### arrows

[https://docs.makie.org/stable/reference/plots/arrows](https://docs.makie.org/stable/reference/plots/arrows)

```julia
f = Figure(size = (800, 800))
Axis(f[1, 1], backgroundcolor = "black")

xs = LinRange(0, 2pi, 20)
ys = LinRange(0, 3pi, 20)
us = [sin(x) * cos(y) for x in xs, y in ys]
vs = [-cos(x) * sin(y) for x in xs, y in ys]
xy = TensorField(OpenParameter(xs,ys),Chain.(us,vs))
strength = vec(fiber(norm(xy)))

arrows2d!(xy, lengthscale = 0.2, color = strength)

f
```
```julia
using GLMakie
ps = OpenParameter(-5:2:5,-5:2:5,-5:2:5)
ns = map(p -> 0.1 * Chain(p[2], p[3], p[1]), ps)
arrows3d(
    TensorField(ps, ns),
    shaftcolor = :gray, tipcolor = :black,
    align = :center, axis=(type=Axis3,)
)
```
```julia
lengths = vec(norm.(ns))
arrows3d(
    TensorField(ps, ns), color = lengths, lengthscale = 1.5,
    align = :center, axis=(type=Axis3,)
)
```

### contour

[https://docs.makie.org/stable/reference/plots/contour](https://docs.makie.org/stable/reference/plots/contour)

```julia
f = Figure()
Axis(f[1, 1])

xs = LinRange(0, 10, 100)
ys = LinRange(0, 15, 100)
zs = [cos(x) * sin(y) for x in xs, y in ys]
xyz = TensorField(OpenParameter(xs,ys),zs)

contour!(xyz)
contour!(xyz,levels=-1:0.1:1)

f
```
```julia
himmelblau(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
x = y = range(-6, 6; length=100)
z = himmelblau.(x, y')

levels = 10.0.^range(0.3, 3.5; length=10)
colorscale = ReversibleScale(x -> x^(1 / 10), x -> x^10)
xyz = TensorField(OpenParameter(x,y),z)
f, ax, ct = contour(xyz; labels=true, levels, colormap=:hsv, colorscale)
f
```
```julia
x = -10:10
y = -10:10
# The curvilinear grid:
xs = [x + 0.01y^3 for x in x, y in y]
ys = [y + 10cos(x/40) for x in x, y in y]

# Now, for simplicity, we calculate the `zs` values to be
# the radius from the center of the grid (0, 10).
zs = sqrt.(xs .^ 2 .+ (ys .- 10) .^ 2)

# We can use Makie's tick finders to get some nice looking contour levels:
levels = Makie.get_tickvalues(Makie.LinearTicks(7), extrema(zs)...)

xyz = TensorField(GridBundle(Chain.(xs,ys)),zs)

# and now, we plot!
fig, ax, srf = mesh(xyz; shading = NoShading, axis = (; type = Axis, aspect = DataAspect()))
ctr = contour!(ax, xs, ys, zs; color = :orange, levels = levels, labels = true, labelfont = :bold, labelsize = 12)

fig
```

### contour3d

[https://docs.makie.org/stable/reference/plots/contour3d](https://docs.makie.org/stable/reference/plots/contour3d)

```julia
using GLMakie
f = Figure()
Axis3(f[1, 1], aspect=(0.5,0.5,1), perspectiveness=0.75)

xs = ys = LinRange(-0.5, 0.5, 100)
zs = [sqrt(x^2+y^2) for x in xs, y in ys]

xyz = TensorField(OpenParameter(xs,ys),zs)

contour3d!(-xyz, linewidth=2, color=:blue2)
contour3d!(+xyz, linewidth=2, color=:red2)

f
```
```julia
f = Figure()
Axis3(f[1, 1], aspect=(0.5,0.5,1), perspectiveness=0.75)

contour3d!(-xyz, levels=-(.025:0.05:.475), linewidth=2, color=:blue2)
contour3d!(+xyz, levels=  .025:0.05:.475,  linewidth=2, color=:red2)
```

```julia
r = range(-pi, pi, length = 21)
data2d = [cos(x) + cos(y) for x in r, y in r]
data3d = [cos(x) + cos(y) + cos(z) for x in r, y in r, z in r]

f = Figure(size = (700, 400))
a1 = Axis3(f[1, 1], title = "3D contour()")

rrr = OpenParameter(r,r,r)
contour!(TensorField(rrr, data3d))
```
```julia
a2 = Axis3(f[1, 2], title = "contour3d()")
rr = OpenParameter(r,r)
contour3d!(TensorField(rr,data2d), linewidth = 3, levels = 10)
f
```
```julia
f = Figure(size = (700, 300))
a1 = Axis3(f[1, 1])
contour!(TensorField(rrr,data3d), isorange = 0.04)

# small alpha can be used to see into the contour plot
a2 = Axis3(f[1, 2])
contour!(TensorField(rrr,data3d), data3d, alpha = 0.05)
f
```

### contourf

[https://docs.makie.org/stable/reference/plots/contourf](https://docs.makie.org/stable/reference/plots/contourf)

```julia
# continued from curvilinear contour example
f = Figure()

ax1 = Axis(f[1, 1])
ctrf1 = contourf!(TensorField(ProductSpace(x,y)), xyz;
    levels = levels)

ax2 = Axis(f[1, 2])
ctrf2 = contourf!(xy,xyz; levels = levels)
f
```

### heatmap

[https://docs.makie.org/stable/reference/plots/heatmap](https://docs.makie.org/stable/reference/plots/heatmap)

```julia
f = Figure()
ax = Axis(f[1, 1])

centers_x = [1, 2, 4, 7, 11]
centers_y = [6, 7, 9, 12, 16]
xy = ProductSpace(centers_x,centers_y)

heatmap!(TensorField(xy,reshape(1:25, 5, 5)))
scatter!(TensorField(xy,collect(xy)),
    color=:white, strokecolor=:black, strokewidth=1)
f
```

```julia
xs = range(0, 2π, length=100)
ys = range(0, 2π, length=100)
zs = [sin(x*y) for x in xs, y in ys]

xyz = TensorField(OpenParameter(xs,ys),zs)

fig, ax, hm = heatmap(xyz)
Colorbar(fig[:, end+1], hm)

fig
```
```julia
x = 10.0.^(1:0.1:4)
y = 1.0:0.1:5.0
z = broadcast((x, y) -> x - 10, x, y')

xyz = TensorField(OpenParameter(x,collect(y)),z)

scale = ReversibleScale(x -> asinh(x / 2) / log(10), x -> 2sinh(log(10) * x))
fig, ax, hm = heatmap(xyz; colorscale = scale, axis = (; xscale = scale))
Colorbar(fig[1, 2], hm)

fig
```

### lines

[https://docs.makie.org/stable/reference/plots/lines](https://docs.makie.org/stable/reference/plots/lines)

### linesegments

[https://docs.makie.org/stable/reference/plots/linesegments](https://docs.makie.org/stable/reference/plots/linesegments)

```julia
f = Figure()
Axis(f[1, 1])

xs = TensorField(1:0.2:10)
ys = sin(xs)

linesegments!(ys)
linesegments!(ys - 1, linewidth = 5)
linesegments!(ys - 2, linewidth = 5,
    color = LinRange(1, 5, length(xs)))
f
```

### mesh

[https://docs.makie.org/stable/reference/plots/mesh](https://docs.makie.org/stable/reference/plots/mesh)

```julia
rs = 1:10
thetas = 0:10:360

xs = rs .* cosd.(thetas')
ys = rs .* sind.(thetas')
zs = sin.(rs) .* cosd.(thetas')

xyz = TensorField(ProductSpace(rs,thetas),Chain.(xs,ys,zs))
mesh(xyz,TensorField(xyz,zs))
```
```julia
xy = TensorField(ProductSpace(rs,thetas),Chain.(xs,ys))
mesh(xy,TensorField(xy,zs))
```

### poly

[https://docs.makie.org/stable/reference/plots/poly](https://docs.makie.org/stable/reference/plots/poly)

### scatter

[https://docs.makie.org/stable/reference/plots/scatter](https://docs.makie.org/stable/reference/plots/scatter)

```julia
xs = range(0, 10, length = 30)
ys = 0.5 .* sin.(xs)
xy = TensorField(xs,ys)

scatter(xy)
```
```julia
xs = range(0, 10, length = 30)
ys = 0.5 .* sin.(xs)
pts = TensorField(xs,Chain.(xs, ys))

scatter(pts, color = 1:30, markersize = range(5, 30, length = 30),
    colormap = :thermal)
```

### scatterlines

[https://docs.makie.org/stable/reference/plots/scatterlines](https://docs.makie.org/stable/reference/plots/scatterlines)

### streamplot

[https://docs.makie.org/stable/reference/plots/streamplot](https://docs.makie.org/stable/reference/plots/streamplot)

```julia
v(x::Point2{T}) where T = Point2f(x[2], 4*x[1])
streamplot(v, -2..2, -2..2)
```
```julia
struct FitzhughNagumo{T}
    e::T
    s::T
    y::T
    b::T
end

P = FitzhughNagumo(0.1, 0.0, 1.5, 0.8)
fun(x) = fun(x, P)
fun(x, P::FitzhughNagumo) = Chain(
    (x[1]-x[2]-x[1]^3+P.s)/P.e,
    P.y*x[1]-x[2] + P.b)

xy = OpenParameter(-1.5:0.1:1.5,-1.5:0.1:1.5)
fig, ax, pl = streamplot(fun.(xy), colormap = :magma)
```
```julia
streamplot(fun.(xy), color=(p)-> RGBAf(p..., 0.0, 1))
```

### surface

[https://docs.makie.org/stable/reference/plots/surface](https://docs.makie.org/stable/reference/plots/surface)

```julia
using GLMakie
xs = LinRange(0, 10, 100)
ys = LinRange(0, 15, 100)
zs = [cos(x) * sin(y) for x in xs, y in ys]
xyz = TensorField(OpenParameter(xs,ys),zs)

surface(xyz, axis=(type=Axis3,))
```
```julia
using GLMakie
rs = 1:10
thetas = 0:10:360

xs = rs .* cosd.(thetas')
ys = rs .* sind.(thetas')
zs = sin.(rs) .* cosd.(thetas')
xyz = TensorField(OpenParameter(collect(rs),collect(thetas)),Chain.(xs,ys,zs))

mesh(xy,TensorField(xy,zs))
```
```julia
xy = TensorField(OpenParameter(collect(rs),collect(thetas)),Chain.(xs,ys))

mesh(xy,TensorField(xy,zs),shading=NoShading)
```

### volume

[https://docs.makie.org/stable/reference/plots/volume](https://docs.makie.org/stable/reference/plots/volume)

```julia
using GLMakie
r = LinRange(-1, 1, 100)
rrr = OpenParameter(r,r,r)
cube = Real(abs2(rrr))
contour(abs2(rrr),alpha=0.5)
```
```julia
cube_with_holes = cube*(cube .> 1.4)
Makie.volume(cube_with_holes,algorithm=:iso,isorange=0.05,isovalue=1.7)
```
```julia
using GLMakie
r = -5:5
data = map([(x,y,z) for x in r, y in r, z in r]) do (x,y,z)
    1 + min(abs(x), abs(y), abs(z))
end
colormap = [:red, :transparent, :transparent, RGBAf(0,1,0,0.5), :transparent, :blue]
rrr = OpenParameter(r,r,r)
volume(TensorField(rrr,data), algorithm = :indexedabsorption, colormap = colormap,
    interpolate = false, absorption = 5)
```

### voxels

[https://docs.makie.org/stable/reference/plots/voxels](https://docs.makie.org/stable/reference/plots/voxels)

```julia
using GLMakie
# Same as volume example
r = LinRange(-1, 1, 100)
rrr = OpenParameter(r,r,r)
cube = Real(abs2(rrr))
cube_with_holes = cube*(cube .> 1.4)

# To match the volume example with isovalue=1.7 and isorange=0.05 we map all
# values outside the range (1.65..1.75) to invisible air blocks with is_air
f, a, p = voxels(cube_with_holes, is_air = x -> !(1.65 <= x <= 1.75))
```
```julia
using GLMakie
chunk = TensorField(OpenParameter(3,3,3),reshape(collect(1:27), 3, 3, 3))
voxels(chunk, gap = 0.33)
```
```julia
using GLMakie
chunk = TensorField(OpenParameter(8,8,8),reshape(collect(1:512), 8, 8, 8))

f, a, p = voxels(chunk,
    colorrange = (65, 448), colorscale = log10,
    lowclip = :red, highclip = :orange,
    colormap = [:blue, :green]
)
```

### wireframe

[https://docs.makie.org/stable/reference/plots/wireframe](https://docs.makie.org/stable/reference/plots/wireframe)

```julia
using GLMakie
x, y = collect(-8:0.5:8), collect(-8:0.5:8)
z = [sinc(√(X^2 + Y^2) / π) for X ∈ x, Y ∈ y]
xyz = TensorField(ProductSpace(x,y),z)
wireframe(graph(xyz), axis=(type=Axis3,), color=:black)
```

## UnicodePlots.jl

[https://github.com/JuliaPlots/UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl)

```julia
using UnicodePlots
lineplot(TensorField([-1, 2, 3, 7], [-1, 2, 9, 4]), title="Example", name="my line", xlabel="x", ylabel="y")
```
```julia
plt = lineplot(TensorField([-1, 2, 3, 7], [-1, 2, 9, 4]), title="Example", name="my line",
               xlabel="x", ylabel="y", canvas=DotCanvas, border=:ascii)
```
```julia
lineplot!(plt, TensorField([0, 4, 8], [10, 1, 10]), color=:cyan, name="other line")
```

### lineplot

```julia
lineplot(TensorField([1, 2, 7], [9, -6, 8]), title="My Lineplot")
```
```julia
lineplot(TensorField(1:10, Chain.(0:9,3:12,reverse(5:14),fill(4, 10))), color=[:green :red :yellow :cyan])
```
```julia
lineplot(TensorField(1:10, 1:10), head_tail=:head, head_tail_frac=.1, height=4)
```

### scatterplot

```julia
scatterplot(TensorField(randn(50), randn(50)), title="My Scatterplot")
```
```julia
scatterplot(TensorField(1:10, 1:10), xscale=:log10, yscale=:log10)
```
```julia
scatterplot(TensorField(1:4, 1:4), xscale=:log10, yscale=:ln, unicode_exponent=false, height=6)
```
```julia
scatterplot(TensorField([1, 2, 3], [3, 4, 1]), marker=[:circle, '', "∫"],
            color=[:cyan, nothing, :yellow], height=2)
```

### histogram

```julia
histogram(TensorField(1:1000, randn(1_000) .* .1), nbins=15, closed=:left)
```
```julia
histogram(TensorField(1:1000, randn(1_000) .* .1), nbins=15, closed=:right, xscale=:log10)
```
```julia
histogram(TensorField(1:100000, randn(100_000) .* .1), nbins=60, vertical=true, height=10)
```

### boxplot

```julia
boxplot(TensorField(1:6, [1, 3, 3, 4, 6, 10]))
```
```julia
boxplot(TensorField(1:8,
        Chain.([1, 2, 3, 4, 5, 4, 3, 2], [2, 3, 4, 5, 6, 7, 8, 9])),
        title="Grouped Boxplot", xlabel="x")
```

### densityplot

```julia
plt = densityplot(Chain.(randn(10_000), randn(10_000)))
densityplot!(plt, Chain.(randn(10_000) .+ 2, randn(10_000) .+ 2))
```
```julia
x = randn(10_000); x[1_000:6_000] .= 2
densityplot(Chain.(x, randn(10_000)); dscale=x -> log(1 + x))
```

### contourplot

```julia
contourplot(TensorField(OpenParameter(-3:.01:3, -7:.01:3), xy -> exp(-(xy[1] / 2)^2 - ((xy[2] + 2) / 4)^2)))
```

### polarplot

```julia
polarplot(TensorField(range(0, 2π, length=20), range(0, 2, length=20)))
```

### heatmap

```julia
heatmap(TensorField(OpenParameter(11,11), repeat(collect(0:10)', outer=(11, 1))), zlabel="z")
```
```julia
heatmap(TensorField(OpenParameter(31,31), collect(0:30) * collect(0:30)'), xfact=.1, yfact=.1, xoffset=-1.5, colormap=:inferno)
```

### surface

```julia
sombrero(xy) = 15sinc(√(xy[1]^2 + xy[2]^2) / π)
surfaceplot(TensorField(OpenParameter(-8:.5:8,-8:.5:8),sombrero),colormap=:jet)
```
```julia
surfaceplot(TensorField(OpenParameter(-3:3, -3:3),
    xy -> 15sinc(√(xy[1]^2 + xy[2]^2) / π)),
    zscale=z -> 0, lines=true, colormap=:jet)
```

### isosurface

```julia
rrr = OpenParameter(-1:.1:1, -1:.1:1, -1:.1:1)
torus(xyz, r=0.2, R=0.5) = (√(xyz[1]^2 + xyz[2]^2) - R)^2 + xyz[3]^2 - r^2
isosurface(TensorField(rrr,torus), cull=true, zoom=2, elevation=50)
```
