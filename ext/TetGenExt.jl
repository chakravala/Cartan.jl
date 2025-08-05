module TetGenExt

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
isdefined(Cartan, :Requires) ? (import Cartan: TetGen) : (using TetGen)

function TetGen.JLTetGenIO(mesh::SimplexBundle;
        marker = :markers, holes = TetGen.Point{3, Float64}[])
    f = TetGen.TriangleFace{Cint}.(immersion(mesh))
    kw_args = Any[:facets => f,:holes => holes]
    if hasproperty(f, marker)
        push!(kw_args, :facetmarkers => getproperty(f, marker))
    end
    pm = points(mesh); V = Manifold(pm)
    TetGen.JLTetGenIO(TetGen.Point.(↓(V).(pm)); kw_args...)
end
function Cartan.initmesh(tio::TetGen.JLTetGenIO, command = "Qp")
    r = TetGen.tetrahedralize(tio, command); V = Submanifold(ℝ^4)
    p = PointCloud([Chain{V,1}(Values{4,Float64}(1.0,k...)) for k ∈ r.points])
    t = Values{4,Int}.(r.tetrahedra)
    e = Values{3,Int}.(r.trifaces) # Values{2,Int}.(r.edges)
    n = Ref(length(p))
    return p(SimplexTopology(t,n)),p(SimplexTopology(e,n))
end
function TetGen.tetrahedralize(mesh::SimplexBundle, command = "Qp";
        marker = :markers, holes = TetGen.Point{3, Float64}[])
    initmesh(TetGen.JLTetGenIO(mesh;marker=marker,holes=holes),command)
end

end # module
