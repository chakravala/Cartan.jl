using Documenter, Grassmann, Cartan

makedocs(
    # options
    modules = [Cartan],#Adapode],
    doctest = false,
    format = Documenter.HTML(prettyurls = true), #get(ENV, "CI", nothing) == "true"),
    remotes = nothing,
    sitename = "Cartan.jl",
    authors = "Michael Reed",
    pages = Any[
        "Home" => "index.md",
        "Fiber" => "fiber.md",
        "Videos" => "videos.md",
        "Library" => "library.md",
        "Plotting" => "plot.md",
        "AGPL-3.0" => "agpl.md"
        ]
)

deploydocs(
    repo   = "github.com/chakravala/Cartan.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)

# make multiline HTML
# find build -name "*.html" -exec tidy -m -i -q {} \;
