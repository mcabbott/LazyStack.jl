# LazyStack.jl

[![Github CI](https://github.com/mcabbott/LazyStack.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/LazyStack.jl/actions?query=workflow%3ACI+branch%3Amaster)

This package exports one function, `lazystack`, for turning a list of arrays 
into one `AbstractArray`. Given several arrays with the same `eltype`, 
or an array of such arrays, it returns a lazy `Stacked{T,N}` view of these:

```julia
lazystack([zeros(2,2), ones(2,2)])  # isa Stacked{Float64, 3, <:Vector{<:Matrix}}
lazystack([1,2,3], 4:6)             # isa Stacked{Int, 2, <:Tuple{<:Vector, <:UnitRange}}
```

Before v0.1 this function used to be called `stack`, but that name is now exported by Base (from Julia 1.9).
However, `Base.stack` always returns a new dense array, not a lazy container.

Generators such as `lazystack([i,2i] for i in 1:5)` and arrays of mixed eltype like `lazystack([1,2], [3.0, 4.0], [5im, 6im])` used to be be handled here, and are now simply passed through to `Base.stack`.

When the individual slices aren't backed by an Array, as for instance with `CuArray`s on a GPU, then again `Base.stack` is called.
This should make one big `CuArray`, since scalar indexing of individual slices won't work well.

### Ragged stack

There is also a version which does not demand that slices have equal `size` (or equal `ndims`),
which always returns a new `Array`. You can control the position of slices `using OffsetArrays`:

```julia
raggedstack([1:n for n in 1:10])           # upper triangular Matrix{Int}
raggedstack(OffsetArray(fill(n,4), rand(-2:2)) for n in 1:10; fill=NaN)
```

### Other packages

This one plays well with [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl), and ChainRules-compatible AD such as [Zygote.jl](https://github.com/FluxML/Zygote.jl).

Besides which, there are several other ways to achieve similar things:

* For an array of arrays, you can also use [`JuliennedArrays.Align`](https://bramtayl.github.io/JuliennedArrays.jl/latest/#JuliennedArrays.Align). This requires (or enables) you to specify which dimensions of the output belong to the sub-arrays, instead of writing `PermutedDimsArray(stack(...), ...)`. 
* There is also [`RecursiveArrayTools.VectorOfArray`](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl#vectorofarray) which as its name hints only allows a one-dimensional container. Linear indexing retreives a slice, not an element, which is sometimes surprising.
* For a tuple of arrays, [`LazyArrays.Hcat`](https://github.com/JuliaArrays/LazyArrays.jl#concatenation) is at present faster to index than `stack`, but doesn't allow arbitrary dimensions.

And a few more:

* When writing this I missed [`SplitApplyCombine.combinedimsview`](https://github.com/JuliaData/SplitApplyCombine.jl#combinedimsviewarray), which is very similar to `stack`, but doesn't handle tuples.
* Newer than this package is [StackViews.jl](https://github.com/JuliaArrays/StackViews.jl) handles both, with `StackView(A,B,dims=4) == StackView([A,B],4)` creating a 4th dimension; the container is always one-dimensional. 
* [`Flux.stack`](https://fluxml.ai/Flux.jl/stable/utilities/#Flux.stack) similarly takes a dimension, but eagerly creates an `Array`.

The lazy inverse:

* The package [ArraysOfArrays.jl](https://github.com/JuliaArrays/ArraysOfArrays.jl) solves the opposite problem, of accessing one large array as if it were many slices.

* As does [`JuliennedArrays.Slices`](https://bramtayl.github.io/JuliennedArrays.jl/latest/#JuliennedArrays.Slices-Union{Tuple{NumberOfDimensions},%20Tuple{Item},%20Tuple{AbstractArray{Item,NumberOfDimensions},Vararg{Int64,N}%20where%20N}}%20where%20NumberOfDimensions%20where%20Item).

* As does [`PackedVectorsOfVectors`](https://github.com/synchronoustechnologies/PackedVectorsOfVectors.jl), although only 1+1 dimensions. Also has an eager `pack` method which turns a vector of vectors into view of a single larger matrix. 

* [`Base.eachslice`](https://docs.julialang.org/en/v1/base/arrays/#Base.eachslice) also views one large array as many slices. This is a generator, but [JuliaLang#32310](https://github.com/JuliaLang/julia/pull/32310) should upgrade it to a multi-dimensional container indexable container.

Eager:

* After writing this I learned of [JuliaLang#31644](https://github.com/JuliaLang/julia/pull/31644) which extends `reduce(hcat,...)` to work on generators. 

* Later, [JuliaLang#43334](https://github.com/JuliaLang/julia/pull/43334) has added a better version of this package's `stack_iter` method to Base.
