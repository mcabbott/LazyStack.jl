# LazyStack.jl

[![Build Status](https://travis-ci.org/mcabbott/LazyStack.jl.svg?branch=master)](https://travis-ci.org/mcabbott/LazyStack.jl)

This package exports one function, `stack`, for turning a list of arrays 
into one `AbstractArray`. Given several arrays with the same `eltype`, 
or an array of such arrays, it returns a lazy `Stacked{T,N}` view of these:

```julia
stack([zeros(2,2), ones(2,2)])  # isa Stacked{Float64, 3, <:Vector{<:Matrix}}
stack([1,2,3], 4:6)             # isa Stacked{Int, 2, <:Tuple{<:Vector, <:UnitRange}}
```

Given a generator, it instead iterates through the elements and writes into a new array.
Given a function and then some arrays, it behaves like `map(f, A, B)` but immediately writes
into a new array:

```julia
stack([i,2i] for i in 1:5)            # isa Matrix{Int}     # size(ans) == (2, 5)
stack(*, eachcol(ones(2,4)), 1:4)     # == Matrix(stack(map(*, eachcol(...), 1:4)))
```

The same `stack_iter` method is also used for any list of arrays of heterogeneous element type,
and for arrays of tuples. Notice that like `map(identity, Any[1, 1.0, 5im])`, this promotes using 
`promote_typejoin`, to `Number` here, rather than to `Complex{Float64}`:

```julia
stack([1,2], [3.0, 4.0], [5im, 6im])  # isa Matrix{Number}  # size(ans) == (2, 3)
stack([(i,2.0,3//j) for i=1:4, j=1:5])# isa Array{Real, 3}  # size(ans) == (3, 4, 5)
```

The slices must all have the same `size`, but they (and the container) 
can have any number of dimensions. `stack` always places the slice dimensions first.
There are no options.

### Ragged stack

There is also a version which does not demand that slices have equal `size` (or equal `ndims`),
which always returns a new `Array`. You can control the position of slices `using OffsetArrays`:

```julia
rstack([1:n for n in 1:10])           # upper triangular Matrix{Int}
rstack(OffsetArray(fill(n,4), rand(-2:2)) for n in 1:10; fill=NaN)
```

### Other packages

This one plays well with [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl),
[NamedDims.jl](https://github.com/invenia/NamedDims.jl), and 
[Zygote.jl](https://github.com/FluxML/Zygote.jl).

Besides which, there are several other ways to achieve similar things:

* For an array of arrays, you can also use [`JuliennedArrays.Align`](https://bramtayl.github.io/JuliennedArrays.jl/latest/#JuliennedArrays.Align). This requires (or enables) you to specify which dimensions of the output belong to the sub-arrays, instead of writing `PermutedDimsArray(stack(...), ...)`. 
* There is also [`RecursiveArrayTools.VectorOfArray`](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl#vectorofarray) which as its name hints only allows a one-dimensional container. Linear indexing retreives a slice, not an element, which is sometimes surprising.
* And there is [`SplitApplyCombine.combinedimsview`](https://github.com/JuliaData/SplitApplyCombine.jl#combinedimsviewarray), which is very similar to `stack`, but doesn't handle tuples.
* For a tuple of arrays, [`LazyArrays.Hcat`](https://github.com/JuliaArrays/LazyArrays.jl#concatenation) is at present faster to index than `stack`, but doesn't allow arbitrary dimensions.
* For a generator of arrays, the built-in `reduce(hcat,...)` may work, but it slow compared to `stack`: see [test/speed.jl](test/speed.jl) for some examples.

The package [ArraysOfArrays.jl](https://github.com/JuliaArrays/ArraysOfArrays.jl) solves the opposite problem, of accessing one large array as if it were many slices. As does [`JuliennedArrays.Slices`](https://bramtayl.github.io/JuliennedArrays.jl/latest/#JuliennedArrays.Slices-Union{Tuple{NumberOfDimensions},%20Tuple{Item},%20Tuple{AbstractArray{Item,NumberOfDimensions},Vararg{Int64,N}%20where%20N}}%20where%20NumberOfDimensions%20where%20Item), and of course [`Base.eachslice`](https://docs.julialang.org/en/v1/base/arrays/#Base.eachslice).

Finally, after writing this I learned of [julia/31644](https://github.com/JuliaLang/julia/pull/31644) which extends `reduce(hcat,...)` to work on generators. 
