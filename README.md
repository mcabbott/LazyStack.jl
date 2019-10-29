# LazyStack.jl

[![Build Status](https://travis-ci.org/mcabbott/LazyStack.jl.svg?branch=master)](https://travis-ci.org/mcabbott/LazyStack.jl)

This package exports one function, `stack`, for turning a list of arrays 
into one `AbstractArray`. Given several arrays with the same `eltype`, 
or an array of such arrays, it returns a lazy `Stacked{T,N}` view of these. 

Given a generator, it instead iterates through the elements and writes into a new array.
(This is lazy only in that it need not `collect` the generator first.)
The same method is also used for any list of arrays of heterogeneous element type.

The slices must all have the same `size`, but they (and the container) 
can have any number of dimensions. `stack` always places the slice dimensions first.
There are no options.

### Other packages

This one plays well with [Zygote.jl](https://github.com/FluxML/Zygote.jl) 
and [NamedDims.jl](https://github.com/invenia/NamedDims.jl). 
Besides which, there are several other ways to achieve similar things:

* For an array of arrays, you can also use [`JuliennedArrays.Align`](https://bramtayl.github.io/JuliennedArrays.jl/latest/#JuliennedArrays.Align). This requires (or enables) you to specify which dimensions of the output belong to the sub-arrays, instead of writing `PermutedDimsArray(stack(...), ...)`.
* There is also [`RecursiveArrayTools.VectorOfArray`](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl#vectorofarray) which as its name hints only allows a one-dimensional container.
* For a tuple of arrays, [`LazyArrays.Hcat`](https://github.com/JuliaArrays/LazyArrays.jl#concatenation) is at present faster than `stack`!
* For a generator of arrays, the built-in `reduce(hcat,...)` may work, but it slow compared to `stack`: see [tests/speed.jl](tests/speed.jl) for some examples.
