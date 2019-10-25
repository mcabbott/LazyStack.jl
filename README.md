# LazyStack.jl

[![Build Status](https://travis-ci.org/mcabbott/LazyStack.jl.svg?branch=master)](https://travis-ci.org/mcabbott/LazyStack.jl)

This little package exports one function, `stack`, for turning a list of arrays 
into one `AbstractArray`. Given several `Array`s, or an `Array{<:Array{T}}`, 
it returns a lazy `Stacked{T,N}` view of these; use `collect∘stack` to copy into a new array.

Given a generator, it instead iterates through the elements and writes into a new array.
I guess that is lazy only in that it need not collect the generator first.
The same method is also used for any list of arrays of heterogeneous type.

The slices must all have the same `size`, but they (and the container) 
can have any number of dimensions. `Stacked` always places the slice dimensions first.
There are no options.
