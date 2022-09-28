export flatten
using Base: IteratorSize, HasShape, HasLength, @default_eltype, haslength

"""
    flatten([T], iter)

Given an iterator of iterators, returns a `Vector{T}` containing all of their elements.
Should be equal to `collect(Iterators.flatten(iter))`, but faster.

Acting on a vector of vectors, this is equal to `vcat(iter...)` or `reduce(vcat, iter)`,
and to `vec(stack(iter))` when this is allowed, i.e. when `allequal(length.(iter))`.

!!! Warning
   `flatten` is an experiment which got committed here by accident, more or less.
   I'm not sure it's part of the long-term future of LazyStack.

# Examples
```
julia> v = flatten((1:2, 8:9))
4-element Vector{Int64}:
 1
 2
 8
 9

julia> v == vcat(1:2, 8:9) == reduce(vcat, [1:2, 8:9]) == vec(stack((1:2, 8:9)))
true

julia> flatten(Complex{Float32}, [1:1, 9:10])  # specify eltype
3-element Vector{ComplexF32}:
  1.0f0 + 0.0f0im
  9.0f0 + 0.0f0im
 10.0f0 + 0.0f0im

julia> flatten(([1 2], [5.0], (x = false,)))  # ignores shape, promotes eltype
4-element Vector{Float64}:
 1.0
 2.0
 5.0
 0.0

julia> collect(Iterators.flatten(([1 2], [5.0], (x = false,))))  # wider eltype
4-element Vector{Real}:
     1
     2
     5.0
 false

julia> flatten(42)  # as numbers are iterable
1-element Vector{Int64}:
 42

julia> flatten(Vector{Int}[]), flatten(())  # empty case, with and without known eltype
(Int64[], Union{}[])
```
"""
function flatten end

"""
    flatten([T], f, args...)

Equivalent to `flatten(map(f, args...))`, but without allocating the result of `map`.

```
julia> flatten(x -> (x, x/2), 5:6)
4-element Vector{Real}:
 5
 2.5
 6
 3.0
```
"""
flatten(f, iter) = flatten(f(x) for x in iter)
flatten(f, xs, yzs...) = flatten(f(xy...) for xy in zip(xs, yzs...))

flatten(::Type{T}, iter) where {T} = _typed_flatten(T, IteratorSize(@default_eltype iter), iter)
flatten(::Type{T}, f, iter) where {T} = flatten(T, f(x) for x in iter)
flatten(::Type{T},f, xs, yzs...) where {T} = flatten(T, f(xy...) for xy in zip(xs, yzs...))

function flatten(iter)
  S = @default_eltype iter
  T = S != Union{} ? eltype(S) : Any  # Union{} occurs for e.g. flatten(1,2), postpone the error
  if isconcretetype(T)
      _typed_flatten(T, IteratorSize(S), iter)
  else
      _untyped_flatten(iter)
  end
end

function _typed_flatten(::Type{T}, ::Union{HasShape, HasLength}, A::Union{AbstractArray, Tuple}) where {T}
    len = sum(length, A; init=0)
    B = Vector{T}(undef, len)
    off = 1
    for x in A
        copyto!(B, off, x)
        off += length(x)
    end
    B
end

# _typed_flatten(::Type{T}, ::Union{HasShape, HasLength}, A) where {T} = _flatten(collect(A))

# Non-array iterators, whose elements are arrays or tuples, should probably be collected & sent to the above path.

function _typed_flatten(::Type{T}, ::IteratorSize, A) where {T}
    xit = iterate(A)
    nothing === xit && return T[]
    x1, _ = xit
    B = Vector{T}(undef, 0)
    if haslength(x1) && haslength(A)
        # sizehint!(B, _flatten_alloc_length(T, x1, A))
        guess = length(x1) * length(A)
        alloc = min(guess, (2^30) ÷ max(1, sizeof(T)))  # don't allocate > 1GB
        sizehint!(B, alloc)
    end
    while xit !== nothing
        x, state = xit
        append!(B, x)
        xit = iterate(A, state)
    end
    B
end

# function _flatten_alloc_length(::Type{T}, x1, A) where {T}
#     guess = length(x1) * length(A)
#     min(guess, (2^30) ÷ max(1, sizeof(T)))  # don't allocate > 1GB
# end

function _untyped_flatten(A)
    xit = iterate(A)
    nothing === xit && return Union{}[]
    x1, state = xit
    B = convert(Vector, vec(collect(x1)))  # can you do this better?
    # if haslength(x1) && haslength(A)  # doesn't seem to help
    #     sizehint!(B, _flatten_alloc_length(eltype(B), B, A))
    # end
    _untyped_flatten!(B, A, state)
end

#= For the empty case:

julia> collect(Iterators.flatten(()))
Union{}[]

=#

function _untyped_flatten!(B, A, state)
    xit = iterate(A, state)
    while xit !== nothing
        x, state = xit
        if eltype(x) <: eltype(B)
            append!(B, x)
        else
            C = vcat(B, vec(collect(x)))  # can you do this better?
            return _untyped_flatten!(C, A, state)
        end
        xit = iterate(A, state)
    end
    return B
end



#=

let
    tups = [Tuple(rand(10)) for _ in 1:100]
    println("vector of tuples")
    @btime stack($tups)
    @btime flatten($tups)
    @btime collect(Iterators.flatten($tups))

    println("generator of tuples")
    @btime stack(t for t in $tups)
    @btime flatten(t for t in $tups)
    @btime collect(Iterators.flatten(t for t in $tups))

    println("  ... extra to collet")
    @btime collect(t for t in $tups)

    println("vectors")
    vecs = [rand(100) for _ in 1:100]
    @btime stack($vecs)
    @btime flatten($vecs)
    @btime collect(Iterators.flatten($vecs))

    println("generator of vectors")
    @btime stack(v for v in $vecs)
    @btime flatten(v for v in $vecs)
    @btime collect(Iterators.flatten(v for v in $vecs))

    println("  ... extra to collet")
    @btime collect(t for t in $vecs)
end;

vector of tuples
  min 952.636 ns, mean 2.109 μs (1 allocation, 7.94 KiB)
  min 845.333 ns, mean 2.071 μs (1 allocation, 7.94 KiB)
  min 11.708 μs, mean 12.503 μs (1 allocation, 7.94 KiB)
generator of tuples
  min 945.800 ns, mean 2.055 μs (1 allocation, 7.94 KiB)
  min 1.608 μs, mean 2.613 μs (2 allocations, 7.94 KiB)   <--- could be better?
  min 16.292 μs, mean 18.288 μs (7 allocations, 21.92 KiB)
  ... extra to collet
  min 482.051 ns, mean 1.713 μs (1 allocation, 7.94 KiB)
vectors
  min 2.643 μs, mean 7.749 μs (2 allocations, 78.17 KiB)
  min 2.704 μs, mean 7.681 μs (2 allocations, 78.17 KiB)
  min 70.375 μs, mean 77.521 μs (9 allocations, 326.55 KiB)
generator of vectors
  min 2.588 μs, mean 7.277 μs (2 allocations, 78.17 KiB)
  min 4.738 μs, mean 9.588 μs (2 allocations, 78.25 KiB)   <--- could be better
  min 68.625 μs, mean 76.385 μs (10 allocations, 326.61 KiB)
  ... extra to collet
  min 192.396 ns, mean 217.751 ns (1 allocation, 896 bytes)  <--- worth it!


let
    tups = [Tuple(rand(10)) for _ in 1:100]
    println("unknown number of tuples")
    @btime stack(t for t in $tups if true)
    @btime flatten(t for t in $tups if true)
    @btime collect(Iterators.flatten(t for t in $tups if true))

    println("  ... extra to collet")
    @btime collect(t for t in $tups if true)

    vecs = [rand(100) for _ in 1:100]
    println("unknown number of vectors")
    @btime stack(v for v in $vecs if true)
    @btime flatten(v for v in $vecs if true)
    @btime collect(Iterators.flatten(v for v in $vecs if true))

    println("  ... extra to collet")
    @btime collect(v for v in $vecs if true)
end;

unknown number of tuples
  min 2.454 μs, mean 5.807 μs (6 allocations, 25.72 KiB)
  min 2.042 μs, mean 4.953 μs (6 allocations, 21.91 KiB)
  min 16.292 μs, mean 18.510 μs (7 allocations, 21.92 KiB)
  ... extra to collet
  min 1.450 μs, mean 3.878 μs (5 allocations, 17.78 KiB)
unknown number of vectors
  min 3.625 μs, mean 9.854 μs (7 allocations, 80.12 KiB)
  min 7.917 μs, mean 16.509 μs (7 allocations, 198.11 KiB)   <--- could be better?
  min 71.875 μs, mean 79.867 μs (10 allocations, 326.61 KiB)
  ... extra to collet
  min 1.042 μs, mean 1.158 μs (5 allocations, 1.95 KiB)  <--- worth it!

let
    nums = rand(128, 100)
    println("vector")
    @btime stack($nums)
    @btime flatten($nums)
    @btime copy(vec($nums))
end;

vector
  min 8.361 μs, mean 20.294 μs (2 allocations, 100.05 KiB)
  min 9.333 μs, mean 19.566 μs (2 allocations, 100.05 KiB)
  min 1.995 μs, mean 10.940 μs (4 allocations, 100.12 KiB)

let
    tups = [Tuple(rand(rand(1:20))) for _ in 1:100]
    println("tuples of varying length")
    @btime flatten($tups)
    @btime collect(Iterators.flatten($tups))

    vecs = [rand(1:200) for _ in 1:100]
    println("vectors of varying length")
    @btime flatten($vecs)
    @btime collect(Iterators.flatten($vecs))
end;

tuples of varying length
  min 14.167 μs, mean 15.792 μs (52 allocations, 9.11 KiB)
  min 8.833 μs, mean 11.346 μs (6 allocations, 21.86 KiB)
vectors of varying length
  min 105.169 ns, mean 125.837 ns (1 allocation, 896 bytes)
  min 106.701 ns, mean 127.880 ns (1 allocation, 896 bytes)

let
    tups = vcat([Tuple(rand(1:10, 10)) for _ in 1:50], [Tuple(rand(10)) for _ in 1:50])
    println("tuples of varying eltype")
    @btime stack($tups)
    @btime flatten($tups)
    @btime collect(Iterators.flatten($tups))

    println("vectors of varying eltype")
    vecs = vcat([rand(1:10, 100) for _ in 1:50], [rand(100) for _ in 1:50])
    @btime stack($vecs)
    @btime flatten($vecs)
    @btime collect(Iterators.flatten($vecs))
end;

tuples of varying eltype
  min 23.917 μs, mean 26.051 μs (49 allocations, 8.69 KiB)
  min 10.417 μs, mean 13.462 μs (7 allocations, 20.41 KiB)
  min 28.041 μs, mean 29.826 μs (501 allocations, 15.75 KiB)
vectors of varying eltype
  min 2.646 μs, mean 10.090 μs (2 allocations, 78.17 KiB)
  min 2.713 μs, mean 9.107 μs (2 allocations, 78.17 KiB)
  min 68.750 μs, mean 77.771 μs (9 allocations, 326.55 KiB)

=#


#=

julia> f(x, y) = reduce(vcat, map(current -> searchsorted(x, current), y));  # collected

julia> g(x, y) = mapreduce(current -> searchsorted(x, current), vcat, y);  # pairwise

julia> h(x, y) = flatten(current -> searchsorted(x, current), y);  # append!

julia> let
           Random.seed!(1)
           list = sort(rand(1:99, 10^6))
           needed = unique(sort(rand(1:99, 10)))
           @btime f($list, $needed)   # reduce(vcat, map(...
           # @btime g($list, $needed) # mapreduce(...
           @btime h($list, $needed)   # flatten(...
           @btime flatten([searchsorted($list, current) for current in $needed])
       end;
  min 17.958 μs, mean 74.229 μs (4 allocations, 789.84 KiB)
  min 19.917 μs, mean 57.096 μs (2 allocations, 807.81 KiB)  # here the guess is good
  min 19.542 μs, mean 72.105 μs (3 allocations, 789.70 KiB)

julia> let
           Random.seed!(42)
           list = sort(rand(1:99, 10^6))
           needed = unique(sort(rand(1:99, 10)))
           @btime f($list, $needed)   # reduce(vcat, map(...
           # @btime g($list, $needed) # mapreduce(...
           @btime h($list, $needed)   # flatten(...
           @btime flatten([searchsorted($list, current) for current in $needed])
       end;
  min 17.791 μs, mean 67.903 μs (4 allocations, 790.41 KiB)
  min 37.375 μs, mean 128.942 μs (3 allocations, 2.13 MiB)  # here the guess is not so good
  min 19.417 μs, mean 77.225 μs (3 allocations, 790.27 KiB)

=#