module LazyStack

export stack, rstack

#===== Tuples =====#

ndims(A) = Base.ndims(A)
ndims(::Tuple) = 1
ndims(::NamedTuple) = 1

axes(A) = Base.axes(A)
axes(A, d) = Base.axes(A, d)
if VERSION < v"1.1" # because, on Julia 1.0, axes((1,2)) === Base.OneTo(2)
    axes(t::Tuple) = tuple(Base.axes(t))
end
axes(nt::NamedTuple) = tuple(Base.OneTo(length(nt)))

size(A) = Base.size(A)
size(t::Tuple) = tuple(length(t))
size(t::NamedTuple) = tuple(length(t))

similar_vector(x::AbstractArray, n::Int) = similar(x, n::Int)
similar_vector(x::Tuple, n::Int) = Vector{eltype(x)}(undef, n::Int)
similar_vector(x::NamedTuple, n::Int) = Vector{eltype(x)}(undef, n::Int)

# eltype(x::Tuple) = Base.promote_type(x...) # to match choice below
# eltype(x::NamedTuple) = Base.promote_type(x...)

#===== Slices =====#

"""
    stack([A, B, C])
    stack(A, B, C)

Creates a very simple lazy `::Stacked` view of an array of arrays,
or a tuple of arrays, provided they have a consistent `eltype`.
The dimensions of the inner arrays come first:
```
julia> stack([1,2,3], [10,20,30])
3×2 LazyStack.Stacked{Int64,2,Tuple{Array{Int64,1},Array{Int64,1}}}:
 1  10
 2  20
 3  30

julia> vecs = [[1,2,3] .+ j*im for j=2:2:8];

julia> A = stack(vecs)
3×4 Stacked{Complex{Int64},2,Array{Array{Complex{Int64},1},1}}:
 1+2im  1+4im  1+6im  1+8im
 2+2im  2+4im  2+6im  2+8im
 3+2im  3+4im  3+6im  3+8im
```
A few `Base` functions will immediately look inside:
```
julia> eachcol(A) |> typeof
Array{Array{Complex{Int64},1},1}

julia> view(A, :,1)
3-element Array{Complex{Int64},1}:
 1 + 2im
 2 + 2im
 3 + 2im
```
"""
stack(xs::AbstractArray{IT,ON}) where {IT<:AbstractArray{T,IN}} where {T,IN,ON} =
    stack_slices(xs, Val(T), Val(IN+ON))
stack(xs::Tuple{Vararg{AbstractArray{T,IN}}}) where {T,IN} =
    stack_slices(xs, Val(T), Val(IN+1))
stack(xs::AbstractArray{T}...) where {T} = stack(xs)

function stack_slices(xs::AT, ::Val{T}, ::Val{N}) where {T,N,AT}
    length(xs) >= 1 || throw(ArgumentError("stacking an empty collection is not allowed"))
    storage_type(first(xs)) <: Union{Array, AbstractRange} || return stack_iter(xs)
    s = size(first(xs))
    for x in xs
        size(x) == s || throw(DimensionMismatch(
            "slices being stacked must share a common size. Expected $s, got $(size(x))"))
    end
    Stacked{T, N, AT}(xs)
end

struct Stacked{T,N,AT} <: AbstractArray{T,N}
    slices::AT
end

Base.size(x::Stacked) = (size(first(x.slices))..., size(x.slices)...)
Base.size(x::Stacked{T,N,<:Tuple}) where {T,N} = (size(first(x.slices))..., length(x.slices))

Base.axes(x::Stacked) = (axes(first(x.slices))..., axes(x.slices)...)

Base.parent(x::Stacked) = x.slices

outer_ndims(x::Stacked) = ndims(x.slices)
outer_ndims(x::Stacked{T,N,<:Tuple}) where {T,N} = 1

inner_ndims(x::Stacked) = ndims(x) - outer_ndims(x)

@inline function Base.getindex(x::Stacked{T,N}, inds::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(x, inds...)
    IN, ON = inner_ndims(x), outer_ndims(x)
    outer = @inbounds getindex(x.slices, ntuple(d -> inds[d+IN], ON)...)
    @inbounds getindex(outer, ntuple(d -> inds[d], IN)...)::T
end

Base.unaliascopy(x::Stacked) = stack(map(Base.unaliascopy, x.slices))

if VERSION >= v"1.1"
    Base.eachcol(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}) where {T} = x.slices
end

Base.collect(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}) where {T} = reduce(hcat, x.slices)

"""
    LazyStack.ensure_dense(A)

This `collect`s the result of `stack` if it is a `Stacked` container,
but does nothing to other arrays, such as those produced by `stack_iter`.
"""
ensure_dense(x::Stacked) = collect(x)
ensure_dense(x::AbstractArray) = x

Base.view(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}, ::Colon, i::Int) where {T} = x.slices[i]

function Base.push!(x::Stacked{T,N,<:AbstractVector}, y::AbstractArray) where {T,N}
    s = size(first(x.slices))
    isempty(y) || size(y) == s || throw(DimensionMismatch(
            "slices being stacked must share a common size. Expected $s, got $(size(y))"))
    push!(x.slices, y)
    x
end

function Base.showarg(io::IO, x::Stacked, toplevel)
    print(io, "stack(")
    Base.showarg(io, parent(x), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(x))
end

#===== Iteration =====#

ITERS = [:Flatten, :Drop, :Filter]
for iter in ITERS
    @eval ndims(::Iterators.$iter) = 1
end
ndims(gen::Base.Generator) = ndims(gen.iter)
ndims(zed::Iterators.Zip) = maximum(ndims, zed.is)
if VERSION < v"1.1"
    ndims(zed::Iterators.Zip2) = max(ndims(zed.a), ndims(zed.b))
end

similar_vector(x, n::Int) = throw(ArgumentError())

"""
    stack(::Generator)
    stack(::Array{T}, ::Array{S}, ...)

This constructs a new array, calling `stack_iter`.
Can handle inconsistent eltypes, but not inconsistent sizes.

```
julia> stack([i i; i 10i] for i in 1:2:3)
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1   1
 1  10

[:, :, 2] =
 3   3
 3  30

julia> stack(c<9 ? [c,2c] : [9.99, 10] for c in 1:10)
2×10 Array{Real,2}:
 1  2  3  4   5   6   7   8   9.99   9.99
 2  4  6  8  10  12  14  16  10.0   10.0

julia> stack(1:3, ones(3), zeros(3) .+ im)
3×3 Array{Number,2}:
 1  1.0  0.0+1.0im
 2  1.0  0.0+1.0im
 3  1.0  0.0+1.0im
```
"""
stack(gen::Base.Generator) = stack_iter(gen)
for iter in ITERS
    @eval stack(gen::Iterators.$iter) = stack_iter(gen)
end
stack(arr::AbstractArray{Any}) = stack_iter(arr) # e.g. from arr=[]; push!(arr, rand(3)); ...

stack(arr::AbstractArray{<:AbstractArray}) = stack_iter(arr)
stack(arr::AbstractArray{<:Tuple}) = stack_iter(arr)
stack(arr::AbstractArray{<:NamedTuple}) = stack_iter(arr)

stack(tup::AbstractArray...) = stack_iter(tup)
stack(tup::Tuple...) = stack_iter(tup)
stack(tup::NamedTuple...) = stack_iter(tup)
# stack(tup::Tuple{Vararg{AbstractArray}}) = stack_iter(tup)

function stack_iter(itr)
    if itr isa Tuple || itr isa Base.Generator{<:Tuple} || ndims(itr) == 1
        outsize = tuple(:)
    else
        Base.haslength(itr) || return stack_iter(collect(itr))
        outsize = size(itr)
    end

    w, val = vstack_plus(itr)

    z = reshape(w, size(val)..., outsize...)

    rewrap_like(z, val)
end

vstack(itr) = first(vstack_plus(itr))

function vstack_plus(itr)
    zed = iterate(itr)
    zed === nothing && throw(ArgumentError("stacking an empty collection is not allowed"))
    val, state = zed
    val isa Union{AbstractArray, Tuple, NamedTuple} || return collect(itr), val

    s = size(val)
    n = Base.haslength(itr) ? prod(s)*length(itr) : nothing

    v = similar_vector(no_wraps(val), something(n, prod(s)))
    copyto!(v, 1, no_wraps(val), 1, prod(s))

    w = stack_rest(v, 0, n, s, itr, state)#::Vector
    w, val
end

function stack_rest(v, i, n, s, itr, state)
    while true
        zed = iterate(itr, state)
        zed === nothing && return v
        val, state = zed
        s == size(val) || throw(DimensionMismatch(
            "slices being stacked must share a common size. Expected $s, got $(size(val))"))

        i += 1
        if eltype(val) <: eltype(v)
            if n isa Int
                copyto!(v, i*prod(s)+1, no_wraps(val), 1, prod(s))
            else
                append!(v, vec(no_wraps(val)))
            end
        else

            T′ = Base.promote_typejoin(eltype(v), eltype(val))
            # T′ = Base.promote_type(eltype(v), eltype(val)) # which do I want?
            v′ = similar(v, T′)
            copyto!(v′, v)

            if n isa Int
                copyto!(v′, i*prod(s)+1, no_wraps(val), 1, prod(s))
            else
                append!(v′, vec(no_wraps(val)))
            end

            return stack_rest(v′, i, n, s, itr, state)
        end

    end
end

"""
    stack(fun, iters...)
    stack(eachcol(A), eachslice(B, dims=3)) do a, b
        f(a,b)
    end

If the first argument is a function, then this is mapped over the other argumenst.
The result should be `== stack(map(fun, iters...))`, but done using `stack_iter`
to return a dense `Array` instead of a `Stacked` object.
"""
stack(fun::Function, iter) = stack(fun(arg) for arg in iter)
function stack(fun::Function, iters...)
    if all(Base.haslength, iters)
        sz = size(first(iters))
        all(a -> size(a)==sz, iters) || throw(DimensionMismatch(
            "sizes of all argumens must match, in stack(f, A, B, C). " *
            "This is slightly stricter than map(f, A, B, C), for now."))
    end
    stack(fun(args...) for args in zip(iters...))
end

#===== Offset =====#

using OffsetArrays

no_wraps(a) = a
no_wraps(a::OffsetArray) = parent(a)

rewrap_like(A, a) = A
function rewrap_like(A, a::OffsetArray)
    B = rewrap_like(A, parent(a))
    OffsetArray(B, axes(a)..., axes(A, ndims(A)))
end

#===== NamedDims =====#

using NamedDims

ensure_named(a::AbstractArray, L::Tuple) = NamedDimsArray(a, L)
ensure_named(a::NamedDimsArray, L::Tuple) = refine_names(a, L)

# array of arrays
stack(xs::NamedDimsArray{<:Any,<:AbstractArray}) =
    ensure_named(stack(parent(xs)), getnames(xs))
stack(x::AT) where {AT <: AbstractArray{<:NamedDimsArray{L,T,IN},ON}} where {T,IN,ON,L} =
    ensure_named(Stacked{T, IN+ON, AT}(x), getnames(x))

getnames(xs::AbstractArray{<:AbstractArray}) =
    (dimnames(eltype(xs))..., dimnames(xs)...)

# tuple of arrays
stack(x::AT) where {AT <: Tuple{Vararg{NamedDimsArray{L,T,IN}}}} where {T,IN,L} =
    ensure_named(stack(map(parent, x)), getnames(x))

getnames(xs::Tuple{Vararg{<:NamedDimsArray}}) =
    (dimnames(first(xs))..., :_)

function rewrap_like(A, a::NamedDimsArray{L}) where {L}
    B = rewrap_like(A, parent(a))
    ensure_named(B, (L..., ntuple(_ -> :_, ndims(A) - ndims(a))...))
end

no_wraps(a::NamedDimsArray) = no_wraps(parent(a))

"""
    stack(name, things...)

If you give one `name::Symbol` before the pieces to be stacked,
this will be the name of the last dimension of the resulting `NamedDimsArray`.
(Names attached to slices, and to containers, should also be preserved.)
"""
function LazyStack.stack(s::Symbol, args...)
    data = stack(args...)
    name_last = ntuple(d -> d==ndims(data) ? s : :_, ndims(data))
    ensure_named(data, name_last)
end

#===== Gradients =====#

using ChainRulesCore: ChainRulesCore, rrule, NoTangent

function ChainRulesCore.rrule(::typeof(stack), vec::AbstractArray{<:AbstractArray{<:Any,IN}}) where {IN}
    stack(vec), Δ -> (NoTangent(), [view(Δ, ntuple(_->(:),IN)..., Tuple(I)...) for I in eachindex(vec)],)
end

function ChainRulesCore.rrule(::typeof(stack), tup::Tuple{Vararg{<:AbstractArray{<:Any,IN}}}) where {IN}
    stack(tup), Δ -> (NoTangent(), ntuple(i -> view(Δ, ntuple(_->(:),IN)..., i), length(tup)),)
end

function ChainRulesCore.rrule(::typeof(stack), gen::Base.Generator)
    stack(gen), Δ -> error("not yet!")
end

#===== CuArrays =====#
# Send these to stack_iter, by testing  storage_type(first(xs)) <: Array

function storage_type(x::AbstractArray)
    p = parent(x)
    typeof(x) === typeof(p) ? typeof(x) : storage_type(p)
end

#===== Ragged =====#

"""
    rstack(arrays; fill=0)

Ragged `stack`, which allows slices of varying size, and fills the gaps with zero
or the given `fill`. Always returns an `Array`.

```
julia> rstack(1:n for n in 1:5)
5×5 Array{Int64,2}:
 1  1  1  1  1
 0  2  2  2  2
 0  0  3  3  3
 0  0  0  4  4
 0  0  0  0  5

julia> rstack([[1,2,3], [10,20.0]], fill=missing)
3×2 Array{Union{Missing, Float64},2}:
 1.0  10.0
 2.0  20.0
 3.0    missing

julia> using OffsetArrays

julia> rstack(1:3, OffsetArray([2.0,2.1,2.2], -1), OffsetArray([3.2,3.3,3.4], +1))
5×3 OffsetArray(::Array{Real,2}, 0:4, 1:3) with eltype Real with indices 0:4×1:3:
 0  2.0  0
 1  2.1  0
 2  2.2  3.2
 3  0    3.3
 0  0    3.4
```

"""
rstack(x::AbstractArray, ys::AbstractArray...; kw...) = rstack((x, ys...); kw...)
rstack(g::Base.Generator; kw...) = rstack(collect(g); kw...)
rstack(f::Function, ABC...; kw...) = rstack(map(f, ABC...); kw...)
rstack(list::AbstractArray{<:AbstractArray}; fill=zero(eltype(first(list)))) = rstack_iter(list; fill=fill)
rstack(list::Tuple{Vararg{<:AbstractArray}}; fill=zero(eltype(first(list)))) = rstack_iter(list; fill=fill)

function rstack_iter(list; fill)
    T = mapreduce(eltype, Base.promote_typejoin, list, init=typeof(fill))
    # T = mapreduce(eltype, Base.promote_type, list, init=typeof(fill))
    N = maximum(ndims, list)
    ax = ntuple(N) do d
        hi = maximum(x -> last(axes(x,d)), list)
        if all(x -> axes(x,d) isa Base.OneTo, list)
            Base.OneTo(hi)
        else
            lo = minimum(x -> first(axes(x,d)), list)
            lo:hi
        end
    end
    arr = Array{T}(undef, map(length, ax)..., size(list)...)
    fill!(arr, fill)
    out = if ax isa Tuple{Vararg{Base.OneTo}}
        arr
    else
        OffsetArray(arr, (ax..., axes(list)...))
    end
    z = rstack_copyto!(out, list, Val(N))

    rewrap_names(z, first(list)) # now I want to separate names & offsets again!
end

function rstack_copyto!(out, list, ::Val{N}) where {N}
    for i in tupleindices(list)
        item = list[i...]
        o = ntuple(_->1, N - ndims(item))
        out[CartesianIndices(axes(item)), o..., i...] .= item

        # https://github.com/JuliaArrays/OffsetArrays.jl/issues/100
        # view(out, axes(item)..., o..., i...) .= item

        # for I in CartesianIndices(item)
        #     out[Tuple(I)..., o..., i...] = item[I]
        # end
    end
    out
end

tupleindices(t::Tuple) = ((i,) for i in 1:length(t))
tupleindices(A::AbstractArray) = (Tuple(I) for I in CartesianIndices(A))

rewrap_names(A, a) = A
function rewrap_names(A, a::NamedDimsArray{L}) where {L}
    B = rewrap_names(A, parent(a))
    ensure_named(B, (L..., ntuple(_ -> :_, ndims(A) - ndims(a))...))
end
function rstack(s::Symbol, args...; kw...)
    data = rstack(args...; kw...)
    name_last = ntuple(d -> d==ndims(data) ? s : :_, ndims(data))
    ensure_named(data, name_last)
end

end # module
