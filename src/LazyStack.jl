module LazyStack

export lazystack

include("ragged.jl")
export raggedstack

using Compat

@deprecate stack lazystack false  # don't export it 
@deprecate stack_iter Compat.stack false  # don't export it 

@deprecate rstack raggedstack

include("flatten.jl")  # probably not permanent, just prototyping here!
include("concatenate.jl")

#===== Tuples =====#

_ndims(A) = Base.ndims(A)
_ndims(::Tuple) = 1
_ndims(::NamedTuple) = 1

_axes(A) = Base.axes(A)
_axes(A, d) = Base.axes(A, d)
_axes(nt::NamedTuple) = tuple(Base.OneTo(length(nt)))

_size(A) = Base.size(A)
_size(t::Tuple) = tuple(length(t))
_size(t::NamedTuple) = tuple(length(t))


#===== Slices =====#

"""
    lazystack([A, B, C])
    lazystack(A, B, C) == lazystack((A, B, C))

Creates a very simple lazy `::Stacked` view of an array of arrays,
or a tuple of arrays, provided they have a consistent `eltype`.
The dimensions of the inner arrays come first.

# Examples
```
julia> lazystack([1,2,3], [10,20,30])
3×2 lazystack(::Tuple{Vector{Int64}, Vector{Int64}}) with eltype Int64:
 1  10
 2  20
 3  30

julia> vecs = [[1,2,3] .+ j*im for j=2:2:8];

julia> A = lazystack(vecs)
3×4 lazystack(::Vector{Vector{Complex{Int64}}}) with eltype Complex{Int64}:
 1+2im  1+4im  1+6im  1+8im
 2+2im  2+4im  2+6im  2+8im
 3+2im  3+4im  3+6im  3+8im
```

When the slices aren't backed by `Array` (or a few other `Base` types)
then it reverts to eager `Base.stack`:

```
julia> using JLArrays  # fake GPUArray

julia> lazystack(jl([1, 2, 3f0]), jl([4, 5, 6f0]))
3×2 JLArray{Float32, 2}:
 1.0  4.0
 2.0  5.0
 3.0  6.0
```

A few `Base` functions will immediately look inside the container:

```
julia> eachcol(A) |> typeof
Vector{Vector{Complex{Int64}}} (alias for Array{Array{Complex{Int64}, 1}, 1})

julia> view(A, :,1)
3-element Vector{Complex{Int64}}:
 1 + 2im
 2 + 2im
 3 + 2im
```
"""
lazystack(xs::AbstractArray{IT,ON}) where {IT<:AbstractArray{T,IN}} where {T,IN,ON} =
    stack_slices(xs, Val(T), Val(IN+ON))
lazystack(xs::Tuple{Vararg{AbstractArray{T,IN}}}) where {T,IN} =
    stack_slices(xs, Val(T), Val(IN+1))

# This lets `lazystack([1,2], [3,4])` act like hcat, slightly dodgy?
lazystack(xs::AbstractArray{T}...) where {T} = lazystack(xs)
# But `lazystack([1,2])` should not do this, violates equivalence with `Base.stack`.
lazystack(xs::AbstractArray{<:Number}) = xs

function stack_slices(xs::AT, ::Val{T}, ::Val{N}) where {T,N,AT}
    length(xs) >= 1 || throw(ArgumentError("stacking an empty collection is not allowed"))
    storage_type(first(xs)) <: Union{Array, AbstractRange} || return Compat.stack(xs)
    ax = _axes(first(xs))
    for x in xs
        _axes(x) == ax || throw(DimensionMismatch(
            "slices being stacked must share a common size. Expected $ax, got $(_axes(x))"))
    end
    Stacked{T, N, AT}(xs)
end

struct Stacked{T,N,AT} <: AbstractArray{T,N}
    slices::AT
end

Base.size(x::Stacked) = (_size(first(x.slices))..., _size(x.slices)...)
Base.size(x::Stacked{T,N,<:Tuple}) where {T,N} = (_size(first(x.slices))..., length(x.slices))

Base.axes(x::Stacked) = (_axes(first(x.slices))..., _axes(x.slices)...)

Base.parent(x::Stacked) = x.slices

outer_ndims(x::Stacked) = _ndims(x.slices)
outer_ndims(x::Stacked{T,N,<:Tuple}) where {T,N} = 1

inner_ndims(x::Stacked) = _ndims(x) - outer_ndims(x)

@inline function Base.getindex(x::Stacked{T,N}, inds::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(x, inds...)
    IN, ON = inner_ndims(x), outer_ndims(x)
    outer = @inbounds getindex(x.slices, ntuple(d -> inds[d+IN], ON)...)
    @inbounds getindex(outer, ntuple(d -> inds[d], IN)...)::T
end

Base.unaliascopy(x::Stacked) = stack(map(Base.unaliascopy, x.slices))

Base.eachcol(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}) where {T} = x.slices

Base.collect(x::Stacked) = Compat.stack(x.slices)

"""
    LazyStack.ensure_dense(A)

This `collect`s the result of `stack` if it is a `Stacked` container,
but does nothing to other arrays, such as those produced by `stack_iter`.
"""
ensure_dense(x::Stacked) = collect(x)
ensure_dense(x::AbstractArray) = x

Base.view(x::Stacked{T,2,<:AbstractArray{<:AbstractArray{T,1}}}, ::Colon, i::Int) where {T} = x.slices[i]

function Base.push!(x::Stacked{T,N,<:AbstractVector}, y::AbstractArray) where {T,N}
    s = _size(first(x.slices))
    isempty(y) || _size(y) == s || throw(DimensionMismatch(
            "slices being stacked must share a common size. Expected $s, got $(_size(y))"))
    push!(x.slices, y)
    x
end

function Base.showarg(io::IO, x::Stacked, toplevel)
    print(io, "lazystack(")
    Base.showarg(io, parent(x), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(x))
end

#===== Iteration =====#

"""
    lazystack(::Generator)
    lazystack(::Array{T}, ::Array{S}, ...)

These used to call this package's own `stack_iter`,
but now they just call `Base.stack`.
"""
lazystack(gen::Base.Generator) = Compat.stack(gen)
for iter in [:Flatten, :Drop, :Filter]
    @eval lazystack(gen::Iterators.$iter) = Compat.stack(gen)
end
lazystack(arr::AbstractArray{Any}) = Compat.stack(arr) # e.g. from arr=[]; push!(arr, rand(3)); ...

lazystack(arr::AbstractArray{<:AbstractArray}) = Compat.stack(arr)
lazystack(arr::AbstractArray{<:Tuple}) = Compat.stack(arr)
lazystack(arr::AbstractArray{<:NamedTuple}) = Compat.stack(arr)

lazystack(tup::AbstractArray...) = Compat.stack(tup)
lazystack(tup::Tuple...) = Compat.stack(tup)
lazystack(tup::NamedTuple...) = Compat.stack(tup)
# lazystack(tup::Tuple{Vararg{AbstractArray}}) = Compat.stack(tup)


"""
    lazystack(fun, iters...)
    lazystack(eachcol(A), eachslice(B, dims=3)) do a, b
        f(a,b)
    end

If the first argument is a function, then this is mapped over the other argumenst.
Always uses `Base.stack`.
"""
lazystack(fun::Function, iter) = Compat.stack(fun, iter)
lazystack(fun::Function, iters...) = Compat.stack(fun, iters...)


#===== Gradients =====#

using ChainRulesCore: ChainRulesCore, rrule, NoTangent

function ChainRulesCore.rrule(::typeof(lazystack), vec::AbstractArray{<:AbstractArray{<:Any,IN}}) where {IN}
    lazystack(vec), Δ -> (NoTangent(), [view(Δ, ntuple(_->(:),IN)..., Tuple(I)...) for I in eachindex(vec)],)
end

function ChainRulesCore.rrule(::typeof(lazystack), tup::Tuple{Vararg{AbstractArray{<:Any,IN}}}) where {IN}
    lazystack(tup), Δ -> (NoTangent(), ntuple(i -> view(Δ, ntuple(_->(:),IN)..., i), length(tup)),)
end

function ChainRulesCore.rrule(::typeof(lazystack), gen::Base.Generator)
    lazystack(gen), Δ -> error("not yet!")
end

#===== CuArrays =====#
# Send these to stack_iter, by testing  storage_type(first(xs)) <: Array

function storage_type(x::AbstractArray)
    p = parent(x)
    typeof(x) === typeof(p) ? typeof(x) : storage_type(p)
end

end # module
