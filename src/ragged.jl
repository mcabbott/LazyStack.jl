"""
    raggedstack(arrays; fill=0)

Ragged `stack`, which allows slices of varying _size, and fills the gaps with zero
or the given `fill`. Always returns an `Array`.

```
julia> raggedstack(1:n for n in 1:5)
5×5 Array{Int64,2}:
 1  1  1  1  1
 0  2  2  2  2
 0  0  3  3  3
 0  0  0  4  4
 0  0  0  0  5

julia> raggedstack([[1,2,3], [10,20.0]], fill=missing)
3×2 Array{Union{Missing, Float64},2}:
 1.0  10.0
 2.0  20.0
 3.0    missing

julia> using OffsetArrays

julia> raggedstack(1:3, OffsetArray([2.0,2.1,2.2], -1), OffsetArray([3.2,3.3,3.4], +1))
5×3 OffsetArray(::Array{Real,2}, 0:4, 1:3) with eltype Real with indices 0:4×1:3:
 0  2.0  0
 1  2.1  0
 2  2.2  3.2
 3  0    3.3
 0  0    3.4
```

"""
raggedstack(x::AbstractArray, ys::AbstractArray...; kw...) = raggedstack((x, ys...); kw...)
raggedstack(g::Base.Generator; kw...) = raggedstack(collect(g); kw...)
raggedstack(f::Function, ABC...; kw...) = raggedstack(map(f, ABC...); kw...)
raggedstack(list::AbstractArray{<:AbstractArray}; fill=zero(eltype(first(list)))) = raggedstack_iter(list; fill)
raggedstack(list::Tuple{Vararg{AbstractArray}}; fill=zero(eltype(first(list)))) = raggedstack_iter(list; fill)

function raggedstack_iter(list; fill)
    T = mapreduce(eltype, Base.promote_typejoin, list, init=typeof(fill))
    # T = mapreduce(eltype, Base.promote_type, list, init=typeof(fill))
    N = maximum(_ndims, list)
    ax = ntuple(N) do d
        hi = maximum(x -> last(_axes(x,d)), list)
        if all(x -> _axes(x,d) isa Base.OneTo, list)
            Base.OneTo(hi)
        else
            lo = minimum(x -> first(_axes(x,d)), list)
            lo:hi
        end
    end
    out = similar(1:0, T, ax..., _axes(list)...)
    fill!(out, fill)
    raggedstack_copyto!(out, list, Val(N))
end

function raggedstack_copyto!(out, list, ::Val{N}) where {N}
    for i in tupleindices(list)
        item = list[i...]
        o = ntuple(_->1, N - _ndims(item))
        out[CartesianIndices(_axes(item)), o..., i...] .= item

        # https://github.com/JuliaArrays/OffsetArrays.jl/issues/100
        # view(out, _axes(item)..., o..., i...) .= item

        # for I in CartesianIndices(item)
        #     out[Tuple(I)..., o..., i...] = item[I]
        # end
    end
    out
end

tupleindices(t::Tuple) = ((i,) for i in 1:length(t))
tupleindices(A::AbstractArray) = (Tuple(I) for I in CartesianIndices(A))

# rewrap_names(A, a) = A
# function rewrap_names(A, a::NamedDimsArray{L}) where {L}
#     B = rewrap_names(A, parent(a))
#     ensure_named(B, (L..., ntuple(_ -> :_, _ndims(A) - _ndims(a))...))
# end
# function raggedstack(s::Symbol, args...; kw...)
#     data = raggedstack(args...; kw...)
#     name_last = ntuple(d -> d==_ndims(data) ? s : :_, _ndims(data))
#     ensure_named(data, name_last)
# end
