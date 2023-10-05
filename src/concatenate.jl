
export concatenate, concatenate!
export concatenate1, concatenate2, concatenate2!, concatenate3, concatenate3!

using Base: IteratorSize, HasLength, HasShape

const LazyString = isdefined(Base, :LazyString) ? Base.LazyString : string

# From here, surely this can be done much better:
# https://github.com/JuliaLang/julia/pull/46003#issuecomment-1181228513
concatenate1(a::AbstractArray{<:AbstractArray}) = Base.hvncat(size(a), false, a...)

#####
##### Take 2, aiming at @nloops
#####

function concatenate2(A::AbstractArray)
# function concatenate2(A::AbstractArray{<:AbstractArray})
    isempty(A) && throw(ArgumentError("concatenate over an empty collection is not allowed"))
    N = max(ndims(A), maximum(ndims, A))
    T = mapreduce(eltype, promote_type, A)
    outsize = ntuple(d -> sum(size(a, d) for a in view(A, ntuple(i -> i==d ? (:) : 1, N)...)), N)
    B = similar(first(A), T, outsize)
    concatenate2!(B, A)
end

function concatenate2(A::Tuple)
    isempty(A) && throw(ArgumentError("concatenate over an empty collection is not allowed"))
    N = max(1, maximum(ndims, A))
    T = mapreduce(eltype, promote_type, A)
    outsize = ntuple(d -> d == 1 ? sum(x -> size(x,1), A) : size(A[1], d), N)
    B = similar(first(A), T, outsize)
    concatenate2!(B, A)
end

function concatenate2!(B::AbstractArray, A::Union{Tuple, AbstractVector})
# function concatenate2!(B::AbstractArray, A::AbstractVector{<:AbstractArray})
    Base.require_one_based_indexing(B)
    rest = ntuple(_ -> (:), ndims(B)-1)

    off1 = 0
    for a1 in axes(A)[1]
        x1 = A[a1]
        b1 = UnitRange(off1+1, off1+=size(x1,1))

        B[b1, rest...] = x1
    end
    B
end

function concatenate2!(B::AbstractArray, A::AbstractMatrix{<:AbstractArray})
    Base.require_one_based_indexing(B)
    rest = ntuple(_ -> (:), ndims(B)-ndims(A))

    off2 = 0
    for a2 in axes(A,2)
        x2 = A[begin, a2]
        b2 = UnitRange(off2+1, off2+=size(x2,2))

        off1 = 0
        for a1 in axes(A,1)
            x1 = A[a1, a2]
            b1 = UnitRange(off1+1, off1+=size(x1,1))

            B[b1, b2, rest...] = x1
            # copyto!(view(B, b1, b2, rest...), x1)  # slower
        end
    end
    B
end


#=

selectone(A::AbstractArray, ::Val{d}, i::Int) where d = A[ntuple(k -> k==d ? i : firstindex(A,k))]

using Base.Cartesian

@nexprs $N d -> out_d = 0  # only really need the Nth

@nloops $N a A d -> begin  # pre:
    x_d = selectone(A, Val(d), a_d)
    b_d = UnitRange(off_d+1, off_d+=size(x_d,d))
    off_(d+1) = 0  # this doesn't work
    $(Symbol(:off_, d+1)) = 0
end begin  # body:
    (@nref 2 B b) = @nref 3 A i
end

=#


#####
##### Take 3, recursion
#####

function concatenate3(A::Union{Tuple, AbstractArray})
    isempty(A) && throw(ArgumentError("concatenate over an empty collection is not allowed"))
    x = first(A)
    _concatenate_arraycheck(x)
    T = mapreduce(eltype, promote_type, A)
    B = similar(x, T, _concatenate_size(A))
    concatenate3!(B, A)
end
concatenate3(A::NamedTuple) = concatenate3(Tuple(A))
# For iterators, we need all the sizes to allocate B, so must collect?
concatenate3(A) = concatenate3(collect(A)::AbstractArray)

# Could be lazier for a small class of iterators: known 1D, known to make vectors
# concatenate3(A) = _concatenate3(A, IteratorSize(A), Base.@default_eltype A)
# _concatenate3(A, ::Unon{HasLength, HasShape{1}}, ::Type{<:AbstractVector}) = flatten(A)
# _concatenate3(A, ::IteratorSize, ::Type) = concatenate3(collect(A))
# But hard to find cases where this seems worth it:
#=

julia> using LazyStack, StaticArrays

julia> @btime concatenate($(SA[i,i^2] for i in 1:100 if iseven(i)));
  min 823.529 ns, mean 1.361 μs (6 allocations, 4.61 KiB)

julia> @btime flatten($(SA[i,i^2] for i in 1:100 if iseven(i)));
  min 725.801 ns, mean 770.371 ns (4 allocations, 1.94 KiB)

=#

function _concatenate_size(A::AbstractArray)
    N = max(ndims(A), maximum(ndims, A))
    ntuple(d -> sum(size(a, d) for a in view(A, ntuple(i -> i==d ? (:) : firstindex(A,i), N)...)), N)
    # ntuple(N) do d
    #     sum(axes(A,d)) do k
    #         x = A[ntuple(n -> n==d ? k : firstindex(A,n), N)...]
    #         _concatenate_arraycheck(x)
    #         size(x,d)
    #     end
    # end
end
function _concatenate_size(A::Tuple)
    N = max(1, maximum(ndims, A))
    ntuple(d -> d==1 ? sum(x -> size(x,1), A) : size(A[1], d), N)
end

_concatenate_arraycheck(x::AbstractArray) = nothing
_concatenate_arraycheck(x) = throw(ArgumentError(LazyString("concatenate only works on arrays, got ", typeof(x))))

function concatenate3!(B::AbstractArray, A::Union{Tuple, AbstractVector})
    Base.require_one_based_indexing(B)
    post = ntuple(Returns(:), ndims(B) - 1)
    postax = Base.tail(axes(B))
    off = 0
    for x in A
        _concatenate_arraycheck(x)
        is = UnitRange(off + 1, off += size(x,1)::Int)
        # B[is, post...] = x
        # B[is, post...] .= x
        # @inbounds copyto!(view(B, is, post...), x)

        if ndims(x) == ndims(B)
            copyto!(B, CartesianIndices((is, postax...)), x, CartesianIndices(x))  # very quick when possible!
        else
            vB = view(B, is, post...)
            length(vB) == length(x) || throw(DimensionMismatch("wrong width..."))
            copyto!(vB, x)
        end

        # copyto!(B, CartesianIndices((is, postax...)[1:ndims(x)]), x, CartesianIndices(x))  # not a solution

        # xax = (axes(x)..., ntuple(_ -> Base.OneTo(1), ndims(B) - ndims(x))...)
        # copyto!(B, CartesianIndices((is, postax...)), x, CartesianIndices(xax))  # does not work
    end
    off == size(B,1) || throw(DimensionMismatch(LazyString(
        "concatenate expected this column to have ", size(B,1), " rows, but got only ", off)))
    B
end

# An attempt to go faster in the easy case, could be a branch of the above.
#=
function concatenate3!(B::AbstractVector, A::Union{Tuple, AbstractVector})
    Base.require_one_based_indexing(B)
    off = 0
    for x in A
        _concatenate_arraycheck(x)

        copyto!(B, off+1, x, firstindex(x), length(x))  # this is 15% slower than B[is] = x, @inbounds no help
        off += length(x)

        # is = UnitRange(off + 1, off += size(x,1)::Int)
        # copyto!(B, is, 1:1, x, eachindex(x), 1:1)  # undocumented method I found?
        # B[is] = x
    end
    off == size(B,1) || throw(DimensionMismatch(LazyString(
        "concatenate expected this column to have ", size(B,1), " rows, but got only ", off)))
    B
end
=#
# Or just call the existing code?
function concatenate3!(B::AbstractVector, A::Base.AbstractVecOrTuple{AbstractVector})
    # function _typed_vcat!(a::AbstractVector{T}, V::AbstractVecOrTuple{AbstractVector}) where T
    sum(length, A) == length(B) || throw(DimensionMismatch(LazyString(
        "concatenate expected this column to have ", length(B), " rows, but got ", sum(length, A) )))
    Base._typed_vcat!(B, A)
end

function concatenate3!(B::AbstractArray, A::AbstractArray)
    Base.require_one_based_indexing(B)
    pre = ntuple(Returns(:), ndims(A) - 1)
    post = ntuple(Returns(:), ndims(B) - ndims(A))
    off = 0
    for Aslice in eachslice(A, dims = ndims(A))
        x = first(Aslice)
        is = UnitRange(off + 1, off += size(x, ndims(A))::Int)
        concatenate3!(view(B, pre..., is, post...), Aslice)
    end
    B
end



#=

concatenate1([1:3, 4:5]) == 1:5
concatenate2([1:3, 4:5]) == 1:5

@code_warntype concatenate2([1:3, 4:5])
@code_warntype concatenate2([[1 2], [3 4]])

@code_warntype concatenate3([1:3, 4:5])
@code_warntype concatenate3([[1 2], [3 4]])


##### Test cases -- vcat-like


julia> let vecs = [rand(100) for _ in 1:100]
           a = @btime concatenate2($vecs)  # pretty good!
           b = @btime concatenate3($vecs)  # best copyto! version, sans _typed_vcat
           c = @btime reduce(vcat, $vecs)  # target
           # @btime concatenate1($vecs)  # hvncat, 10x slower
           a == b == c
       end
  2.477 μs (2 allocations: 78.17 KiB)
  2.889 μs (2 allocations: 78.17 KiB)
  2.583 μs (3 allocations: 79.05 KiB)

julia> let vm = [rand(10, 10) for _ in 1:100]
           a = @btime concatenate2($vm)
           b = @btime concatenate3($vm) 
           c = @btime reduce(vcat, $vm)  # target -- beaten!
           @btime stack($vm)   # better access pattern, easier?
           a == b == c
       end
  13.125 μs (4 allocations: 78.27 KiB)
  6.608 μs (4 allocations: 78.27 KiB)
  13.000 μs (2 allocations: 78.17 KiB)
  3.203 μs (2 allocations: 78.19 KiB)


##### Test cases -- hcat-like


julia> let rm = [rand(10, 10) for _ in 1:1, _ in 1:100]
           a = @btime concatenate2($rm)
           b = @btime concatenate3($rm)
           c = @btime reduce(hcat, $(vec(rm)))  # target -- very hard to match
           d = @btime stack($(vec(rm)))         # here the same access pattern, linear copyto!
           # @btime concatenate3($(vec(adjoint.(rm))'))  # hack? 7.698 μs (2 allocations: 78.17 KiB)
           a == b == c == reshape(d, 10, :)
       end
  14.542 μs (9 allocations: 78.36 KiB)
  8.778 μs (9 allocations: 78.36 KiB)
  3.172 μs (2 allocations: 78.17 KiB)
  2.896 μs (2 allocations: 78.19 KiB)
true

julia> let rv = [rand(100) for _ in 1:1, _ in 1:100]
          a = @btime concatenate2($rv)
          b = @btime concatenate3($rv)
          c = @btime reduce(hcat, $(vec(rv)))  # target -- far off!
          d = @btime stack($(vec(rv)))
          a == b == c == d
       end
  12.625 μs (8 allocations: 78.34 KiB)
  17.458 μs (8 allocations: 78.34 KiB)
  2.781 μs (2 allocations: 78.17 KiB)
  2.662 μs (2 allocations: 78.17 KiB)
true


##### Test cases -- hvcat instead?


julia> let mats = [rand(10,10) for _ in 1:10, _ in 1:10]
           @btime concatenate2($mats)
           @btime concatenate3($mats)    # slower with .=, fast with Cartesian
           @btime stack($mats; dims=2)   # maybe a comparably hard access pattern?
           # @btime concatenate1($mats)  # hvncat, 4x slower
           # @btime hvcat(10, $mats...)  # slower
       end;
  14.333 μs (8 allocations: 78.34 KiB)
  8.528 μs (8 allocations: 78.34 KiB)
  14.291 μs (2 allocations: 78.19 KiB)

julia> let mv = [rand(100) for _ in 1:10, _ in 1:10] 
       # let mv = [rand(100) for _ in 1:100, _ in 1:1]  # similar results
           @btime concatenate2($mv)
           @btime concatenate3($mv)  # faster with .=
           @btime stack($mv)  # the same access pattern, better exploited
       end;
  12.583 μs (8 allocations: 78.34 KiB)
  16.666 μs (8 allocations: 78.34 KiB)
  2.630 μs (2 allocations: 78.19 KiB)





=#



"""
    concatenate(A)

Combines a collection of arrays into one larger array, equivalent to using
[`cat`](@ref) on the arrays along each dimension of the outer collection.

* For a vector of arrays (or a tuple of arrays) this is equal to
  [`vcat`](@ref)`(A...) == reduce(vcat, A)`.

* For a matrix of matrices, all the same size, this is a block matrix
  equal to [`hvcat`](@ref)`(size(A,2), permutedims(A)...)`.

* In general, this is equal to [`hvncat`](@ref Base.hvncat)`(size(collect(A)), false, A...)`
  but should be faster. 

The arrays need not be the same size. They are concatenated first along the
1st dimension, then along the 2nd, and so on. For a block matrix this is the
opposite order to [`hvcat`](@ref), but won't matter if the blocks form a grid.

Unlike [`cat`](@ref), numbers are not accepted in place of arrays:
`concatenate((1, [2]))` gives an error, while `vcat(1, [2]) == [1,2]`.

Use [`concatenate!`](@ref) to fill a given output array.

See also [`stack`](@ref) which combines arrays, all the same size,
along new dimensions instead.

!!! Warning
   `concatenate` is an experiment, along with `flatten`.
   I'm not sure it's part of the long-term future of LazyStack.

# Examples
```jldoctest
julia> concatenate([[1,2], [3,4,5], [6]]) == 1:6  # vector of vectors
true

julia> concatenate(Any[[0.1, 0.2]', [30 40; 50 60]])  # vector of matrices
3×2 Matrix{Float64}:
  0.1   0.2
 30.0  40.0
 50.0  60.0

julia> mats = [fill(10i+j, i, j) for i in 1:2, j in 3:5];

julia> concatenate(mats)  # block matrix
3×12 Matrix{Int64}:
 13  13  13  14  14  14  14  15  15  15  15  15
 23  23  23  24  24  24  24  25  25  25  25  25
 23  23  23  24  24  24  24  25  25  25  25  25

julia> ans == hvcat(3, permutedims(mats)...)
true

julia> concatenate([rand(2,2,2,2) for _ in 1:3, _ in 1:4, _ in 1:5]) |> size
(6, 8, 10, 2)

julia> x4 = randn(4,5,6,7);

julia> y4 = eachslice(x4, dims=(3,4), drop=false);  # 1×1×6×7 Slices, of 4×5 matrices

julia> concatenate(y4) == x4
true
```

Example with different-size blocks in each column:

```
julia> sym = reshape([fill(:a,2,3), fill(:β,1,3), fill(:C,1), fill(:Δ,2)], (2,2));

julia> concatenate(sym)  # columns are assembled first
3×4 Matrix{Symbol}:
 :a  :a  :a  :C
 :a  :a  :a  :Δ
 :β  :β  :β  :Δ

julia> permutedims(ans) == hvcat(2, permutedims.(sym)...)  # hvcat works along rows
true
```
"""
concatenate(A) = concatenate3(A)

"""
    concatenate!(dst, src)

In-place version of [`concatenate`](@ref), equivalent to `copyto!(dst, concatenate2(src))`.
The destination array must be of the correct size, apart trailing dimensions of size 1.

# Examples
```jldoctest
julia> concatenate!(rand(3), ([1,2], [30;;]))  # concatenate gives a 3×1 Matrix
3-element Vector{Float64}:
  1.0
  2.0
 30.0

julia> concatenate!(rand(2,3), Any[[1, 2, 3]', [40 50 60]])  # vector of 1-row matrices
2×3 Matrix{Float64}:
  1.0   2.0   3.0
 40.0  50.0  60.0
```
"""
concatenate!(B, A) = concatenate3!(B, A)

