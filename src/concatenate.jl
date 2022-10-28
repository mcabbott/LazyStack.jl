
export concatenate, concatenate!
export concatenate1, concatenate2, concatenate2!, concatenate3, concatenate3!

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
# For iterators, we need all the sizes to allocate B, so must collect:
concatenate3(A) = concatenate3(collect(A))
concatenate3(A::NamedTuple) = concatenate3(Tuple(A))

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

    # ntuple(d -> sum(a -> size(a, d), first(eachslice(A, dims=d))), N)  # WRONG

        # Aslice = view(A, ntuple(i -> i==d ? (:) : firstindex(A,i), N)...)
        # sum(a -> size(a, d), Aslice)


_concatenate_arraycheck(x::AbstractArray) = nothing
_concatenate_arraycheck(x) = throw(ArgumentError("concatenate only works on arrays, got $(typeof(x))"))

function concatenate3!(B::AbstractArray, A::Union{Tuple, AbstractVector})
    Base.require_one_based_indexing(B)
    post = ntuple(Returns(:), ndims(B) - 1)
    off = 0
    for x in A
        _concatenate_arraycheck(x)
        is = UnitRange(off + 1, off += size(x,1)::Int)
        B[is, post...] = x
        # B[is, post...] .= x
        # copyto!(view(B, is, post...), x)
    end
    B
end

    # for a in Base.axes1(A)
    #     x = A[a]
    #     _concatenate_arraycheck(x)
    #     b = UnitRange(off+1, off+=size(x,1))

    #     # B[b, post...] = x
    #     copyto!(view(B, b, post...), x)  # slower
    #     # view(B, b, post...) .= x  # faster
    # end

# function concatenate3!(B::AbstractArray, A::AbstractMatrix{<:AbstractArray})
#     Base.require_one_based_indexing(B)
#     rest = ntuple(_ -> (:), ndims(B)-ndims(A))

#     off2 = 0
#     for a2 in axes(A,2)
#         x2 = A[begin, a2]
#         b2 = UnitRange(off2+1, off2+=size(x2,2))

#         concatenate3!(view(B, :, b2, rest...), view(A, :, a2))
#     end
#     B
# end

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
    # firsts = ntuple(d -> firstindex(A,d), ndims(A) - 1)
    # for k in axes(A)[end]
    #     x = A[firsts..., k]
    #     is = UnitRange(off + 1, off += size(x, ndims(A)))

    #     concatenate3!(view(B, pre..., is, post...), view(A, pre..., k))
    # end
    B
end


    # if size(A)[end] == 1  # This doesn't help, and might be wrong:
    #     return concatenate3!(B, reshape(A, size(A)[1:end-1]...))
    # end



#=

concatenate1([1:3, 4:5]) == 1:5
concatenate2([1:3, 4:5]) == 1:5

@code_warntype concatenate2([1:3, 4:5])
@code_warntype concatenate2([[1 2], [3 4]])

@code_warntype concatenate3([1:3, 4:5])
@code_warntype concatenate3([[1 2], [3 4]])


let vecs = [rand(100) for _ in 1:100]
    @btime concatenate2($vecs)  # pretty good!
    @btime concatenate3($vecs)
    @btime reduce(vcat, $vecs)
    @btime concatenate1($vecs)  # hvncat
end;

  2.528 μs (2 allocations: 78.17 KiB)
  2.569 μs (2 allocations: 78.17 KiB)
  2.611 μs (3 allocations: 79.05 KiB)
  26.416 μs (18 allocations: 84.12 KiB)


let mats = [rand(10,10) for _ in 1:10, _ in 1:10]
    @btime concatenate2($mats)
    @btime concatenate3($mats)  # slower with .=
    @btime stack($mats)   # better access pattern, easier?
    @btime concatenate1($mats)  # hvncat
    @btime hvcat(10, $mats...)
end;

  13.625 μs (8 allocations: 78.34 KiB)
  13.916 μs (8 allocations: 78.34 KiB)
  2.810 μs (3 allocations: 78.23 KiB)
  58.666 μs (20 allocations: 84.34 KiB)
  19.708 μs (9 allocations: 80.23 KiB)


let mv = [rand(100) for _ in 1:10, _ in 1:10] # now the same pattern, still faster!
# let mv = [rand(100) for _ in 1:100, _ in 1:1]  # similar results
    @btime concatenate2($mv)
    @btime concatenate3($mv)  # faster with .=
    @btime stack($mv) 
end;
  12.250 μs (8 allocations: 78.34 KiB)
  4.056 μs (8 allocations: 78.34 KiB)  # fast with .=
  2.778 μs (2 allocations: 78.19 KiB)


let vm = [rand(10, 10) for _ in 1:100]
    @btime concatenate2($vm)
    @btime concatenate3($vm) 
    @btime reduce(vcat, $vm)
    @btime stack($vm)   # better access pattern, easier?
end;

  12.791 μs (4 allocations: 78.27 KiB)
  12.791 μs (4 allocations: 78.27 KiB)
  12.875 μs (2 allocations: 78.17 KiB)
  2.819 μs (2 allocations: 78.19 KiB)



julia> let m = rand(10)
         x = ones(5)
         @btime $m[2:6] = $x
         @btime $m[2:6] .= $x
         @btime @views copyto!($m[2:6], $x)
       end;
  min 2.500 ns, mean 2.635 ns (0 allocations)
  min 13.318 ns, mean 13.895 ns (0 allocations)
  min 13.192 ns, mean 13.374 ns (0 allocations)

julia> let m = rand(10,1)
         x = ones(5)
         @btime $m[2:6,:] = $x
         @btime $m[2:6,:] .= $x
         @btime @views copyto!($m[2:6,:], $x)
       end;
  min 12.178 ns, mean 12.334 ns (0 allocations)
  min 5.458 ns, mean 5.605 ns (0 allocations)
  min 16.115 ns, mean 16.250 ns (0 allocations)

julia> let m = rand(10,10)
         x = ones(5)
         @btime $m[2:6,2] = $x
         @btime $m[2:6,3] .= $x
         @btime @views copyto!($m[2:6,4], $x)
       end;
  min 10.677 ns, mean 10.803 ns (0 allocations)
  min 15.155 ns, mean 15.288 ns (0 allocations)
  min 14.570 ns, mean 14.766 ns (0 allocations)


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

