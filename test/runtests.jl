using Test, LazyStack
using OffsetArrays

@testset "basics" begin

    v34 = [rand(3) for i in 1:4]

    @test lazystack(v34) == hcat(v34...)
    @test lazystack(v34) isa LazyStack.Stacked

    @test lazystack(v34...) == hcat(v34...)
    @test lazystack(v34...).slices isa Tuple
    @test LazyStack.ensure_dense(lazystack(v34...)) isa Array

    @test lazystack(v34[i] for i in 1:4) == hcat(v34...)
    @test lazystack(v34[i] for i in 1:4) isa Array

    @test lazystack(v34)[1] == v34[1][1] # linear indexing
    @test lazystack(v34)[1,1,1] == v34[1][1] # trailing dims
    @test lazystack(v34) * ones(4) ≈ hcat(v34...) * ones(4) # issue #6
    @test lazystack(v34) * ones(4,2) ≈ hcat(v34...) * ones(4,2)
    @test axes(lazystack(v34)) === axes(lazystack(v34...)) === axes(lazystack(v34[i] for i in 1:4))

end
@testset "tuples" begin

    vt = [(1,2), (3,4), (5,6)]
    vnt = [(a=1,b=2), (a=3,b=4), (a=5,b=6)]
    @test lazystack(vt) isa Array
    @test lazystack(vt) == reshape(1:6, 2,3)
    @test lazystack(vnt) isa Array

    @test lazystack(vt...) isa Array
    @test lazystack(vnt...) isa Array

end
@testset "generators" begin

    g234 = (ones(2) .* (10i + j) for i in 1:3, j in 1:4)
    @test lazystack(g234) == lazystack(collect(g234))

    @test lazystack(Iterators.filter(_ -> true, g234)) == reshape(lazystack(g234), 2,:)

    @test lazystack(Iterators.drop(g234, 2)) == reshape(lazystack(g234), 2,:)[:, 3:end]

    g16 = (ones(1) .* (10i + j) for j in 1:3 for i in 1:2) # Iterators.Flatten
    @test lazystack(collect(g16)) == lazystack(g16)

    g29 = (fill(i,2) for i=1:9)
    @test lazystack(Iterators.reverse(g29)) == reverse(lazystack(g29), dims=2)

    gt = ((1,2,3) for i in 1:4)
    @test lazystack(gt) isa Array

end
@testset "functions" begin

    m1 = rand(1:99, 3,10)
    _eachcol(m) = (view(m, :, c) for c in axes(m,2))

    @test lazystack(sum, _eachcol(m1)) == vec(mapslices(sum, m1, dims=1))

    f1(x,y) = x .+ y
    @test lazystack(f1, _eachcol(m1), _eachcol(m1)) == 2 .* m1
    @test lazystack(f1, _eachcol(m1), 1:10) == m1 .+ (1:10)'

    # @test_throws DimensionMismatch map(f1, _eachcol(m1), 1:12) # it doesn't throw!
    # @test_throws DimensionMismatch lazystack(f1, _eachcol(m1), 1:12)

    # This is where lazystack doesn't quite follow map's behaviour:
    @test size(lazystack(map(*, [ones(2) for i=1:3, j=1:4], ones(3)))) == (2,3)
    @test size(lazystack(map(*, [ones(2) for i=1:3, j=1:4], ones(5)))) == (2,5)
    # @test_throws DimensionMismatch lazystack(*, [ones(2) for i=1:3, j=1:4], ones(3))
    # @test_throws DimensionMismatch map(*, [ones(2) for i=1:3, j=1:4], ones(3,1))

end
@testset "types" begin

    @test lazystack([1:3 for _=1:2]...) isa LazyStack.Stacked{Int,2,<:Tuple{UnitRange,UnitRange}}

    @test_broken eltype(lazystack(1:3, ones(3))) == Real
    @test_broken eltype(lazystack([1:3, ones(3)])) == Real
    @test_broken eltype(lazystack(i==1 ? (1:3) : ones(3) for i in 1:2)) == Real

    acc = []
    for i=1:3
        push!(acc, fill(i, 4))
    end
    @test lazystack(acc) isa Array{Int}
    push!(acc, rand(4))
    @test_broken lazystack(acc) isa Array{Real}

    acc = Array[]
    for i=1:3
        push!(acc, fill(i, 4))
    end
    lazystack(acc) isa Array{Int}

end
@testset "offset" begin

    oin = [OffsetArray(ones(3), 3:5) for i in 1:4]
    @test axes(lazystack(oin)) == (3:5, 1:4)
    @test axes(lazystack(oin...)) == (3:5, 1:4)
    @test axes(copy(lazystack(oin))) == (3:5, 1:4)

    oout = OffsetArray([ones(3) for i in 1:4], 11:14)
    @test axes(lazystack(oout)) == (1:3, 11:14)
    @test axes(copy(lazystack(oout))) ==  (1:3, 11:14)

    oboth = OffsetArray(oin, 11:14)
    @test axes(lazystack(oboth)) == (3:5, 11:14)

    ogen = (OffsetArray([3,4,5], 3:5) for i in 1:4)
    @test axes(lazystack(ogen)) == (3:5, 1:4)

end
@testset "push!" begin

    v34 = [rand(3) for i in 1:4]
    s3 = lazystack(v34)

    @test size(push!(s3, ones(3))) == (3,5)
    @test s3[1,end] == 1

end
@testset "errors" begin

    @test_throws ArgumentError lazystack([])
    @test_throws ArgumentError lazystack(x for x in 1:0)

    @test_throws DimensionMismatch lazystack(1:n for n in 1:3)
    @test_throws DimensionMismatch lazystack([1:n for n in 1:3])

    @test_throws DimensionMismatch push!(lazystack([rand(2)]), rand(3))

end
@testset "readme" begin

    using LazyStack: Stacked
    _eachcol(m) = (view(m, :, c) for c in axes(m,2))

    @test lazystack([zeros(2,2), ones(2,2)]) isa Stacked{Float64, 3, <:Vector{<:Matrix}}
    @test lazystack([1,2,3], 4:6) isa Stacked{Int, 2, <:Tuple{<:Vector, <:UnitRange}}

    @test lazystack([i,2i] for i in 1:5) isa Matrix{Int}
    @test lazystack(*, _eachcol(ones(2,4)), 1:4) isa Matrix{Float64}
    @test_broken lazystack([1,2], [3.0, 4.0], [5im, 6im]) isa Matrix{Number}

    @test raggedstack([1:n for n in 1:10]) isa Matrix{Int}
    @test raggedstack(OffsetArray(fill(n,4), rand(-2:2)) for n in 1:10; fill=NaN) isa OffsetArray{Real,2}

end
@testset "ragged" begin

    @test raggedstack([1,2], 1:3) == [1 1; 2 2; 0 3]
    @test raggedstack([[1,2], 1:3], fill=99) == [1 1; 2 2; 99 3]

    @test raggedstack(1:2, OffsetArray([2,3], +1)) == [1 0; 2 2; 0 3]
    @test raggedstack(1:2, OffsetArray([0.1,1], -1)) == OffsetArray([0 0.1; 1 1.0; 2 0],-1,0)

end
@testset "tuple functions" begin

    @test LazyStack._ndims([1,2]) == 1
    @test LazyStack._ndims((1,2)) == 1
    @test LazyStack._ndims((a=1,b=2)) == 1

    @test LazyStack._size([1,2]) == (2,)
    @test LazyStack._size((1,2)) == (2,)
    @test LazyStack._size((a=1,b=2)) == (2,)

    @test LazyStack._axes([1,2]) == (1:2,)
    @test LazyStack._axes((1,2)) == (1:2,)
    @test LazyStack._axes((a=1,b=2)) == (1:2,)

end
@info "loading Zygote"
using Zygote
@testset "zygote" begin

    @test Zygote.gradient((x,y) -> sum(lazystack(x,y)), ones(2), ones(2)) == ([1,1], [1,1])
    @test Zygote.gradient((x,y) -> sum(lazystack([x,y])), ones(2), ones(2)) == ([1,1], [1,1])

    f399(x) = sum(lazystack(x) * sum(x))
    f399c(x) = sum(collect(lazystack(x)) * sum(x))
    @test Zygote.gradient(f399, [ones(2), ones(2)]) == ([[4,4], [4,4]],)
    @test Zygote.gradient(f399c, [ones(2), ones(2)]) == ([[4,4], [4,4]],)
    ftup(x) = sum(lazystack(x...) * sum(x))
    ftupc(x) = sum(collect(lazystack(x...)) * sum(x))
    @test Zygote.gradient(ftup, (ones(2), ones(2))) == (([4,4], [4,4]),)
    @test Zygote.gradient(ftupc, (ones(2), ones(2))) == (([4,4], [4,4]),)

end
