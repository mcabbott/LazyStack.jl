using Test, LazyStack
using OffsetArrays, NamedDims

@testset "basics" begin

    v34 = [rand(3) for i in 1:4]

    @test stack(v34) == hcat(v34...)
    @test stack(v34) isa LazyStack.Stacked

    @test stack(v34...) == hcat(v34...)
    @test stack(v34...).slices isa Tuple
    @test LazyStack.ensure_dense(stack(v34...)) isa Array

    @test stack(v34[i] for i in 1:4) == hcat(v34...)
    @test stack(v34[i] for i in 1:4) isa Array

    @test stack(v34)[1] == v34[1][1] # linear indexing
    @test stack(v34)[1,1,1] == v34[1][1] # trailing dims
    @test stack(v34) * ones(4) ≈ hcat(v34...) * ones(4) # issue #6
    @test stack(v34) * ones(4,2) ≈ hcat(v34...) * ones(4,2)
    @test axes(stack(v34)) === axes(stack(v34...)) === axes(stack(v34[i] for i in 1:4))

end
@testset "tuples" begin

    vt = [(1,2), (3,4), (5,6)]
    vnt = [(a=1,b=2), (a=3,b=4), (a=5,b=6)]
    @test stack(vt) isa Array
    @test stack(vt) == reshape(1:6, 2,3)
    @test stack(vnt) isa Array

    @test stack(vt...) isa Array
    @test stack(vnt...) isa Array

end
@testset "generators" begin

    g234 = (ones(2) .* (10i + j) for i in 1:3, j in 1:4)
    @test stack(g234) == stack(collect(g234))

    @test stack(Iterators.filter(_ -> true, g234)) == reshape(stack(g234), 2,:)

    @test stack(Iterators.drop(g234, 2)) == reshape(stack(g234), 2,:)[:, 3:end]

    g16 = (ones(1) .* (10i + j) for j in 1:3 for i in 1:2) # Iterators.Flatten
    @test stack(collect(g16)) == stack(g16)

    g29 = (fill(i,2) for i=1:9)
    @test stack(Iterators.reverse(g29)) == reverse(stack(g29), dims=2)

    gt = ((1,2,3) for i in 1:4)
    @test stack(gt) isa Array

end
@testset "functions" begin

    m1 = rand(1:99, 3,10)
    _eachcol(m) = (view(m, :, c) for c in axes(m,2))

    @test stack(sum, _eachcol(m1)) == vec(mapslices(sum, m1, dims=1))

    f1(x,y) = x .+ y
    @test stack(f1, _eachcol(m1), _eachcol(m1)) == 2 .* m1
    @test stack(f1, _eachcol(m1), 1:10) == m1 .+ (1:10)'

    @test_throws DimensionMismatch map(f1, _eachcol(m1), 1:12)
    @test_throws DimensionMismatch stack(f1, _eachcol(m1), 1:12)

    # This is where stack doesn't quite follow map's behaviour:
    @test size(stack(map(*, [ones(2) for i=1:3, j=1:4], ones(3)))) == (2,3)
    @test size(stack(map(*, [ones(2) for i=1:3, j=1:4], ones(5)))) == (2,5)
    @test_throws DimensionMismatch stack(*, [ones(2) for i=1:3, j=1:4], ones(3))
    @test_throws DimensionMismatch map(*, [ones(2) for i=1:3, j=1:4], ones(3,1))

end
@testset "types" begin

    @test stack([1:3 for _=1:2]...) isa LazyStack.Stacked{Int,2,<:Tuple{UnitRange,UnitRange}}

    @test eltype(stack(1:3, ones(3))) == Real
    @test eltype(stack([1:3, ones(3)])) == Real
    @test eltype(stack(i==1 ? (1:3) : ones(3) for i in 1:2)) == Real

    acc = []
    for i=1:3
        push!(acc, fill(i, 4))
    end
    @test stack(acc) isa Array{Int}
    push!(acc, rand(4))
    @test stack(acc) isa Array{Real}

    acc = Array[]
    for i=1:3
        push!(acc, fill(i, 4))
    end
    stack(acc) isa Array{Int}

end
@testset "names" begin

    nin = [NamedDimsArray(ones(3), :a) for i in 1:4]
    @test dimnames(stack(nin)) == (:a, :_)
    @test dimnames(stack(nin...)) == (:a, :_)
    @test dimnames(stack(:b, nin)) == (:a, :b)
    @test dimnames(stack(:b, nin...)) == (:a, :b)
    @test stack(nin).data.slices[1] isa NamedDimsArray # vector container untouched,
    @test stack(nin...).data.slices[1] isa Array # but tuple container cleaned up.

    nout = NamedDimsArray([ones(3) for i in 1:4], :b)
    @test dimnames(stack(nout)) == (:_, :b)
    @test dimnames(stack(:b, nout)) == (:_, :b)
    @test_throws Exception stack(:c, nout)

    nboth = NamedDimsArray([NamedDimsArray(ones(3), :a) for i in 1:4], :b)
    @test dimnames(stack(nboth)) == (:a, :b)

    ngen = (NamedDimsArray(ones(3), :a) for i in 1:4)
    @test dimnames(stack(ngen)) == (:a, :_)
    @test dimnames(stack(:b, ngen)) == (:a, :b)

    nmat = [NamedDimsArray(ones(3), :a) for i in 1:3, j in 1:4]
    @test dimnames(stack(:c, nmat)) == (:a, :_, :c)

end
@testset "offset" begin

    oin = [OffsetArray(ones(3), 3:5) for i in 1:4]
    @test axes(stack(oin)) == (3:5, 1:4)
    @test axes(stack(oin...)) == (3:5, 1:4)
    @test axes(copy(stack(oin))) == (3:5, 1:4)

    oout = OffsetArray([ones(3) for i in 1:4], 11:14)
    @test axes(stack(oout)) == (1:3, 11:14)
    @test axes(copy(stack(oout))) ==  (1:3, 11:14)

    oboth = OffsetArray(oin, 11:14)
    @test axes(stack(oboth)) == (3:5, 11:14)

    ogen = (OffsetArray([3,4,5], 3:5) for i in 1:4)
    @test axes(stack(ogen)) == (3:5, 1:4)

end
@testset "named offset" begin

    noin = [NamedDimsArray(OffsetArray(ones(3), 3:5), :a) for i in 1:4]
    @test dimnames(stack(noin)) == (:a, :_)
    @test dimnames(stack(noin...)) == (:a, :_)
    @test dimnames(stack(:b, noin)) == (:a, :b)
    @test dimnames(stack(:b, noin...)) == (:a, :b)
    @test axes(stack(noin)) == (3:5, 1:4)
    @test axes(stack(noin...)) == (3:5, 1:4)

    noout = NamedDimsArray(OffsetArray([ones(3) for i in 1:4], 11:14), :b)
    @test dimnames(stack(noout)) == (:_, :b)
    @test dimnames(stack(:b, noout)) == (:_, :b)
    @test_throws Exception stack(:c, noout)
    @test axes(stack(noout)) == (1:3, 11:14)
    @test axes(copy(stack(noout))) ==  (1:3, 11:14)

    nogen = (NamedDimsArray(OffsetArray([3,4,5], 3:5), :a) for i in 1:4)
    @test dimnames(stack(nogen)) == (:a, :_)
    @test dimnames(stack(:b, nogen)) == (:a, :b)
    @test axes(stack(nogen)) == (3:5, 1:4)

end
@testset "push!" begin

    v34 = [rand(3) for i in 1:4]
    s3 = stack(v34)

    @test size(push!(s3, ones(3))) == (3,5)
    @test s3[1,end] == 1

end
@testset "errors" begin

    @test_throws ArgumentError stack([])
    @test_throws ArgumentError stack(x for x in 1:0)

    @test_throws DimensionMismatch stack(1:n for n in 1:3)
    @test_throws DimensionMismatch stack([1:n for n in 1:3])

    @test_throws DimensionMismatch push!(stack([rand(2)]), rand(3))

end
@testset "readme" begin

    using LazyStack: Stacked
    _eachcol(m) = (view(m, :, c) for c in axes(m,2))

    @test stack([zeros(2,2), ones(2,2)]) isa Stacked{Float64, 3, <:Vector{<:Matrix}}
    @test stack([1,2,3], 4:6) isa Stacked{Int, 2, <:Tuple{<:Vector, <:UnitRange}}

    @test stack([i,2i] for i in 1:5) isa Matrix{Int}
    @test stack(*, _eachcol(ones(2,4)), 1:4) isa Matrix{Float64}
    @test stack([1,2], [3.0, 4.0], [5im, 6im]) isa Matrix{Number}

end
@testset "vstack" begin

    v34 = [rand(3) for i in 1:4]
    @test LazyStack.vstack(v34) == reduce(vcat, v34)

    g234 = (ones(2) .* (10i + j) for i in 1:3, j in 1:4)
    @test LazyStack.vstack(g234) == reduce(vcat, collect(g234))

end
@testset "ragged" begin

    @test rstack([1,2], 1:3) == [1 1; 2 2; 0 3]
    @test rstack([[1,2], 1:3], fill=99) == [1 1; 2 2; 99 3]

    @test rstack(1:2, OffsetArray([2,3], +1)) == [1 0; 2 2; 0 3]
    @test rstack(1:2, OffsetArray([0.1,1], -1)) == OffsetArray([0 0.1; 1 1.0; 2 0],-1,0)

    @test dimnames(rstack(:b, 1:2, [3,4,5], fill=NaN)) == (:_, :b)
    @test dimnames(rstack(:b, NamedDimsArray(1:2, :a), OffsetArray([2,3], +1))) == (:a, :b)

end
@testset "tuple functions" begin

    @test LazyStack.ndims([1,2]) == 1
    @test LazyStack.ndims((1,2)) == 1
    @test LazyStack.ndims((a=1,b=2)) == 1

    @test LazyStack.size([1,2]) == (2,)
    @test LazyStack.size((1,2)) == (2,)
    @test LazyStack.size((a=1,b=2)) == (2,)

    @test LazyStack.axes([1,2]) == (1:2,)
    @test LazyStack.axes((1,2)) == (1:2,)
    @test LazyStack.axes((a=1,b=2)) == (1:2,)

end
@info "loading Zygote"
using Zygote
@testset "zygote" begin

    @test Zygote.gradient((x,y) -> sum(stack(x,y)), ones(2), ones(2)) == ([1,1], [1,1])
    @test Zygote.gradient((x,y) -> sum(stack([x,y])), ones(2), ones(2)) == ([1,1], [1,1])

    f399(x) = sum(stack(x) * sum(x))
    f399c(x) = sum(collect(stack(x)) * sum(x))
    @test Zygote.gradient(f399, [ones(2), ones(2)]) == ([[4,4], [4,4]],)
    @test Zygote.gradient(f399c, [ones(2), ones(2)]) == ([[4,4], [4,4]],)
    ftup(x) = sum(stack(x...) * sum(x))
    ftupc(x) = sum(collect(stack(x...)) * sum(x))
    @test Zygote.gradient(ftup, (ones(2), ones(2))) == (([4,4], [4,4]),)
    @test Zygote.gradient(ftupc, (ones(2), ones(2))) == (([4,4], [4,4]),)

end
