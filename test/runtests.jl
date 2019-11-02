using Test, LazyStack
using OffsetArrays, NamedDims, Zygote

@testset "basics" begin

    v34 = [rand(3) for i in 1:4]

    @test stack(v34) == hcat(v34...)
    @test stack(v34) isa LazyStack.Stacked

    @test stack(v34...) == hcat(v34...)
    @test stack(v34...).slices isa Tuple

    @test stack(v34[i] for i in 1:4) == hcat(v34...)
    @test stack(v34[i] for i in 1:4) isa Array

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
    @test NamedDims.names(stack(nin)) == (:a, :_)

    nout = NamedDimsArray([ones(3) for i in 1:4], :b)
    @test NamedDims.names(stack(nout)) == (:_, :b)

    nboth = NamedDimsArray([NamedDimsArray(ones(3), :a) for i in 1:4], :b)
    @test NamedDims.names(stack(nboth)) == (:a, :b)

    ngen = (NamedDimsArray(ones(3), :a) for i in 1:4)
    NamedDims.names(stack(ngen)) == (:a, :_)

end
@testset "offset" begin

    oin = [OffsetArray(ones(3), 3:5) for i in 1:4]
    @test axes(stack(oin)) == (3:5, 1:4)
    @test axes(copy(stack(oin))) == (3:5, 1:4)

    oout = OffsetArray([ones(3) for i in 1:4], 11:14)
    @test axes(stack(oout)) == (1:3, 11:14)
    @test axes(copy(stack(oout))) ==  (1:3, 11:14)

    ogen = (OffsetArray([3,4,5], 3:5) for i in 1:4)
    @test axes(stack(ogen)) == (3:5, 1:4)

end
@testset "errors" begin

    @test_throws ArgumentError stack([])
    @test_throws ArgumentError stack(x for x in 1:0)

    @test_throws DimensionMismatch stack(1:n for n in 1:3)
    @test_throws DimensionMismatch stack([1:n for n in 1:3])

end
@testset "zygote" begin

    @test Zygote.gradient((x,y) -> sum(stack(x,y)), ones(2), ones(2)) == ([1,1], [1,1])
    @test Zygote.gradient((x,y) -> sum(stack([x,y])), ones(2), ones(2)) == ([1,1], [1,1])

end
@testset "readme" begin

    using LazyStack: Stacked

    @test stack([zeros(2,2), ones(2,2)]) isa Stacked{Float64, 3, <:Vector{<:Matrix}}
    @test stack([1,2,3], 4:6) isa Stacked{Int, 2, <:Tuple{<:Vector, <:UnitRange}}

    @test stack([i,2i] for i in 1:5) isa Matrix{Int}
    @test stack([1,2], [3.0, 4.0], [5im, 6im]) isa Matrix{Number}

end
