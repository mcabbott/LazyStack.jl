using Test, LazyStack, OffsetArrays

@testset "flatten" begin
    
end


@testset for concatenate in [concatenate2, concatenate3]
    @test @inferred(concatenate([1:2, 3:4, 5:6])) == 1:6
    @test @inferred(concatenate((1:2, 3:4, 5:6))) == 1:6

    @test concatenate([[1,2], [3f0;;;]]) isa Array{Float32, 3}
    @test @inferred(concatenate(([1,2], [3f0;;;]))) isa Array{Float32, 3}

    matofvecs = [rand(4) for _ in 1:2, _ in 1:3]
    @test @inferred(concatenate(matofvecs)) == hvcat(3, permutedims(matofvecs)...)

    mats = [rand(4,5) for _ in 1:3, _ in 1:7]
    @test @inferred(concatenate(mats)) == hvcat(7, permutedims(mats)...)

    sym = reshape([fill(:a,2,3), fill(:β,1,3), fill(:C,1), fill(:Δ,2)], (2,2));
    @test concatenate(sym) == concatenate1(sym)

    concatenate == concatenate2 && continue
    # Things which only work with concatenate3:

    ovv = [OffsetArray(collect(i:2i), 3) for i in 1:3]
    @test_broken vcat(ovv...) == [1,2, 2,3,4, 3,4,5,6]
    @test_broken reduce(vcat, ovv) == [1,2, 2,3,4, 3,4,5,6]
    @test_skip concatenate(ovv) == [1,2, 2,3,4, 3,4,5,6]

    ovm = [OffsetArray(collect(i:2i), 3) for i in 1:3, _ in 1:1]
    @test_broken vcat(ovm...) == [1;2; 2;3;4; 3;4;5;6;;]
    @test concatenate(ovm) == [1;2; 2;3;4; 3;4;5;6;;]

    omv = [OffsetArray([i 2i; i/2 0], 3, 4) for i in 1:2]
    @test vcat(omv...) == [1 2; 0.5 0; 2 4; 1 0]
    @test reduce(vcat, omv) == [1 2; 0.5 0; 2 4; 1 0]
    @test concatenate(omv) == [1 2; 0.5 0; 2 4; 1 0]

    @test concatenate(OffsetArray([[i 0] for i in 1:3], 7)) == [1 0; 2 0; 3 0]
    @test concatenate(OffsetArray([[i j] for i in 1:3, j in 5:6, _ in 1:2], 7,8,9)) isa Array{Int,3}

    matofvecs = Any[x for x in matofvecs]
    @test concatenate(matofvecs) == hvcat(3, permutedims(matofvecs)...)

    tri = [rand(2,3,4) for _ in 1:5, _ in 1:3, _ in 1:2]
    @test concatenate(tri) == concatenate1(tri)

    @test @inferred(concatenate([i j] for i in 1:2, j in 5:7)) == [1 5 1 6 1 7; 2 5 2 6 2 7]
end


