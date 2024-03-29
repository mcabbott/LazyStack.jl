using LazyStack, BenchmarkTools
using JuliennedArrays, LazyArrays, SliceMap

# Array of Arrays

v1 = [rand(10) for i=1:10];
v4 = [rand(10^4) for i=1:10^4];

@code_warntype stack(v1)

@btime reduce(hcat, $v1);   # 155.417 ns (1 allocation: 896 bytes)
@btime collect(stack($v1)); # 290.430 ns (2 allocations: 912 bytes)
@btime collect(Align($v1,True(),False())); # 332.769 ns (2 allocations: 912 bytes)

@btime reduce(hcat, $v4);   # 429.879 ms (2 allocations: 762.94 MiB)
@btime collect(stack($v4)); # 445.761 ms (3 allocations: 762.94 MiB)
@btime collect(Align($v4,True(),False())); # 452.899 ms (3 allocations: 762.94 MiB)

# Tuple of Arrays

t1 = Tuple(v1);

@code_warntype stack(t1)

@btime hcat($t1...);          #   178.379 ns (4 allocations: 1.03 KiB)

@btime collect(stack($t1));   # 1.032 μs (2 allocations: 912 bytes) -- was quicker I swear
@btime collect($(stack(t1))); # 1.065 μs (1 allocation: 896 bytes)

@btime collect(Hcat($t1...)); # 149.001 ns (3 allocations: 1008 bytes) -- match this

@btime $(hcat(t1...))[9,9]    # 1.424 ns (0 allocations: 0 bytes)
@btime $(stack(t1))[9,9]      # 1.700 ns (0 allocations: 0 bytes)

# Generators

@code_warntype stack(rand(10) for i=1:10)
@code_warntype stack(i<5 ? ones(Int,10) : rand(10) for i=1:10)
@code_warntype collect(i<5 ? ones(Int,10) : rand(10) for i=1:10)

@btime reduce(hcat, [rand(10^3) for i=1:10^3]); # 1.900 ms (1003 allocations: 15.39 MiB)
@btime reduce(hcat, (rand(10^3) for i=1:10^3)); # 2.866 s (3999 allocations: 3.74 GiB)
@btime stack(rand(10^3) for i=1:10^3);          # 1.631 ms (1009 allocations: 15.38 MiB)
@btime stack([rand(10^3) for i=1:10^3]);        # 832.543 μs (1002 allocations: 7.76 MiB)
@btime [rand(10^3) for i=1:10^3];               # 858.156 μs (1001 allocations: 7.76 MiB)

@btime reduce(hcat, [i < 10 ? (i+1:i+10^2) : rand(10^2) for i=1:10^2]); # 65.562 μs (405 allocations: 170.16 KiB)
@btime reduce(hcat, i < 10 ? (i+1:i+10^2) : rand(10^2) for i=1:10^2); # 1.492 ms (396 allocations: 3.94 MiB)
@btime stack(i < 10 ? (i+1:i+10^2) : rand(10^2) for i=1:10^2); # 89.508 μs (9209 allocations: 378.72 KiB)

@btime reduce(hcat, [ones(10^2) for i=1:10^2]); #    12.755 μs (103 allocations: 166.58 KiB)
@btime reduce(hcat, (ones(10^2) for i=1:10^2)); # 1.419 ms (280 allocations: 3.95 MiB)
@btime stack(ones(10^2) for i=1:10^2);          #    12.286 μs (109 allocations: 166.03 KiB)
@btime stack([ones(10^2) for i=1:10^2]);        #     7.141 μs (102 allocations: 88.39 KiB)
@btime collect(stack([ones(10^2) for i=1:10^2])); #  13.037 μs (104 allocations: 166.59 KiB)

# Mapslices

M1 = rand(10,1000);
f1(x) = begin length(x)==10 || error(); identity.(x ./ length(x)) end
f1(x) = sqrt.(x ./ length(x))

@btime mapslices(f1, $M1; dims=1)       # 496.050 μs (7502 allocations: 399.89 KiB)
@btime stack(f1, eachcol($M1))          #  76.542 μs (2012 allocations: 281.72 KiB)
@btime stack(f1, eachslice($M1, dims=2))
@btime mapcols(f1, $M1)                 #  82.132 μs (2006 allocations: 289.33 KiB)

