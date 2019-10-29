using LazyStack, BenchmarkTools, JuliennedArrays, LazyArrays

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
@btime stack(rand(10^3) for i=1:10^3);          # 1.645 ms (2004 allocations: 15.43 MiB)
@btime stack([rand(10^3) for i=1:10^3]);        # 832.543 μs (1002 allocations: 7.76 MiB)
@btime [rand(10^3) for i=1:10^3];               # 858.156 μs (1001 allocations: 7.76 MiB)

@btime reduce(hcat, [i < 10 ? (i+1:i+10^2) : rand(10^2) for i=1:10^2]); # 65.562 μs (405 allocations: 170.16 KiB)
@btime stack(i < 10 ? (i+1:i+10^2) : rand(10^2) for i=1:10^2); # 88.022 μs (9303 allocations: 383.14 KiB)

