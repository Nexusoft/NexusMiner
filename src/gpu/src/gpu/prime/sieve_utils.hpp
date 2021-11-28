#ifndef NEXUSMINER_GPU_SIEVE_UTILS_HPP
#define NEXUSMINER_GPU_SIEVE_UTILS_HPP

// https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
// C++14 compile time square root using binary search

template <typename T>
constexpr T sqrt_helper(T x, T lo, T hi)
{
    if (lo == hi)
        return lo;

    const T mid = (lo + hi + 1) / 2;

    if (x / mid < mid)
        return sqrt_helper<T>(x, lo, mid - 1);
    else
        return sqrt_helper(x, mid, hi);
}

template <typename T>
constexpr T ct_sqrt(T x)
{
    return sqrt_helper<T>(x, 0, x / 2 + 1);
}

//return the offset from x to the next integer multiple of n greater than x that is not divisible by 2, 3, or 5.  
//x must be a multiple of the primorial 30 and n must be a prime greater than 5.
template <typename T1, typename T2>
static T2 get_offset_to_next_multiple(T1 x, T2 n)
{
    T2 m = n - static_cast<T2>(x % n);
    if (m % 2 == 0)
    {
        m += n;
    }
    while (m % 3 == 0 || m % 5 == 0)
    {
        m += 2 * n;
    }
    return m;
}

//return the offset from x to the next integer multiple of n greater than x that is not divisible by 2, 3, 5, or 7.  
//x must be a multiple of the primorial 210 and n must be a prime greater than 7.
template <typename T1, typename T2>
static T2 get_offset_to_next_multiple_7(T1 x, T2 n)
{
    T2 m = n - static_cast<T2>(x % n);
    if (m % 2 == 0)
    {
        m += n;
    }
    //this loop executes max 4 times
    while (m % 3 == 0 || m % 5 == 0 || m % 7 == 0)
    {
        m += 2 * n;
    }

    return m;
}


#endif