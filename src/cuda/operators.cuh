// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_OPERATORS_CUH
#define BLAS_OPERATORS_CUH

#include "blas/device.hh"

namespace blas {

// max
__device__ inline int64_t max_device(int64_t x, int64_t y)
{
    return (x > y) ? x : y;
}

// min
__device__ inline int64_t min_device(int64_t x, int64_t y)
{
    return (x < y) ? x : y;
}

// conj_device
__device__ inline float conj_device(float x)
{
    return x;
}

__device__ inline double conj_device(double x)
{
    return x;
}

__device__ inline cuComplex conj_device(cuComplex z)
{
    return cuConjf(z);
}

__device__ inline cuDoubleComplex conj_device(cuDoubleComplex z)
{
    return cuConj(z);
}

// operator +
__device__ inline cuComplex operator +(cuComplex x, cuComplex y)
{
    return cuCaddf(x, y);
}

__device__ inline cuDoubleComplex operator +(cuDoubleComplex x, cuDoubleComplex y)
{
    return cuCadd(x, y);
}

// operator *
__device__ inline cuComplex operator *(cuComplex x, cuComplex y)
{
    return cuCmulf(x, y);
}

__device__ inline cuDoubleComplex operator *(cuDoubleComplex x, cuDoubleComplex y)
{
    return cuCmul(x, y);
}

}  // namespace blas

#endif        // #ifndef BLAS_OPERATORS_CUH