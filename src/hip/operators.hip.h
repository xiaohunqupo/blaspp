// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_OPERATORS_HIP_H
#define BLAS_OPERATORS_HIP_H

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

__device__ inline hipFloatComplex conj_device(hipFloatComplex z)
{
    return hipConjf(z);
}

__device__ inline hipDoubleComplex conj_device(hipDoubleComplex z)
{
    return hipConj(z);
}

// operator +
__device__ inline hipFloatComplex operator +(hipFloatComplex x, hipFloatComplex y)
{
    return hipCaddf(x, y);
}

__device__ inline hipDoubleComplex operator +(hipDoubleComplex x, hipDoubleComplex y)
{
    return hipCadd(x, y);
}

// operator *
__device__ inline hipFloatComplex operator *(hipFloatComplex x, hipFloatComplex y)
{
    return hipCmulf(x, y);
}

__device__ inline hipDoubleComplex operator *(hipDoubleComplex x, hipDoubleComplex y)
{
    return hipCmul(x, y);
}

}  // namespace blas

#endif        // #ifndef BLAS_OPERATORS_HIP_H