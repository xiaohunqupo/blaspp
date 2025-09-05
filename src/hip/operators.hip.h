// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_OPERATORS_HIP_H
#define BLAS_OPERATORS_HIP_H

#include "blas/device.hh"

namespace blas {

//------------------------------------------------------------------------------
/// @return max( x, y )
__device__
inline int64_t max_device( int64_t x, int64_t y )
{
    return (x > y) ? x : y;
}

//------------------------------------------------------------------------------
/// @return min( x, y )
__device__
inline int64_t min_device( int64_t x, int64_t y )
{
    return (x < y) ? x : y;
}

//------------------------------------------------------------------------------
/// @return conj( x ). For non-complex types, returns x.
template <typename T, typename = std::enable_if_t< std::is_arithmetic_t<T> > >
__device__
inline T conj_device( T x )
{
    return x;
}

/// @return conj( x ).
__device__
inline hipFloatComplex conj_device( hipFloatComplex z )
{
    return hipConjf( z );
}

/// @return conj( x ).
__device__
inline hipDoubleComplex conj_device( hipDoubleComplex z )
{
    return hipConj( z );
}

//------------------------------------------------------------------------------
/// @return x + y
__device__
inline hipFloatComplex operator + ( hipFloatComplex x, hipFloatComplex y )
{
    return hipCaddf( x, y );
}

/// @return x + y
__device__
inline hipDoubleComplex operator + ( hipDoubleComplex x, hipDoubleComplex y )
{
    return hipCadd( x, y );
}

//------------------------------------------------------------------------------
/// @return x * y
__device__
inline hipFloatComplex operator * ( hipFloatComplex x, hipFloatComplex y )
{
    return hipCmulf( x, y );
}

/// @return x * y
__device__
inline hipDoubleComplex operator * ( hipDoubleComplex x, hipDoubleComplex y )
{
    return hipCmul( x, y );
}

}  // namespace blas

#endif        // #ifndef BLAS_OPERATORS_HIP_H
