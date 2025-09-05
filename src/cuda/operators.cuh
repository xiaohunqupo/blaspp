// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_OPERATORS_CUH
#define BLAS_OPERATORS_CUH

#include "blas/device.hh"

#include <type_traits>

namespace blas {

//------------------------------------------------------------------------------
/// Map C++ type to CUDA type, e.g.:
///     device_cast_traits< double               >::type == double
///     device_cast_traits< std::complex<double> >::type == cuDoubleComplex
///
template <typename T>
struct device_cast_traits
{
    using type = T;
};

// cuComplex and cuFloatComplex are aliases.
template <>
struct device_cast_traits< std::complex<float> >
{
    using type = cuFloatComplex;
};

template <>
struct device_cast_traits< std::complex<double> >
{
    using type = cuDoubleComplex;
};

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
template <typename T, typename = std::enable_if_t< std::is_arithmetic_v<T> > >
__device__
inline T conj_device( T x )
{
    return x;
}

/// @return conj( x ).
__device__
inline cuFloatComplex conj_device( cuFloatComplex z )
{
    return cuConjf( z );
}

/// @return conj( x ).
__device__
inline cuDoubleComplex conj_device( cuDoubleComplex z )
{
    return cuConj( z );
}

//------------------------------------------------------------------------------
/// @return x + y
__device__
inline cuFloatComplex operator + ( cuFloatComplex x, cuFloatComplex y )
{
    return cuCaddf( x, y );
}

/// @return x + y
__device__
inline cuDoubleComplex operator + ( cuDoubleComplex x, cuDoubleComplex y )
{
    return cuCadd( x, y );
}

//------------------------------------------------------------------------------
/// @return x * y
__device__
inline cuFloatComplex operator * ( cuFloatComplex x, cuFloatComplex y )
{
    return cuCmulf( x, y );
}

/// @return x * y
__device__
inline cuDoubleComplex operator * ( cuDoubleComplex x, cuDoubleComplex y )
{
    return cuCmul( x, y );
}

}  // namespace blas

#endif        // #ifndef BLAS_OPERATORS_CUH
