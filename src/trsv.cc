// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas_internal.hh"
#include "blas/counter.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup trsv_internal
inline void trsv(
    char uplo,
    char trans,
    char diag,
    blas_int n,
    float const* A, blas_int lda,
    float*       x, blas_int incx )
{
    BLAS_strsv( &uplo, &trans, &diag, &n, A, &lda, x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup trsv_internal
inline void trsv(
    char uplo,
    char trans,
    char diag,
    blas_int n,
    double const* A, blas_int lda,
    double*       x, blas_int incx )
{
    BLAS_dtrsv( &uplo, &trans, &diag, &n, A, &lda, x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup trsv_internal
inline void trsv(
    char uplo,
    char trans,
    char diag,
    blas_int n,
    std::complex<float> const* A, blas_int lda,
    std::complex<float>*       x, blas_int incx )
{
    BLAS_ctrsv( &uplo, &trans, &diag, &n,
                (blas_complex_float*) A, &lda,
                (blas_complex_float*) x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup trsv_internal
inline void trsv(
    char uplo,
    char trans,
    char diag,
    blas_int n,
    std::complex<double> const* A, blas_int lda,
    std::complex<double>*       x, blas_int incx )
{
    BLAS_ztrsv( &uplo, &trans, &diag, &n,
                (blas_complex_double*) A, &lda,
                (blas_complex_double*) x, &incx );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup trsv_internal
///
template <typename scalar_t>
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    scalar_t const* A, int64_t lda,
    scalar_t*       x, int64_t incx )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::trsv_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, trans, diag, n };
        counter::insert( element, counter::Id::trsv );

        double gflops = 1e9 * blas::Gflop< scalar_t >::trsv( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );

    blas::Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);

        if constexpr (is_complex_v<scalar_t>) {
            if (trans == Op::ConjTrans) {
                // conjugate x (in-place)
                int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
                for (int64_t i = 0; i < n; ++i) {
                    x[ ix ] = conj( x[ ix ] );
                    ix += incx;
                }
            }
        }
    }
    char uplo_  = to_char( uplo );
    char trans_ = to_char( trans2 );
    char diag_  = to_char( diag );

    // call low-level wrapper
    internal::trsv( uplo_, trans_, diag_, n_, A, lda_, x, incx_ );

    if constexpr (is_complex_v<scalar_t>) {
        if (layout == Layout::RowMajor && trans == Op::ConjTrans) {
            // conjugate x (in-place)
            int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
            for (int64_t i = 0; i < n; ++i) {
                x[ ix ] = conj( x[ ix ] );
                ix += incx;
            }
        }
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const* A, int64_t lda,
    float*       x, int64_t incx )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const* A, int64_t lda,
    double*       x, int64_t incx )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       x, int64_t incx )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       x, int64_t incx )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx );
}

}  // namespace blas
