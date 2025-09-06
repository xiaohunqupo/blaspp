// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "blas/device.hh"

//------------------------------------------------------------------------------
template <typename scalar_t>
void cpu_geadd(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb )
{
    using blas::conj;
    using blas::Layout, blas::Op;

    if (layout == Layout::RowMajor) {
        std::swap( m, n );
    }

    if (trans == Op::NoTrans) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                B[ i + j*ldb ] = beta * B[ i + j*ldb ] + alpha * A[ i + j*lda ];
    }
    else if (trans == Op::Trans) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                B[ i + j*ldb ] = beta * B[ i + j*ldb ] + alpha * A[ j + i*lda ];
    }
    else if (trans == Op::ConjTrans) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                B[ i + j*ldb ] = beta * B[ i + j*ldb ]
                               + alpha * conj( A[ j + i*lda ] );
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_geadd_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op, blas::Layout, blas::max;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    scalar_t beta   = params.beta.get<scalar_t>();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t device  = params.device();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    int64_t Am = (trans == Op::NoTrans ? m : n);
    int64_t An = (trans == Op::NoTrans ? n : m);
    int64_t Bm = m;
    int64_t Bn = n;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
    }
    int64_t lda = max( roundup( Am, align ), 1 );
    int64_t ldb = max( roundup( Bm, align ), 1 );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    scalar_t* A    = new scalar_t[ size_A ];
    scalar_t* B    = new scalar_t[ size_B ];
    scalar_t* Bref = new scalar_t[ size_B ];

    // device specifics
    blas::Queue queue( device );
    scalar_t* dA;
    scalar_t* dB;

    dA = blas::device_malloc<scalar_t>( size_A, queue );
    dB = blas::device_malloc<scalar_t>( size_B, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_lacpy( "g", Bm, Bn, B, ldb, Bref, ldb );

    blas::device_copy_matrix( Am, An, A, lda, dA, lda, queue );
    blas::device_copy_matrix( Bm, Bn, B, ldb, dB, ldb, queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );

    // test error exits
    assert_throw( blas::geadd( Layout(0), trans,  m,  n, alpha, dA, lda, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( layout,    Op(0),  m,  n, alpha, dA, lda, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( layout,    trans, -1,  n, alpha, dA, lda, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( layout,    trans,  m, -1, alpha, dA, lda, beta, dB, ldb, queue ), blas::Error );

    assert_throw( blas::geadd( Layout::ColMajor, Op::NoTrans,   m, n, alpha, dA, m-1, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( Layout::ColMajor, Op::Trans,     m, n, alpha, dA, n-1, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( Layout::ColMajor, Op::ConjTrans, m, n, alpha, dA, n-1, beta, dB, ldb, queue ), blas::Error );

    assert_throw( blas::geadd( Layout::RowMajor, Op::NoTrans,   m, n, alpha, dA, n-1, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( Layout::RowMajor, Op::Trans,     m, n, alpha, dA, m-1, beta, dB, ldb, queue ), blas::Error );
    assert_throw( blas::geadd( Layout::RowMajor, Op::ConjTrans, m, n, alpha, dA, m-1, beta, dB, ldb, queue ), blas::Error );

    assert_throw( blas::geadd( Layout::ColMajor, Op::NoTrans,   m, n, alpha, dA, lda, beta, B, m-1, queue ), blas::Error );
    assert_throw( blas::geadd( Layout::RowMajor, Op::NoTrans,   m, n, alpha, dA, lda, beta, B, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B, ldb );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    blas::geadd( layout, trans, m, n, alpha, dA, lda, beta, dB, ldb, queue );
    queue.sync();

    blas::device_copy_matrix( Bm, Bn, dB, ldb, B, ldb, queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "B2 = " ); print_matrix( Bm, Bn, B, ldb );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        cpu_geadd( layout, trans, m, n, alpha, A, lda, beta, Bref, ldb );

        if (verbose >= 2) {
            printf( "Bref = " ); print_matrix( Bm, Bn, Bref, ldb );
        }

        // check error compared to reference
        real_t error = 0;
        for (int i = 0; i < Bm; ++i)
            for (int j = 0; j < Bn; ++j) {
                real_t error_ij = std::abs( B[i + j*ldb] - Bref[i + j*ldb] )
                                / std::abs( Bref[i + j*ldb] );
                error = max( error, error_ij );
            }

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex_v<scalar_t>) {
            error /= 2*sqrt(2);
        }

        if (verbose >= 2) {
            printf( "error = %.2e\n", error );
        }

        params.error() = error;
        params.okay() = (error < tol * 16);
    }

    delete[] A;
    delete[] B;
    delete[] Bref;

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
}

//------------------------------------------------------------------------------
void test_geadd_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_geadd_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geadd_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geadd_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geadd_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
