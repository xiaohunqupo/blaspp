// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TX>
void test_her_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using blas::Uplo, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TX >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    real_t alpha    = params.alpha.get<real_t>();  // note: real
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    // setup
    int64_t lda = max( roundup( n, align ), 1 );
    size_t size_A = size_t(lda)*n;
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    TA* A    = new TA[ size_A ];
    TA* Aref = new TA[ size_A ];
    TX* x    = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_lacpy( "g", n, n, A, lda, Aref, lda );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lanhe( "f", to_c_string( uplo ), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // test error exits
    assert_throw( blas::her( Layout(0), uplo,     n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::her( layout,    Uplo(0),  n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::her( layout,    uplo,    -1, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::her( layout,    uplo,     n, alpha, x,    0, A, lda ), blas::Error );
    assert_throw( blas::her( layout,    uplo,     n, alpha, x, incx, A, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e;\n", alpha );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::her( layout, uplo, n, alpha, x, incx, A, lda );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::her( n );
    double gbyte = blas::Gbyte< scalar_t >::her( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_her( cblas_layout_const(layout), cblas_uplo_const(uplo),
                   n, alpha, x, incx, Aref, lda );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        bool okay;
        check_herk( uplo, n, 1, alpha, real_t(1), Xnorm, Xnorm, Anorm,
                    Aref, lda, A, lda, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] Aref;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_her( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_her_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_her_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_her_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
