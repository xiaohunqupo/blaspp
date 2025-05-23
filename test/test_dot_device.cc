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
template <typename TX, typename TY>
void test_dot_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::max;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t verbose = params.verbose();
    int64_t device  = params.device();
    char mode       = params.pointer_mode();
    bool use_dot    = params.routine == "dot";

    scalar_t  result_host;
    scalar_t* result_ptr = &result_host;

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

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );
    TX* x = new TX[ size_x ];
    TY* y = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, abs( incx ) );
    real_t Ynorm = cblas_nrm2( n, y, abs( incy ) );

    // device specifics
    blas::Queue queue( device );
    TX* dx;
    TY* dy;

    dx = blas::device_malloc<TX>( size_x, queue );
    dy = blas::device_malloc<TY>( size_y, queue );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    if (mode == 'd') {
        result_ptr = blas::device_malloc<scalar_t>( 1, queue );
        #if defined( BLAS_HAVE_CUBLAS )
            cublasSetPointerMode( queue.handle(), CUBLAS_POINTER_MODE_DEVICE );
        #elif defined( BLAS_HAVE_ROCBLAS )
            rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

    // test error exits
    if (use_dot) {
        assert_throw( blas::dot( -1, dx, incx, dy, incy, result_ptr, queue ), blas::Error );
        assert_throw( blas::dot(  n, dx,    0, dy, incy, result_ptr, queue ), blas::Error );
        assert_throw( blas::dot(  n, dx, incx, dy,    0, result_ptr, queue ), blas::Error );
    }
    else {
        assert_throw( blas::dotu( -1, dx, incx, dy, incy, result_ptr, queue ), blas::Error );
        assert_throw( blas::dotu(  n, dx,    0, dy, incy, result_ptr, queue ), blas::Error );
        assert_throw( blas::dotu(  n, dx, incx, dy,    0, result_ptr, queue ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n",
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    if (use_dot) {
        blas::dot( n, dx, incx, dy, incy, result_ptr, queue );
    }
    else {
        blas::dotu( n, dx, incx, dy, incy, result_ptr, queue );
    }
    queue.sync();
    time = get_wtime() - time;

    if (mode == 'd') {
        device_memcpy( &result_host, result_ptr, 1, queue );
    }

    double gflop = blas::Gflop<scalar_t>::dot( n );
    double gbyte = blas::Gbyte<scalar_t>::dot( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 1) {
        printf( "dot = %.4e + %.4ei\n", real(result_host), imag(result_host) );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        scalar_t ref;
        if (use_dot) {
            ref = cblas_dot( n, x, incx, y, incy );
        }
        else {
            ref = cblas_dotu( n, x, incx, y, incy );
        }
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "ref = %.4e + %.4ei\n", real(ref), imag(ref) );
        }

        // check error compared to reference
        // treat result as 1 x 1 matrix; k = n is reduction dimension
        // alpha=1, beta=0, Cnorm=0
        real_t error;
        bool okay;
        check_gemm( 1, 1, n, scalar_t(1), scalar_t(0), Xnorm, Ynorm, real_t(0),
                    &ref, 1, &result_host, 1, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] x;
    delete[] y;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
    if (mode == 'd')
        blas::device_free( result_ptr, queue );
}

// -----------------------------------------------------------------------------
void test_dot_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_dot_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_dot_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_dot_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_dot_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
