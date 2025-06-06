// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template <typename TX, typename TY>
void test_axpy_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::max;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    scalar_t alpha  = params.alpha.get<scalar_t>();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
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

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];
    TY* y0   = new TY[ size_y ];

    // device specifics
    blas::Queue queue( device );
    TX* dx;
    TY* dy;

    dx = blas::device_malloc<TX>( size_x, queue );
    dy = blas::device_malloc<TY>( size_y, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );
    cblas_copy( n, y, incy, y0,   incy );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::axpy( -1, alpha, x, incx, y, incy, queue ), blas::Error );
    assert_throw( blas::axpy(  n, alpha, x,    0, y, incy, queue ), blas::Error );
    assert_throw( blas::axpy(  n, alpha, x, incx, y,    0, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ),
                llong( n ), llong( incy ), llong( size_y ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::axpy( n, alpha, dx, incx, dy, incy, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< TY >::axpy( n );
    double gbyte = blas::Gbyte< TY >::axpy( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector( n, dy, abs( incy ), y, abs( incy ), queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_axpy( n, alpha, x, incx, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // maximum component-wise forward error:
        // | fl(yi) - yi | / (2 |alpha xi| + |y0_i|)
        real_t error = 0;
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = abs( y[iy] - yref[iy] )
                  / (2*(abs( alpha * x[ix] ) + abs( y0[iy] )));
            error = std::max( error, real( y[iy] ) );
            ix += incx;
            iy += incy;
        }

        if (verbose >= 2) {
            printf( "err  = " ); print_vector( n, y, incy, "%9.2e" );
        }

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex_v<scalar_t>) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < u);
    }

    delete[] x;
    delete[] y;
    delete[] yref;
    delete[] y0;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

// -----------------------------------------------------------------------------
void test_axpy_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_axpy_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_axpy_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_axpy_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_axpy_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
