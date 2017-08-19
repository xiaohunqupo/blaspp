#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack.hh"
#include "flops.hh"
#include "check_gemm.hh"

#include "syr2k.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC >
void test_syr2k_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits2< TA, TC >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::norm_t norm_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op trans  = params.trans.value();
    blas::Uplo uplo = params.uplo.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if ( ! run)
        return;

    // setup
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Am, align );
    int64_t ldc = roundup(  n, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*An;
    size_t size_C = size_t(ldc)*n;
    TA* A    = new TA[ size_A ];
    TB* B    = new TB[ size_B ];
    TC* C    = new TC[ size_C ];
    TC* Cref = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    // norms for error check
    norm_t work[1];
    norm_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    norm_t Bnorm = lapack_lange( "f", Am, An, B, ldb, work );
    norm_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );

    if (verbose >= 1) {
        printf( "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%5lld, norm %.2e\n"
                "B Bn=%5lld, Bn=%5lld, ldb=%5lld, size=%5lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%5lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                (lld) Am, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld) Am, (lld) An, (lld) ldb, (lld) size_B, Bnorm,
                (lld)  n, (lld)  n, (lld) ldc, (lld) size_C, Cnorm );
    }
    if (verbose >= 2) {
        printf( "A = "    ); //print_matrix( Am, An, A, lda );
        printf( "B = "    ); //print_matrix( Am, An, B, ldb );
        printf( "C = "    ); //print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    blas::syr2k( layout, uplo, trans, n, k,
                 alpha, A, lda, B, ldb, beta, C, ldc );
    time = omp_get_wtime() - time;

    double gflop = gflop_syr2k( n, k, C );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); //print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_syr2k( cblas_layout_const(layout),
                     cblas_uplo_const(uplo),
                     cblas_trans_const(trans),
                     n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); //print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        norm_t error;
        int64_t okay;
        check_herk( uplo, n, 2*k, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
}

// -----------------------------------------------------------------------------
void test_syr2k( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_syr2k_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_syr2k_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr2k_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_syr2k_work< std::complex<float>, std::complex<float>,
                             std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_syr2k_work< std::complex<double>, std::complex<double>,
                             std::complex<double> >( params, run );
            break;
    }
}