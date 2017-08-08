#include "geru.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_geru_work()
{
    int64_t m = 200;
    int64_t lda = m;
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[m];
    T *y = new T[n];
    T alpha = 123;

    blas::geru( blas::Layout::ColMajor, m, n,
                alpha, x, incx, y, incy, A, lda );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_geru()
{
    printf( "\n%s\n", __func__ );
    test_geru_work< int >();
    test_geru_work< float >();
    test_geru_work< double >();
    test_geru_work< std::complex<float> >();
    test_geru_work< std::complex<double> >();
}