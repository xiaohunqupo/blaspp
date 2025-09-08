#include "blas/device.hh"
#include "operators.hip.h"

#if defined(BLAS_HAVE_ROCBLAS)

namespace blas {

//------------------------------------------------------------------------------
// Each thread adds 1 item.
template <typename scalar_t>
__global__ void tzadd_kernel(
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    // Assume matrices are column major. Row major is handled in driver.
    int64_t j_lower = (uplo == Uplo::Lower ? 0 : max_device( 0, i - max_device( 0, m - n ) ));
    int64_t j_upper = (uplo == Uplo::Upper ? n : min_device( n, i + 1 ));

    if (i < m && j >= j_lower && j < j_upper) {
        scalar_t A_op;
        switch (trans) {
            case Op::NoTrans:   A_op = A[ i + j*lda ]; break;
            case Op::Trans:     A_op = A[ j + i*lda ]; break;
            case Op::ConjTrans: A_op = conj_device( A[ j + i*lda ] ); break;
            default: return; // Invalid operation
        }
        B[ i + j*ldb ] = beta * B[ i + j*ldb ] + alpha * A_op;
    }
}

//------------------------------------------------------------------------------
/// Trapezoidal matrix addition:
/// \[
///     B = beta*B + alpha*op( A )
/// \]
/// where $op(X)$ is one of
///     $op(X) = X$,
///     $op(X) = X^T$, or
///     $op(X) = X^H$,
/// alpha and beta are scalars, and op( A ) and B are m-ny-n matrices.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///     What part of the matrices is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: op(A) and B are lower trapezoidal.
///     - Uplo::Upper: op(A) and B are upper trapezoidal.
///     - Uplo::General is illegal (see @ref geadd instead).
///
/// @param[in] trans
///     The operation $op(A)$ to be used:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] m
///     Number of rows of the matrix $op(A)$ and $B$. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix $op(A)$ and $B$. m >= 0.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///     - Otherwise:
///       the n-by-m matrix A, stored in an lda-by-m array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, n)].
///     - Otherwise:           lda >= max(1, n) [RowMajor: lda >= max(1, m)].
///
/// @param[in, out] B
///     - If transB = NoTrans:
///       the m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///     - Otherwise:
///       the n-by-m matrix B, stored in an ldb-by-m array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, m)].
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
    using dev_scalar_t = typename device_cast_traits< scalar_t >::type;

    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor) {
        if (trans == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < n );

        blas_error_if( ldb < m );
    }
    else {
        if (trans == Op::NoTrans)
            blas_error_if( lda < n );
        else
            blas_error_if( lda < m );

        blas_error_if( ldb < n );
    }

    if (m == 0 || n == 0)
        return;

    if (layout == Layout::RowMajor) {
        std::swap(m, n);
    }

    const int64_t Block_x = 32;
    const int64_t Block_y = 32;

    dim3 threads( min( Block_x, m ), min( Block_y, n ) );
    dim3 blocks( ceildiv( m, Block_x ), ceildiv( n, Block_y ) );

    blas_dev_call( hipSetDevice( queue.device() ) );

    // Cast complex types to hipFloatComplex or hipDoubleComplex.
    tzadd_kernel<<<blocks, threads, 0, queue.stream()>>>(
        uplo, trans, m, n,
        *((dev_scalar_t*) &alpha), (dev_scalar_t*) A, lda,
        *((dev_scalar_t*) &beta),  (dev_scalar_t*) B, ldb );

    blas_dev_call( hipGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.

template void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha, float const* A, int64_t lda,
    float beta,  float*       B, int64_t ldb,
    blas::Queue& queue );

template void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha, double const* A, int64_t lda,
    double beta,  double*       B, int64_t ldb,
    blas::Queue& queue );

template void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha, std::complex<float> const* A, int64_t lda,
    std::complex<float> beta,  std::complex<float>*       B, int64_t ldb,
    blas::Queue& queue );

template void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha, std::complex<double> const* A, int64_t lda,
    std::complex<double> beta,  std::complex<double>*       B, int64_t ldb,
    blas::Queue& queue );

} // namespace blas

#endif // BLAS_HAVE_ROCBLAS
