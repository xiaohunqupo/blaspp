#include "blas/device.hh"
#include "operators.cuh"

#if defined(BLAS_HAVE_CUBLAS)

namespace blas {

// Each thread conjugates 1 item
template <typename scalar_t>
__global__ void conj_kernel(
    int64_t n,
    scalar_t const* x, int64_t incx, int64_t ix,
    scalar_t*       y, int64_t incy, int64_t iy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[ i*incy + iy ] = conj_device( x[ i*incx + ix ] );
}

//------------------------------------------------------------------------------
/// Conjugates each element of the vector x and stores in y.
///
/// @param[in] n
///     Number of elements in the vector. n >= 0.
///
/// @param[in] x
///     Pointer to the input vector of length n.
///
/// @param[in] incx
///     Stride between elements of x. incx >= 1.
///
/// @param[out] y
///     Pointer to output vector
///     On exit, each element y[i] is updated as y[i] = conj( x[i] ).
///     y may be the same as x.
///
/// @param[in] incy
///     Stride between elements of y. incy >= 1.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue )
{
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (n == 0)
        return;

    const int64_t BlockSize = 1024;

    int64_t n_threads = min( BlockSize, n );
    int64_t n_blocks = ceildiv(n, n_threads);

    int64_t ix = (incx > 0 ? 0 : (1 - n) * incx);
    int64_t iy = (incy > 0 ? 0 : (1 - n) * incy);

    blas_dev_call( cudaSetDevice( queue.device() ) );

    if constexpr (std::is_same_v<scalar_t, std::complex<float>>) {
        conj_kernel<cuComplex><<<n_blocks, n_threads, 0, queue.stream()>>>(
            n, (cuComplex*) x, incx, ix, (cuComplex*) y, incy, iy );
    }
    else if constexpr (std::is_same_v<scalar_t, std::complex<double>>) {
        conj_kernel<cuDoubleComplex><<<n_blocks, n_threads, 0, queue.stream()>>>(
            n, (cuDoubleComplex*) x, incx, ix, (cuDoubleComplex*) y, incy, iy );
    }
    else {
    conj_kernel<<<n_blocks, n_threads, 0, queue.stream()>>>(
        n, x, incx, ix, y, incy, iy );
    }

    blas_dev_call( cudaGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template void conj(
    int64_t n,
    float const* x, int64_t incx,
    float* y, int64_t incy,
    blas::Queue& queue);

template void conj(
    int64_t n,
    double const* x, int64_t incx,
    double* y, int64_t incy,
    blas::Queue& queue);

template void conj(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue);

template void conj(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue);

} // namespace blas

#endif // BLAS_HAVE_CUBLAS
