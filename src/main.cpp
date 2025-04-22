#include <cassert>
#include <cstdlib>
#include <chrono>

#include <Kokkos_Core.hpp>
#include <fmt/core.h>

using MatrixL = Kokkos::View<double**, Kokkos::LayoutLeft>;
using MatrixR = Kokkos::View<double**, Kokkos::LayoutRight>;

template <class MatrixType>
auto matrix_init(MatrixType& M) -> void {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  Kokkos::parallel_for(
    "init",
    M.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(M.extent(1)); ++j) {
        M(i, j) = drand48();
      }
    }
  );
}

// Classic version
template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  Kokkos::parallel_for(
    "dgemm_kernel",
    A.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(B.extent(1)); ++j) {
        double acc = 0.0;
        for (int k = 0; k < int(A.extent(1)); ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
        C(i, j) *= beta + acc;
      }
    }
  );
}


// Cache blocked version
template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product_cache_blocking(double alpha, const AMatrixType A, const BMatrixType B, double beta, CMatrixType C, int block_size) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0)); // M
  assert(B.extent(1) == C.extent(1)); // N
  assert(A.extent(1) == B.extent(0)); // K

  Kokkos::parallel_for(
    "dgemm_cache_blocked",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {A.extent(0), B.extent(1)}, {block_size, block_size}),
    KOKKOS_LAMBDA(int i, int j) {
      double acc = 0.0;
      for (int kk = 0; kk < (int)A.extent(1); kk += block_size) {
        int fin;
        
        // calcul de fin de block
        if (kk + block_size > (int)A.extent(1)) {
          fin = A.extent(1);
        }
        else {
          fin = kk + block_size;
        }

        for (int k = kk; k < fin; ++k) {
          acc += alpha * A(i, k) * B(k, j);
        }
      }
      C(i, j) *= beta + acc;
    }
  );
}

auto main(int argc, char* argv[]) -> int {
  if (argc < 4) {
    fmt::print("Usage: {} <M> <N> <K>\n", argv[0]);
    return -1;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);

  // Known seed for deterministic RNG
  srand48(42);

  Kokkos::initialize(argc, argv);
  {
    auto A = MatrixR("A", m, k);
    auto B = MatrixL("B", k, n);
    auto C = MatrixR("C", m, n);
    auto C_ref = MatrixR("C_ref", m, n);
    auto C_blocked = MatrixR("C_blocked", m, n);

    double alpha = drand48();
    matrix_init(A);
    matrix_init(B);
    double beta = drand48();
    matrix_init(C);
    Kokkos::deep_copy(C_ref, C);

    Kokkos::fence();
    auto start = std::chrono::high_resolution_clock::now();

    matrix_product(alpha, A, B, beta, C_ref);

    double duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    Kokkos::deep_copy(C_blocked, C);
    Kokkos::fence();
    start = std::chrono::high_resolution_clock::now();

    matrix_product_cache_blocking(alpha, A, B, beta, C_blocked, 16);

    double duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    double difference = 1e-10;
    double diff = 0;

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        diff = std::abs(C_ref(i, j) - C_blocked(i, j));
        if (diff > difference) {
          break;
        }
      }
    }


    // Testing on different sizes

    Kokkos::deep_copy(C_blocked, C);
    Kokkos::fence();
    start = std::chrono::high_resolution_clock::now();

    matrix_product_cache_blocking(alpha, A, B, beta, C_blocked, 32);

    double duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    Kokkos::deep_copy(C_blocked, C);
    Kokkos::fence();
    start = std::chrono::high_resolution_clock::now();

    matrix_product_cache_blocking(alpha, A, B, beta, C_blocked, 64);

    double duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    Kokkos::deep_copy(C_blocked, C);
    Kokkos::fence();
    start = std::chrono::high_resolution_clock::now();

    matrix_product_cache_blocking(alpha, A, B, beta, C_blocked, 128);

    double duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    Kokkos::deep_copy(C_blocked, C);
    Kokkos::fence();
    start = std::chrono::high_resolution_clock::now();

    matrix_product_cache_blocking(alpha, A, B, beta, C_blocked, 256);

    double duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::high_resolution_clock::now() - start).count();

    Kokkos::fence();

    fmt::println("{}, {}, {}, {}, {}, {}, {}, {}", Kokkos::num_threads(), duration1, duration2, duration3, duration4, duration5, duration6, diff);
  }
  Kokkos::finalize();
  return 0;
}
