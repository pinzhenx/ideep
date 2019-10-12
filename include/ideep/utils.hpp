#ifndef IDEEP_UTILS_CPP
#define IDEEP_UTILS_CPP

#include <string>
#include <cstring>
#include <memory>
#include <algorithm>
#include <climits>
#include <random>
#include <numeric>
#include <atomic>
#include <chrono>
#include <vector>
#include <iterator>
#ifdef IDEEP_USE_MKL
#include <mkl_vsl.h>
#include <mkl_vml_functions.h>
#endif
#include <dnnl.h>
#include <dnnl.hpp>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num()  0
#define omp_in_parallel()     0
#endif

namespace ideep {
namespace utils {

// Shallow copied vector
template <class T, class Alloc = std::allocator<T>>
class s_vector {
public:
  using size_type = typename std::vector<T, Alloc>::size_type;
  using reference = typename std::vector<T, Alloc>::reference;
  using const_reference = typename std::vector<T, Alloc>::const_reference;

  s_vector() : n_elems_(0), storage_() {}

  explicit s_vector(size_type count, const Alloc& alloc = Alloc())
    : n_elems_(count) {
    Alloc dup_alloc(alloc);

    storage_.reset(new (dup_alloc.allocate(count)) T [count] (),
       [dup_alloc, count](T* p) mutable {
      for (int i =0; i < count; i ++)
        p[i].~T();
      dup_alloc.deallocate(p, count);
    });
  }

  s_vector(std::initializer_list<T> init, const Alloc& alloc = Alloc())
    : storage_(init.size(), alloc) {
      auto arr = storage_.get();
      auto src = init.begin();
      for (int i = 0; i < init.size(); i ++)
        arr[i] = src[i];
  }

  s_vector(const s_vector& other)
    : n_elems_(other.n_elems_), storage_(other.storage_) {}

  s_vector(s_vector&& other) noexcept
    : n_elems_(other.n_elems_), storage_(std::move(other.storage_)) {}

  s_vector& operator=(const s_vector& other) {
    storage_ = other.storage_;
    n_elems_ = other.n_elems_;
    return *this;
  }

  s_vector& operator=(s_vector&& other) noexcept {
    storage_ = std::move(other.storage_);
    n_elems_ = other.n_elems_;
    return *this;
  }

  reference operator[](size_type pos) {
    return storage_.get()[pos];
  }

  const_reference operator[] (size_type pos) const {
    return storage_.get()[pos];
  }

  size_type size() const noexcept {
    return n_elems_;
  }

  void assign(size_type count, const T& val, const Alloc& alloc = Alloc()) {
    Alloc dup_alloc(alloc);

    storage_.reset(new (dup_alloc.allocate(count)) T [count] (),
       [dup_alloc, count](T* p) mutable {
      for (int i =0; i < count; i ++)
        p[i].~T();
      dup_alloc.deallocate(p, count);
    });

    auto* elems = storage_.get();
    for (int i =0; i < count; i ++)
      elems[i] = val;

    n_elems_ = count;
  }

protected:
  size_type n_elems_;
  std::shared_ptr<T> storage_;
};

static void bernoulli_generate(const long n, const double p, int* r) {
#ifndef IDEEP_USE_MKL
  IDEEP_ENFORCE(0, "can not use bernoulli_generate without MKL support");
#else
  std::srand(std::time(0));
  const int seed = 17 + std::rand() % 4096;

  int nthr = omp_get_max_threads();
#ifdef _OPENMP
  # pragma omp parallel num_threads(nthr)
#endif
  {
    const int ithr = omp_get_thread_num();
    const long avg_amount = (n + nthr - 1) / nthr;
    const long my_offset = ithr* avg_amount;
    const long my_amount = std::min(my_offset + avg_amount, n) - my_offset;

    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
#endif
}

static inline dnnl::memory::dims get_compatible_dilates(const dnnl::memory::dims& dilates) {
    if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
      auto dilates_in = dilates;
      IDEEP_STD_EACH_SUB(dilates_in, 1);
      return dilates_in;
    }
    return {0, 0};
}

static void inline validate_dims() {}

template<typename... Ts>
static void inline validate_dims(const dnnl::memory::dims& dims, Ts&... rest) {
#ifndef NDEBUG
  if (dims.size() > DNNL_MAX_NDIMS) {
    error::wrap_c_api(dnnl_invalid_arguments, "Invalid dimesions");
  }
  validate_dims(rest...);
#endif
}

template<typename T, typename U>
inline T div_up(const T a, const U b) {
    IDEEP_ENFORCE(b != 0, "divide zero in div_up");
    return(a + b - 1) / b;
}
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T& n_start, T& n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

inline void fast_memcpy(char* data_o, char* data_i, size_t len)
{
    size_t nelems = len / 4;
    size_t nelems_char = len % 4;
    const int block_size = 16;
    const auto num_blocks = nelems / block_size;
    const auto rem_elems =  nelems % block_size;
    float* output_f = (float*)data_o;
    float* input_f = (float*) data_i;
    char* output_c = (char*) data_o;
    char* input_c = (char*) data_i;
#ifdef _OPENMP
# pragma omp parallel
#endif
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        balance211(num_blocks, nthr, ithr, start, end);
        start = start * block_size;
        end = end * block_size;
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for
#endif
#endif
        for (size_t e = start; e < end; ++e) {
            output_f[e] = input_f[e];
        }
        if (rem_elems != 0 && ithr ==  nthr -1 )  {
            for (auto e = nelems - rem_elems; e < nelems; ++e) {
                output_f[e] = input_f[e];
            }
        }
        if (nelems_char != 0 && ithr ==  nthr -1){
            for (auto e = nelems*4; e < len; ++e) {
                output_c[e] = input_c[e];
            }
        }
    }
    return;
}

template<typename T>
inline void fast_memset(T* data_o, T val, size_t len)
{
    size_t nelems = len;
    const int block_size = 16;
    const auto num_blocks = nelems / block_size;
    const auto rem_elems =  nelems % block_size;
    float *output_f = (float *)data_o;
#ifdef _OPENMP
# pragma omp parallel
#endif
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{0}, end{0};
        balance211(num_blocks, nthr, ithr, start, end);
        start = start * block_size;
        end = end * block_size;
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for
#endif
#endif
        for (size_t e = start; e < end; ++e) {
            output_f[e] = val;
        }
        if (rem_elems != 0 && ithr ==  nthr -1 )  {
            for (auto e = nelems - rem_elems; e < nelems; ++e) {
                output_f[e] = val;
            }
        }
    }
    return;
}

inline dnnl::algorithm rnn_kind_to_algorithm(rnn_kind rnn) {
  if (rnn == RNN_RELU || rnn == RNN_TANH) {
    return dnnl::algorithm::vanilla_rnn;
  } else if (rnn == LSTM) {
    return dnnl::algorithm::vanilla_lstm;
  } else if (rnn == GRU) {
    return dnnl::algorithm::lbr_gru;
  } else {
    return dnnl::algorithm::undef;
  }
}

inline dnnl::algorithm rnn_kind_to_activation(rnn_kind rnn) {
  if (rnn == RNN_RELU) {
    return dnnl::algorithm::eltwise_relu;
  } else if (rnn == RNN_TANH || rnn == LSTM || rnn == GRU) {
    return dnnl::algorithm::eltwise_tanh;
  } else {
    return dnnl::algorithm::undef;
  }
}

template <typename F, typename T,
          typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T>& vec, F f) {
  std::vector<U> result;
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
  return result;
}

}
}
#endif