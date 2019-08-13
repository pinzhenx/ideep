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
#include <mkl_vsl.h>
#include <mkl_vml_functions.h>
#include <mkldnn.h>
#include <mkldnn.hpp>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num()  0
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

using bytestring = std::string; // typename utils::small_string<>;

inline void to_bytes(bytestring& bytes, const int arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
#ifndef __AVX__
  if (arg == 0) return;
  auto len = sizeof(arg) - (__builtin_clz(arg) / 8);
#else
  unsigned int lz;
  asm volatile ("lzcntl %1, %0": "=r" (lz): "r" (arg));
  auto len = sizeof(int) - lz / 8;
#endif
  bytes.append(as_cstring, len);
}

inline void to_bytes(bytestring& bytes, const bool arg) {
  to_bytes(bytes, arg ? 1 : 0);
  bytes.append(1, 'x');
}

inline void to_bytes(bytestring& bytes, const float arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(float));
}

inline void to_bytes(bytestring& str, const uint64_t arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  str.append(as_cstring, sizeof(uint64_t));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>& arg) {
  if (arg.size() > 0) {
    for (T elems : arg) {
      to_bytes(bytes, elems);
      bytes.append(1, 'x');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'x');
  }
}

template <typename T>
inline void to_bytes(bytestring& bytes, const std::vector<T>& arg) {
  // remove constness, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, const_cast<std::vector<T>&>(arg));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>&& arg) {
  // `arg` is an lval ref now, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, arg);
}

template <typename T,
          typename = typename std::enable_if<std::is_enum<T>::value>::type>
inline void to_bytes(bytestring& bytes, T arg) {
  to_bytes(bytes, static_cast<int>(arg));
}

template <typename T,
          typename = typename std::enable_if<std::is_class<T>::value>::type,
          typename = void>
inline void to_bytes(bytestring& bytes, const T arg) {
  arg.to_bytes(bytes);
}

template <typename T, typename ...Ts>
inline void to_bytes(bytestring& bytes, T&& arg, Ts&&... args) {
  to_bytes(bytes, std::forward<T>(arg));
  bytes.append(1, '*');
  to_bytes(bytes, std::forward<Ts>(args)...);
}

template <typename ...Ts>
inline void create_key(key_t& key_to_create, Ts&&... args) {
  to_bytes(key_to_create, std::forward<Ts>(args)...);
}

#define check_or_create_k(key, ...) \
  if (key.empty()) { utils::create_key(key, __VA_ARGS__); }

static void bernoulli_generate(const long n, const double p, int* r) {
  std::srand(std::time(0));
  const int seed = 17 + std::rand() % 4096;

  int nthr = omp_get_max_threads();
  # pragma omp parallel num_threads(nthr)
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
}

static inline mkldnn::memory::dims get_compatible_dilates(const mkldnn::memory::dims& dilates) {
    if (!dilates.empty() && !IDEEP_STD_ANY_LE(dilates, 0)) {
      auto dilates_in = dilates;
      IDEEP_STD_EACH_SUB(dilates_in, 1);
      return dilates_in;
    }
    return {0, 0};
}

static void inline validate_dims() {}

template<typename... Ts>
static void inline validate_dims(const mkldnn::memory::dims& dims, Ts&... rest) {
#ifndef NDEBUG
  if (dims.size() > TENSOR_MAX_DIMS) {
    error::wrap_c_api(mkldnn_invalid_arguments, "Invalid dimesions");
  }
  validate_dims(rest...);
#endif
}

}
}
#endif
