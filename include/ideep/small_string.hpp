#include <cstring>
#include <string>

namespace ideep {
namespace utils {

template <size_t init_size = 128>
class small_string {
public:
  small_string() : len_(0), limit_(init_size), bufptr_(storage_) {}

  small_string(const small_string& rhs) : len_(rhs.len_), limit_(rhs.limit_) {
    bufptr_ = rhs.is_using_internal_storage() ? storage_ : new char[limit_];
    std::memcpy(bufptr_, rhs.bufptr_, len_);
  }

  small_string(small_string&& rhs) : len_(rhs.len_), limit_(rhs.limit_) {
    if (!rhs.is_using_internal_storage()) {
      bufptr_ = rhs.bufptr_;
      rhs.bufptr_ = nullptr;
    } else {
      bufptr_ = storage_;
      std::memcpy(bufptr_, rhs.bufptr_, len_);
    }
  }

  small_string& operator=(const small_string& rhs) {
    if (this == &rhs) return *this;
    limit_ = rhs.limit_;
    len_ = rhs.len_;
    if (!this->is_using_internal_storage()) delete[] bufptr_;
    bufptr_ = rhs.is_using_internal_storage() ? storage_ : new char[limit_];
    std::memcpy(bufptr_, rhs.bufptr_, len_);
    return *this;
  }

  small_string& operator=(small_string&& rhs) {
    if (this == &rhs) return *this;
    limit_ = rhs.limit_;
    len_ = rhs.len_;
    if (!this->is_using_internal_storage()) delete[] bufptr_;
    if (!rhs.is_using_internal_storage()) {
      bufptr_ = rhs.bufptr_;
      rhs.bufptr_ = nullptr;
    } else {
      bufptr_ = storage_;
      std::memcpy(bufptr_, rhs.bufptr_, len_);
    }
    return *this;
  }

  virtual ~small_string() {
    if (!is_using_internal_storage()) delete[] bufptr_;
  }

  size_t size() const noexcept { return len_; }

  size_t length() const noexcept { return len_; }

  size_t capacity() const noexcept { return limit_; };

  bool empty() const noexcept { return len_ == 0; }

  void clear() noexcept { len_ = 0; }

  const char& operator[](size_t i) const { return bufptr_[i]; }

  bool operator==(const small_string& rhs) const {
    return !std::memcmp(bufptr_, rhs.bufptr_, len_);
  }

  bool operator!=(const small_string& rhs) const { return !(*this == rhs); }

  inline void append(size_t n, char c) {
    request_nchar(n);
    for (size_t i = 0; i < n; i++) bufptr_[len_++] = c;
  }

  inline void append(const char* s, size_t n) {
    request_nchar(n);
    std::memcpy(bufptr_ + len_, s, n);
    len_ += n;
  }

  inline char pop_back() { return len_ > 0 ? bufptr_[--len_] : 0; }

  inline size_t hashcode() const {
    size_t hash = 17;
    int word_size = sizeof(size_t);
    int block_num = len_ / word_size;
    int bytes_left = len_ % word_size;
    auto block_ptr = reinterpret_cast<size_t*>(bufptr_);
    size_t i;
    for (i = 0; i < block_num; i++) hash = hash * 31 + block_ptr[i];
    if (!bytes_left) return hash;

    // TODO: determine byte order (suppose little-endian for now)
    // Truncate tailing unused bytes, i.e. clear high bits in a word
    size_t final_word = block_ptr[i];
    int highbits = 8 * (word_size - bytes_left);
    final_word = final_word << highbits >> highbits;
    return hash * 31 + final_word;
  }

 private:
  inline bool is_using_internal_storage() const { return bufptr_ == storage_; }

  inline void request_nchar(size_t n) {
    if (len_ + n > limit_) grow();
  }

  void grow() {
    limit_ *= 2;
    auto newbuf = new char[limit_];
    std::memcpy(newbuf, bufptr_, len_);
    if (!is_using_internal_storage()) delete[] bufptr_;
    bufptr_ = newbuf;
  }

  char storage_[init_size];
  size_t len_;
  size_t limit_;
  char* bufptr_;
};

struct small_string_hasher {
  size_t operator()(const small_string<>& s) const { return s.hashcode(); }
};

}  // namespace utils
}  // namespace ideep

namespace std {

template <>
struct hash<ideep::utils::small_string<>> {
  std::size_t operator()(const ideep::utils::small_string<>& s) const {
    return s.hashcode();
  }
};

}  // namespace std