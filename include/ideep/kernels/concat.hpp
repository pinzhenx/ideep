#ifndef IDEEP_KERNELS_CONCAT_HPP
#define IDEEP_KERNELS_CONCAT_HPP

#include "common.hpp"

namespace ideep {

struct concat : public dnnl::concat {
  static void compute(std::vector<tensor>& inputs, int axis, tensor& output) {
  }

  static std::vector<int32_t> compute(std::vector<tensor>& inputs, int axis, bool add_axis, tensor& dst) {
  }
};

}  // namespace ideep

#endif