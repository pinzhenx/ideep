#ifndef IDEEP_KERNELS_SUM_HPP
#define IDEEP_KERNELS_SUM_HPP

#include "common.hpp"

namespace ideep {

struct sum : public dnnl::sum {
  static void compute(const scale_t& scales,
                      const std::vector<tensor>& inputs,
                      tensor& output,
                      const engine& aengine = engine::cpu_engine()) {
    auto input_descs = utils::fmap(inputs, [](const tensor& t) {
      // We cannot upcast vector<tensor::desc> to vector<dnnl::memory::desc>
      // even tensor::desc inherits memory::desc. So we use static_cast here
      return static_cast<dnnl::memory::desc>(t.get_desc());
    });
    auto pd = dnnl::sum::primitive_desc(scales, input_descs, aengine);

    output.reinit_if_necessary(pd.dst_desc());

    std::unordered_map<int, dnnl::memory> args {{DNNL_ARG_DST, output}};
    for (int i = 0; i < inputs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs[i]});
    }

    dnnl::sum(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif