#ifndef IDEEP_OPERATORS_SUM_HPP
#define IDEEP_OPERATORS_SUM_HPP

namespace ideep {

struct sum : public dnnl::sum,
             utils::computation_cache<dnnl::sum> {

  using super = dnnl::sum;

  static void compute(const scale_t& scales,
                      const std::vector<tensor>& srcs,
                      tensor& dst,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_descs = utils::fmap(srcs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });
    auto key = utils::create_key(scales, src_descs);
    auto comp = fetch_or_create(key, [&]() {
      auto pd = primitive_desc(scales, src_descs, aengine);
      return super(pd);
    });
    auto pd = utils::get_pd(comp);

    dst.reinit_if_possible(pd.dst_desc());

    exec_args args {{DNNL_ARG_DST, dst}};
    for (int i = 0; i < srcs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs[i]});
    }

    comp.execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif