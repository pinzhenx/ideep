#ifndef IDEEP_OPERATORS_PD_WRAPPER_HPP
#define IDEEP_OPERATORS_PD_WRAPPER_HPP

namespace ideep {

struct pd_wrapper {
  const_dnnl_primitive_desc_t pd_;

  pd_wrapper(const dnnl::primitive& prim) : pd_(prim.get_primitive_desc()) {}

  tensor::desc query_md(query what, int idx = 0) const {
    std::vector<query> valid_q{query::src_md,       query::diff_src_md,
                               query::weights_md,   query::diff_weights_md,
                               query::dst_md,       query::diff_dst_md,
                               query::workspace_md, query::scratchpad_md};
    if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                     [=](query q) { return what == q; }))
      throw error(dnnl_invalid_arguments, "invalid memory query");

    const dnnl_memory_desc_t* cdesc =
        dnnl_primitive_desc_query_md(pd_, dnnl::convert_to_c(what), idx);
    return tensor::desc(*cdesc);
  }
};

struct conv_pd_wrapper : public pd_wrapper {
  using pd_wrapper::pd_wrapper;
  tensor::desc src_desc() const {
    return query_md(query::src_md, 0);
  }
  tensor::desc weights_desc() const {
    return query_md(query::weights_md, 0);
  }
  tensor::desc bias_desc() const {
    return query_md(query::weights_md, 1);
  }
  tensor::desc dst_desc() const {
    return query_md(query::dst_md, 0);
  }
  tensor::desc diff_src_desc() const {
    return query_md(query::diff_src_md, 0);
  }
  tensor::desc diff_weights_desc() const {
    return query_md(query::diff_weights_md, 0);
  }
  tensor::desc diff_bias_desc() const {
    return query_md(query::diff_weights_md, 1);
  }
  tensor::desc diff_dst_desc() const {
    return query_md(query::diff_dst_md, 0);
  }
};

struct bn_pd_wrapper_base : public pd_wrapper {
  using pd_wrapper::pd_wrapper;
  tensor::desc src_desc() const {
    return query_md(query::src_md, 0);
  }
  tensor::desc weights_desc() const {
    return query_md(query::weights_md, 0);
  }
  tensor::desc workspace_desc() const {
    return query_md(query::workspace_md, 0);
  }
  tensor::desc dst_desc() const {
    return query_md(query::dst_md, 0);
  }
  tensor::desc diff_src_desc() const {
    return query_md(query::diff_src_md, 0);
  }
  tensor::desc diff_weights_desc() const {
    return query_md(query::diff_weights_md, 0);
  }
  tensor::desc diff_dst_desc() const {
    return query_md(query::diff_dst_md, 0);
  }
};

struct bn_fwd_pd_wrapper : public bn_pd_wrapper_base {
  using bn_pd_wrapper_base::bn_pd_wrapper_base;
  tensor::desc mean_desc() const {
    return stat_desc(mean);
  }
  tensor::desc variance_desc() const {
    return stat_desc(var);
  }
 private:
  enum { mean = 1, var = 2, };
  tensor::desc stat_desc(int kind) const {
    dnnl_batch_normalization_desc_t* p;
    error::wrap_c_api(
        dnnl_primitive_desc_query(
            pd_, dnnl::convert_to_c(query::batch_normalization_d), 0, &p),
        "could not get a batch-normalization descriptor");
    return query_md(
        p->flags & dnnl_use_global_stats ? query::src_md : query::dst_md, kind);
  }
};

struct bn_bwd_pd_wrapper : public bn_pd_wrapper_base {
  using bn_pd_wrapper_base::bn_pd_wrapper_base;
  tensor::desc mean_desc() const {
    return query_md(query::src_md, 1);
  }
  tensor::desc variance_desc() const {
    return query_md(query::src_md, 2);
  }
};

}
#endif