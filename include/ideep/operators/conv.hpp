#ifndef IDEEP_OPERATORS_CONV_HPP
#define IDEEP_OPERATORS_CONV_HPP

namespace ideep {

static int ideep_cache_conv = -1;

struct convolution_forward : public dnnl::convolution_forward,
                             utils::computation_cache<dnnl::primitive> {
  using super = dnnl::convolution_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  // TODO: XPZ: refactor it
  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto dilates_ = utils::get_compatible_dilates(dilates);

    IDEEP_ENFORCE(
        !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    auto ic = groups * dims_in[1 + grouped];
    auto oc = groups * dims_in[0 + grouped];
    auto kh = dims_in[ndims - 2];
    auto kw = dims_in[ndims - 1];
    int mb, h, w;
    if (src_dims.empty()) {
      // Construct a dummy case
      mb = 1;
      h = 2 * kh;
      w = 4 * kw;
    } else {
      // Use the real data
      mb = src_dims[0];
      h = src_dims[2];
      w = src_dims[3];
    }
    auto oh = (h - ((kh - 1) * (dilates_[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
    auto ow = (w - ((kw - 1) * (dilates_[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

    dims x_dims = { mb, ic, h, w };
    dims y_dims = { mb, oc, oh, ow };
    auto y_dtype = dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is
    // required by winograd primitive. Then, in the cases of variable input
    // shape, the detials of dnnl_wino_fmt will be changed. And, extra weihgts
    // reorder is inevitable each time, leading to bad performance. Here, we set
    // the prop_kind to forward, in order to reorder and cache weights as
    // blocked format, instead of dnnl_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd &&
        aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
        padding_l, padding_r, attr_t(), aalgorithm, apkind);

    // embed group info into weights_desc
    return tensor::desc(pd.weights_desc(), groups);
  }

  template <bool with_bias>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_any = src_desc.to_format_any();
    auto weights_desc_any = weights_desc.to_format_any();
    auto bias_desc_any = with_bias ? bias_desc.to_format_any() : tensor::desc();
    auto dst_desc_any = dst_desc.to_format_any();

    if (with_bias) {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, bias_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    } else {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    }
  }

private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           const dims& dst_dims,
                           tensor& dst,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           int groups,
                           const attr_t& attr,
                           algorithm aalgorithm,
                           prop_kind aprop_kind,
                           const engine& aengine) {

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto dst_desc = attr.has_op_kind(kind::sum)
                        ? dst.get_desc()
                        : tensor::desc(dst_dims, src.get_data_type());

    auto src_desc = src.get_desc();
    auto weights_desc = weights_.get_desc();
    auto bias_desc = bias.get_desc();

    if (ideep_cache_conv < 0) {
      ideep_cache_conv = atoi(std::getenv("IDEEP_CACHE_CONV")) > 0 ? 1 : 0;
      std::cout << (ideep_cache_conv ? "ideep" : "dnnl") << " caches conv\n";
    }

    if (ideep_cache_conv) {
      auto key = utils::create_key(src_desc, weights_desc, with_bias, strides,
                                 dilates_, padding_l, padding_r, attr);
      auto comp = fetch_or_create(key, [&]() {
        auto pd = get_primitive_desc<with_bias>(
            src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
            padding_l, padding_r, attr, aalgorithm, aprop_kind, aengine);
        return super(pd);
      });
      auto pd = conv_pd_wrapper(comp);

      auto expected_src = src.reorder_if_differ_in(pd.src_desc());
      auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
      dst.reinit_if_possible(pd.dst_desc());

      if (with_bias) {
        auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
        comp.execute(stream::default_stream(), 
                     {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_BIAS, expected_bias},
                      {DNNL_ARG_DST, dst}});
      } else {
        comp.execute(stream::default_stream(), 
                     {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_DST, dst}});
      }
    }

    // dnnl cache conv
    else {
      auto pd = get_primitive_desc<with_bias>(
          src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
          padding_l, padding_r, attr, aalgorithm, aprop_kind, aengine);
      auto comp = super(pd);

      auto expected_src = src.reorder_if_differ_in(pd.src_desc());
      auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
      dst.reinit_if_possible(pd.dst_desc());

      if (with_bias) {
        auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
        comp.execute(stream::default_stream(), 
                     {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_BIAS, expected_bias},
                      {DNNL_ARG_DST, dst}});
      } else {
        comp.execute(stream::default_stream(), 
                     {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_WEIGHTS, expected_weights},
                      {DNNL_ARG_DST, dst}});
      }
    }
  }
};


struct convolution_backward_data : public dnnl::convolution_backward_data {

  using super = dnnl::convolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();

    tensor::desc diff_src_desc(diff_src_dims, diff_dst_desc.get_data_type());

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(), 
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};


struct convolution_backward_weights
    : public dnnl::convolution_backward_weights {

  using super = dnnl::convolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

 private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);
    auto diff_weights_desc_usr =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type());
    auto diff_weights_desc = groups > 1
        ? diff_weights_desc_usr.to_format_any().to_grouped(groups)
        : diff_weights_desc_usr.to_format_any();

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto src_desc = src.get_desc().to_format_any();

    auto diff_bias_desc =
        tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type(), tag::any);

    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    diff_weights.reinit_if_possible(pd.diff_weights_desc());

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                         {DNNL_ARG_DIFF_BIAS, diff_bias}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});
    }

    if (groups > 1) {
      diff_weights.to_format(format_tag::abcde);
      diff_weights.set_desc(diff_weights_desc_usr);
    }
  }
};
}  // namespace ideep

#endif
