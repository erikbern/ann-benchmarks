// Run-local comparison adapter: independent QG query on the same immutable
// legacy corpus storage, so representation/graph changes cannot confound
// timing.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/interface/index_factory.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include "algorithm/quantized_graph/quantized_graph_query.h"
#include "algorithm/quantized_graph/quantized_graph_router_io.h"
namespace py = pybind11;
namespace ci = zvec::core_interface;
namespace core = zvec::core;
using Array = py::array_t<float, py::array::c_style>;
static void check(int rc) {
  if (rc) throw std::runtime_error("zvec error " + std::to_string(rc));
}
class StorageSource : public ci::HNSWIndex {
 public:
  using ci::Index::init;
  core::IndexStorage::Pointer storage() {
    return storage_;
  }
  core::IndexContext::Pointer context() {
    return streamer_->create_context();
  }
  const core::IndexMeta &meta() {
    return streamer_->meta();
  }
  int prepare(const float *q, std::string *buffer, const void **out) {
    core::IndexQueryMeta m;
    return _prepare_dense_query({ci::DenseVector{q}}, buffer, out, &m);
  }
  core::IndexMetric::MatrixDistance distance() {
    if (turbo_quantizer_) {
      auto quantizer = turbo_quantizer_;
      return [quantizer](const void *v, const void *q, size_t, float *out) {
        *out = quantizer->calc_distance_dp_query(v, q);
      };
    }
    auto metric = core::IndexFactory::CreateMetric(meta().metric_name());
    if (!metric) throw std::runtime_error("missing metric");
    check(metric->init(meta(), meta().metric_params()));
    return metric->distance();
  }
};
// Internal linkage preserves traversal inlining, as in the original entry.
namespace {
struct Provider {
  using Vector = core::IndexStorage::MemoryBlock;
  const core::HnswEntity &source;
  bool valid(uint32_t id) const {
    return source.get_key(id) != core::kInvalidKey;
  }
  int get_vector(uint32_t id, Vector &v) const {
    return source.get_vector(id, v);
  }
  core::Neighbors neighbors(uint32_t id) const {
    return source.get_neighbors(0, id);
  }
};
}  // namespace
class Index {
  StorageSource source_;
  core::IndexContext::Pointer owner_;
  const core::HnswEntity *entity_ = nullptr;
  std::shared_ptr<core::QuantizedGraphIndex> graph_;
  std::unique_ptr<core::QuantizedGraphQuery> query_;
  core::QuantizedGraphSearchOptions options_;
  size_t dimension_, capacity_ = 2048;

 public:
  Index(const std::string &path, const std::string &config, int dim)
      : dimension_(dim) {
    auto params = ci::IndexFactory::DeserializeIndexParamFromJson(config);
    auto hnsw = std::dynamic_pointer_cast<ci::HNSWIndexParam>(params);
    if (!hnsw) throw std::runtime_error("invalid config");
    check(source_.init(*hnsw));
    ci::StorageOptions storage;
    storage.type = ci::StorageOptions::StorageType::kMMAP;
    storage.read_only = true;
    storage.create_new = false;
    check(source_.open(path, storage));
    owner_ = source_.context();
    auto context = dynamic_cast<core::HnswContext *>(owner_.get());
    if (!context) throw std::runtime_error("missing corpus adapter");
    entity_ = &context->get_entity();
    std::string router;
    auto segment = source_.storage()->get(core::kQuantizedGraphRouterSegment);
    if (segment) {
      if (segment->data_size() > core::kMaxQuantizedGraphRouterBytes)
        throw std::runtime_error("router too big");
      router.resize(segment->data_size());
      core::IndexStorage::MemoryBlock block;
      if (segment->read(0, block, router.size()) != router.size())
        throw std::runtime_error("router read failed");
      std::memcpy(router.data(), block.data(), router.size());
      if (zvec::ailego::Crc32c::Hash(router.data(), router.size()) !=
          segment->data_crc())
        throw std::runtime_error("router checksum");
    }
    zvec::ailego::Params pages;
    py::dict cfg = py::module_::import("json").attr("loads")(config);
    if (cfg.contains("params")) {
      py::dict values = cfg["params"].cast<py::dict>();
      for (const auto &key : {core::quantized_graph_pages::kPolicy,
                              core::quantized_graph_pages::kMinimum,
                              core::quantized_graph_pages::kFloor}) {
        std::string legacy = "proxima.hnsw." + std::string(key).substr(8);
        auto k = values.contains(key.c_str()) ? std::string(key) : legacy;
        if (values.contains(k.c_str())) {
          py::object v = values[k.c_str()];
          if (py::isinstance<py::str>(v))
            pages.set(key, v.cast<std::string>());
          else
            pages.set(key, v.cast<double>());
        }
      }
    }
    check(core::QuantizedGraphIndex::load(
        Provider{*entity_}, entity_->doc_cnt(), dim,
        hnsw->metric_type == ci::MetricType::kCosine, entity_->vector_size(),
        router, false, source_.storage()->get("hnsw.quantized_graph.blocks"),
        graph_, pages));
    query_ = std::make_unique<core::QuantizedGraphQuery>(
        dim, source_.distance(), source_.meta().dimension());
  }
  void configure(const std::string &config) {
    py::dict p = py::module_::import("json").attr("loads")(config);
    capacity_ = p["ef_search"].cast<size_t>();
    options_.margin = p["quantized_graph_margin"].cast<float>();
    options_.bucket_width = p["quantized_graph_bucket_width"].cast<float>();
    options_.max_expansions =
        p["quantized_graph_max_expansions"].cast<uint32_t>();
    options_.route_beam = p["quantized_graph_route_beam"].cast<uint32_t>();
    options_.uncertainty_bonus =
        p["quantized_graph_uncertainty_bonus"].cast<float>();
  }
  py::array_t<uint64_t> search(const Array &input, int k) {
    if (input.ndim() != 1 || size_t(input.shape(0)) != dimension_ || k != 10)
      throw std::runtime_error("invalid query");
    {
      py::gil_scoped_release release;
      std::string buffer;
      const void *q = nullptr;
      check(source_.prepare(input.data(), &buffer, &q));
      check(
          query_->reset(static_cast<const float *>(q), k, capacity_, options_));
      check(graph_->search(Provider{*entity_}, *query_));
    }
    const auto &hits = query_->results();
    auto n = std::min(size_t(k), hits.size());
    py::array_t<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i)
      out.mutable_data()[i] = entity_->get_key(hits[i].first);
    return out;
  }
  void close() {
    graph_.reset();
    query_.reset();
    owner_.reset();
    check(source_.close());
  }
};
PYBIND11_MODULE(_direct_core, m) {
  py::class_<Index>(m, "Index")
      .def(py::init<const std::string &, const std::string &, int>())
      .def("configure", &Index::configure)
      .def("search", &Index::search, py::arg("q").noconvert(), py::arg("k"))
      .def("close", &Index::close);
}
