// CSR provider and snapshot loading adapted from Zvec qg_bench.cc (21c25986).
// Copyright 2026-present the zvec project; Apache-2.0.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include "algorithm/quantized_graph/quantized_graph_query.h"

using namespace zvec;
using namespace zvec::core;
namespace {
void check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error(what);
}
void status(int rc, const char *what) {
  if (rc) throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
}
template<class T> std::vector<T> read(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  check(bool(f), "open " + path);
  auto size = f.tellg();
  check(size >= 0 && size % sizeof(T) == 0, "size " + path);
  std::vector<T> v(size / sizeof(T));
  f.seekg(0);
  check(bool(f.read(reinterpret_cast<char *>(v.data()), size)), "read " + path);
  return v;
}
template<class T> void write(const std::string &path, const std::vector<T> &v) {
  std::ofstream f(path, std::ios::binary);
  check(bool(f.write(reinterpret_cast<const char *>(v.data()), v.size()*sizeof(T))), "write " + path);
}
// The anonymous-namespace type enables inlining. Arrays own all borrowed data.
struct Graph {
  struct Vector { const float *p = nullptr; const void *data() const { return p; } };
  struct Edges {
    const uint32_t *p; size_t n;
    size_t size() const { return n; }
    uint32_t operator[](size_t i) const { return p[i]; }
  };
  size_t dim, count;
  std::vector<float> vectors;
  std::vector<uint64_t> offsets, ids;
  std::vector<uint32_t> edges;
  Graph(const YAML::Node &c) : dim(c["dimension"].as<size_t>()),
      count(c["train_count"].as<size_t>()) {
    auto p = c["provider_prefix"].as<std::string>();
    vectors = read<float>(p+".vectors"); offsets = read<uint64_t>(p+".offsets");
    ids = read<uint64_t>(p+".ids"); edges = read<uint32_t>(p+".edges");
    check(dim && count && vectors.size()==dim*count && ids.size()==count &&
          offsets.size()==count+1 && offsets[0]==0 && offsets.back()==edges.size(), "provider shape");
    auto sorted = ids; std::sort(sorted.begin(), sorted.end());
    for (size_t i=0;i<count;++i) {
      check(sorted[i]==i && offsets[i]<=offsets[i+1], "provider IDs/offsets");
      double norm=0;
      for(size_t d=0;d<dim;++d) { float x=vectors[i*dim+d]; check(std::isfinite(x), "nonfinite vector"); norm+=double(x)*x; }
      if(c["metric"].as<std::string>()=="cosine") check(std::abs(norm-1)<2e-5, "cosine provider must be normalized exactly once upstream");
    }
    for(auto id:edges) check(id<count, "edge ID range");
  }
  bool valid(uint32_t id) const { return id<count; }
  int get_vector(uint32_t id, Vector &v) const {
    if(!valid(id)) return IndexError_NoExist;
    v.p=vectors.data()+size_t(id)*dim; return 0;
  }
  Edges neighbors(uint32_t id) const { return {edges.data()+offsets[id],size_t(offsets[id+1]-offsets[id])}; }
};
}
namespace py = pybind11;
namespace {
class Index {
  YAML::Node config_;
  Graph provider_;
  IndexStorage::Pointer storage_;
  std::shared_ptr<QuantizedGraphIndex> index_;
  std::unique_ptr<QuantizedGraphQuery> query_;
  QuantizedGraphSearchOptions options_;
  size_t capacity_ = 0;
  uint64_t scan_limit_ = 0;
  std::string profile_;
 public:
  explicit Index(const std::string &config) : config_(YAML::Load(config)), provider_(config_) {
    bool cosine = config_["metric"].as<std::string>() == "cosine";
    auto metric = IndexFactory::CreateMetric(cosine ? "InnerProduct" : "SquaredEuclidean");
    check(bool(metric), "missing metric");
    IndexMeta meta; meta.set_meta(IndexMeta::DT_FP32, provider_.dim);
    status(metric->init(meta, {}), "metric init");
    auto distance = metric->distance();
    query_ = std::make_unique<QuantizedGraphQuery>(provider_.dim,
        [distance, cosine](const void *a, const void *b, size_t dim, float *out) {
          distance(a, b, dim, out); if (cosine) *out += 1.f;
        });
  }
  void configure(const std::string &json) {
    auto point = YAML::Load(json);
    auto name = point["profile"].as<std::string>();
    if (!index_ || profile_ != name) {
      index_.reset(); storage_.reset(); profile_.clear();
      auto p = config_["profiles"][name];
      check(bool(p), "unknown profile");
      check(p["encoding"]["rotation_views"].as<int>() == p["router"]["rotation_views"].as<int>(), "rotation views mismatch");
      auto prefix = p["index_prefix"].as<std::string>();
      auto bytes = read<char>(prefix + ".router");
      std::string router(bytes.begin(), bytes.end());
      storage_ = IndexFactory::CreateStorage("MMapFileReadStorage");
      status(storage_->init({}), "storage init");
      status(storage_->open(prefix + ".qg", false), "snapshot open");
      status(QuantizedGraphIndex::load(provider_, provider_.count, provider_.dim,
          config_["metric"].as<std::string>() == "cosine", provider_.dim * 4,
          router, false, storage_->get(QuantizedGraphIndex::kSnapshotSegment), index_), "QG load");
      profile_ = name;
    }
    capacity_ = point["capacity"].as<size_t>();
    check(capacity_ > 0, "invalid capacity");
    scan_limit_ = point["scan_limit"].as<uint64_t>();
    auto p = point["search"];
    options_.margin = p["margin"].as<float>();
    options_.bucket_width = p["bucket_width"].as<float>();
    options_.max_expansions = p["max_expansions"].as<uint32_t>();
    options_.route_beam = p["route_beam"].as<uint32_t>();
    options_.uncertainty_bonus = p["uncertainty_bonus"].as<float>();
  }
  py::array_t<uint64_t> search(const py::array_t<float, py::array::c_style> &input, int k) {
    check(bool(index_) && capacity_ > 0, "configure before querying");
    check(input.ndim() == 1 && size_t(input.shape(0)) == provider_.dim && k == 10, "expected FP32 dimension and top10");
    // Keep the GIL: this index owns one mutable query context, single query thread only.
    status(query_->reset(input.data(), k, capacity_, options_, scan_limit_), "query reset");
    status(index_->search(provider_, *query_), "QG search");
    const auto &hits = query_->results();
    check(hits.size() >= size_t(k) && query_->expanded() > 0, "QG must return top10 and expand nodes");
    py::array_t<uint64_t> ids(k);
    for (int j = 0; j < k; ++j) ids.mutable_data()[j] = provider_.ids[hits[j].first];
    return ids;
  }
  py::dict diagnostics() const {
    check(bool(index_), "configure before diagnostics");
    py::array_t<float> scores(query_->results().size());
    for (size_t j = 0; j < query_->results().size(); ++j)
      scores.mutable_data()[j] = query_->results()[j].second;
    py::dict result;
    result["scores"] = scores; result["expanded"] = query_->expanded();
    result["comparisons"] = query_->comparisons(); return result;
  }
};
}
PYBIND11_MODULE(_zvec_qg_core, m) {
  py::class_<Index>(m, "Index").def(py::init<const std::string &>())
      .def("configure", &Index::configure)
      .def("search", &Index::search, py::arg("query").noconvert(), py::arg("k"))
      .def("diagnostics", &Index::diagnostics);
}
