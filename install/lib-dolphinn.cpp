#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "lib-dolphinn/IO.h"
#include "lib-dolphinn/hypercube.h"

extern "C" {

static int entry_length = 0;

bool configure(const char* var, const char* val) {
  if (strcmp(var, "entry_length") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      entry_length = k;
      return true;
    }
  } else return false;
}

static std::vector<float>* pointset = nullptr;

bool end_configure(void) {
  pointset = new std::vector<float>();
  return true;
}

static size_t entry_count = 0;

std::vector<float> parseEntry(const char* entry) {
  std::vector<float> e;
  std::string line(entry);
  double x;
  auto sstr = std::istringstream(line);
  while (sstr >> x) {
    e.push_back(x);
  }
  return e;
}

bool train(const char* entry) {
  auto parsed_entry = parseEntry(entry);
  if (parsed_entry.size() != entry_length)
    return false;
  pointset->insert(pointset->end(), parsed_entry.begin(), parsed_entry.end());
  entry_count++;
  return true;
}

static Dolphinn::Hypercube<float, char>* hypercube = nullptr;
static std::vector<std::vector<std::pair<int,float>>> results_distances(1);
static size_t position = 0;

void end_train(void) {
  hypercube = new Dolphinn::Hypercube<float, char>(
      *pointset,
      entry_count,
      entry_length,
      /* hypercube_dimension */ floor(log2(entry_count)/2),
      1);
}

size_t query(const char* entry, size_t k) {
  auto parsed_entry = parseEntry(entry);
  position = 0;
  if (parsed_entry.size() != entry_length)
    return 0;
  results_distances.clear();
  results_distances[0].resize(k);
  hypercube->m_nearest_neighbors_query(
      parsed_entry,
      1,
      /* Number of nearest neighbors*/ k, 
      /* max_candidate_count */ entry_count * 1 / 100,
      results_distances,
      1);
  return results_distances[0].size();
}


size_t query_result(void) {
  if (position < results_distances[0].size()) {
    auto elem = std::get<0>(results_distances[0][position++]);
    return elem;
  } else return SIZE_MAX;
}

void end_query(void) {
}

}
