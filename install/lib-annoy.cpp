#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "lib-annoy/src/annoylib.h"
#include "lib-annoy/src/kissrandom.h"

extern "C" {

static int num_trees = 0;
static int search_k = 0;

bool configure(const char* var, const char* val) {
  if (strcmp(var, "num_trees") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      num_trees = k;
      return true;
    }
  } else if (strcmp(var, "search_k") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      search_k = k;
      return true;
    }
  } else return false;
}

static std::vector<std::vector<double>> pointset;

bool end_configure(void) {
  return true;
}

static size_t entry_count = 0;

std::vector<double> parseEntry(const char* entry) {
  std::vector<double> e;
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
  pointset.push_back(parsed_entry);
  entry_count++;
  return true;
}

static AnnoyIndex<int, double, Euclidean, Kiss32Random>* ds = nullptr;
static std::vector<int> results_idxs;
static std::vector<double> parsed_entry; 
static size_t position = 0;

void end_train(void) {
  ds = new AnnoyIndex<int, double, Euclidean, Kiss32Random>((pointset[0]).size());
  for (int i = 0; i < pointset.size(); i++) {
	ds->add_item(i, &((pointset[i])[0]));
  }
  pointset.clear();
  pointset.shrink_to_fit();
  ds->build(num_trees);
}

bool prepare_query(const char* entry) {
  position = 0;
  results_idxs.clear();
  parsed_entry = parseEntry(entry);
  return true;
}

size_t query(const char* entry, size_t k) {
  ds->get_nns_by_vector(&(parsed_entry[0]), k, search_k, &results_idxs, nullptr);
  return results_idxs.size();
}


size_t query_result(void) {
  if (position < results_idxs.size()) {
    auto elem = results_idxs[position++];
    return elem;
  } else return SIZE_MAX;
}

void end_query(void) {
}

}
