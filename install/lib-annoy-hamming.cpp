#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "lib-annoy-hamming/src/annoylib.h"
#include "lib-annoy-hamming/src/kissrandom.h"

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

static std::vector<std::vector<bool>> pointset;

std::vector<bool> parseEntry(const char* entry) {
  std::string line(entry);
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
  std::vector<bool> e;
  for (int i = 0; i < line.length(); i++) {
      e.push_back(line[i] == '1');
  }
  return e;
}

int64_t* mapEntry(std::vector<bool> entry) {
  size_t d = entry.size();
  size_t chunksize = 8 * sizeof(int64_t);
  int64_t* vec = new int64_t[d / chunksize + 1];
  for (int i = 0; i < d / chunksize + 1; i++) {
    int64_t t = 0;
    for (int k = 0; k < chunksize; k++) {
      t += (static_cast<int64_t>(entry[i * chunksize + k]) << (chunksize - 1 - k));
    }
    vec[i] = t;
  }
  return vec;
}

bool end_configure(void) {
  return true;
}

static size_t entry_count = 0;

bool train(const char* entry) {
  auto parsed_entry = parseEntry(entry);
  pointset.push_back(parsed_entry);
  entry_count++;
  return true;
}

static AnnoyIndex<int, int64_t, Hamming, Kiss32Random>* ds = nullptr;
static std::vector<int> results_idxs;
static int64_t* parsed_entry = nullptr;  
static size_t position = 0;

void end_train(void) {
  ds = new AnnoyIndex<int, int64_t, Hamming, Kiss32Random>((pointset[0]).size());
  for (int i = 0; i < pointset.size(); i++) {
	ds->add_item(i, mapEntry(pointset[i]));
  }
  pointset.clear();
  pointset.shrink_to_fit();
  ds->build(num_trees);
}

bool prepare_query(const char* entry) {
  position = 0;
  results_idxs.clear();
  if (parsed_entry == nullptr)
    delete[] parsed_entry;
  parsed_entry = mapEntry(parseEntry(entry));
  return true;
}

size_t query(const char* entry, size_t k) {
  ds->get_nns_by_vector(parsed_entry, k, search_k, &results_idxs, nullptr);
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
