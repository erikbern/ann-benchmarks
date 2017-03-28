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

static int entry_length;
static int max_number_of_points = 0;
static int r = 0;
static int dimModifier = 0;

bool configure(const char* var, const char* val) {
  if (strcmp(var, "r") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      r = k;
      return true;
    }
  } else if (strcmp(var, "modifier") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      dimModifier = k;
      return true;
    }
  } else if (strcmp(var, "numpoints") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      max_number_of_points = k;
      return true;
    }
  } else return false;
}

static std::vector<double>* pointset = nullptr;

bool end_configure(void) {
  pointset = new std::vector<double>();
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
  pointset->insert(pointset->end(), parsed_entry.begin(), parsed_entry.end());
  entry_count++;
  return true;
}

static Dolphinn::Hypercube<double, char>* hypercube = nullptr;
static std::vector<std::vector<std::pair<int,float>>> results_distances(1);
static std::vector<double> parsed_entry;
static size_t position = 0;

void end_train(void) {
  entry_length = pointset->size() / entry_count;
  hypercube = new Dolphinn::Hypercube<double, char>(
      *pointset,
      entry_count,
      entry_length,
      /* hypercube_dimension */ floor(log2(entry_count)) - 2,
      1,
      r);
}

bool prepare_query(const char* entry) {
  parsed_entry = parseEntry(entry);
  position = 0;
  results_distances.clear();
}


size_t query(const char* entry, size_t k) {
  results_distances[0].resize(k);
  hypercube->m_nearest_neighbors_query(
      parsed_entry,
      1,
      /* Number of nearest neighbors*/ k, 
      /* max_candidate_count */ max_number_of_points, 
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
