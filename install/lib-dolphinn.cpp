#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <iostream>

#include "lib-dolphinn/src/IO.h"
#include "lib-dolphinn/src/hypercube.h"

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

static std::vector<int>* pointset = nullptr;

bool end_configure(void) {
  pointset = new std::vector<int>();
  return true;
}

static size_t entry_count = 0;

bool train(const char* entry) {
  std::vector<int> parsed_entry;
  std::istringstream es(entry);
  std::string r;
  std::cout << "entry is " << entry << std::endl;
  while (getline(es, r, ' ')) {
    size_t pos;
    try {
      int k = std::stoi(r, &pos, 10);
      parsed_entry.push_back(k);
    } catch (const std::invalid_argument& e) {
      return false;
    } catch (const std::out_of_range& e) {
      return false;
    }
  }
  if (parsed_entry.size() != entry_length)
    return false;
  pointset->insert(pointset->end(), parsed_entry.begin(), parsed_entry.end());
  entry_count++;
  return true;
}

static Dolphinn::Hypercube<int, char>* hypercube = nullptr;
static std::vector<int>* result_indices = nullptr;

void end_train(void) {
  hypercube = new Dolphinn::Hypercube<int, char>(
      *pointset,
      entry_count,
      entry_length,
      /* hypercube_dimension */ floor(log2(entry_count)/2),
      2);
}

size_t query(const char* entry, size_t k) {
  std::vector<int> parsed_entry;
  std::istringstream es(entry);
  std::string r;
  std::cout << "entry is " << entry << std::endl;
  while (getline(es, r, ' ')) {
    size_t pos;
    try {
      int k = std::stoi(r, &pos, 10);
      parsed_entry.push_back(k);
    } catch (const std::invalid_argument& e) {
      return 0;
    } catch (const std::out_of_range& e) {
      return 0;
    }
  }
  if (parsed_entry.size() != entry_length)
    return 0;
  if (result_indices)
    delete result_indices;
  result_indices = new std::vector<int>(k);
  hypercube->radius_query(
      parsed_entry,
      1,
      /* RADIUS */ 1, 
      /* max_candidate_count */ entry_count * 1 / 100,
      *result_indices,
      2);
  return result_indices->size();
}

size_t position = 0;

size_t query_result(void) {
  if (position < result_indices->size()) {
    return (*result_indices)[position++];
  } else return SIZE_MAX;
}

void end_query(void) {
}

}
