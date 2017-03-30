#include <cstdint>
#include <cstdlib>
#include <cstdbool>

#include <algorithm>
#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <io.h>
#include <types.h>
#include <mihasher.h>
#include <reorder.h>

extern "C" {

static int r = 0;
static int chunks = 1;
static int B = 0;

// Note that MIH assumes that the bitlength of the input 
// divided by the number of chunks is between 6 and 37
// (and segfaults otherwise).
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
  } else if (strcmp(var, "chunks") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      chunks = k;
      return true;
    }
  } else return false;
}

static std::vector<std::vector<bool>> pointset;

bool end_configure(void) {
  return true;
}

static size_t entry_count = 0;

std::vector<bool> parseEntry(const char* entry) {
  std::string line(entry);
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
  std::vector<bool> e;
  for (int i = 0; i < line.length(); i++) {
      e.push_back(line[i] == '1');
  }
  return e;
}

bool train(const char* entry) {
  auto parsed_entry = parseEntry(entry);
  pointset.push_back(parsed_entry);
  entry_count++;
  return true;
}

static mihasher* ds = nullptr;
static size_t position = 0;
static qstat* stats = nullptr;
static UINT8* dataset = nullptr;
static UINT8* querypoint = nullptr; 
static UINT32** results = nullptr;
static UINT32** numres = nullptr;
static int* order = nullptr;

UINT8* create_dataset(std::vector<std::vector<bool>> vec) {
  size_t b = vec[0].size();
  std::cerr << b << std::endl;
  size_t n = vec.size();
  std::cerr << n << std::endl;
  UINT8* dataset = new UINT8[n * b/8];
  for (int i = 0; i < n; i++) {
      // process each chunk
      for (int j = 0; j < b/8; j++) {
          UINT8 t = 0;
          for (int k = 0; k < 8; k++) {
              t += (vec[i][j * 8 + k] << (7 - k));
          }
          dataset[i * b/8 + j] = t;
      }
  }
  return dataset;
}

void end_train(void) {
  size_t b = pointset[0].size();
  B = b;
  size_t n = pointset.size();
  dataset = create_dataset(pointset);
  pointset.clear();
  pointset.shrink_to_fit();
  if (r > 0) { 
    r = n / r; // use r-fraction of the dataset for ordering.
	int* order = new int[B];
	greedyorder(order, dataset, r, B, chunks);
    UINT8* new_dataset = new UINT8[n * B/8];
    reorder(new_dataset, dataset, n, B, order);
    delete[] dataset;
    dataset = new_dataset;
  }
  ds = new mihasher(b, chunks);
  ds->populate(dataset, n, b/8);
  stats = new qstat[1];
}

bool prepare_query(const char* entry) {
  position = 0;
  auto parsed_entry = parseEntry(entry);
  std::vector<std::vector<bool>> queryset;
  queryset.push_back(parsed_entry);
  if (querypoint != nullptr)
      delete[] querypoint;
  if (numres != nullptr) {
      delete numres[0];
      delete[] numres;
  }
  if (results != nullptr) {
      delete[] results[0];
      delete[] results;
  }
  results = new UINT32*[1];
  numres = new UINT32*[1];
  numres[0] = new UINT32[B + 1];
  querypoint = create_dataset(queryset);
  if (r > 0) {
      UINT8* new_query = new UINT8[B/8];
      reorder(new_query, querypoint, 1, B, order);
      delete[] querypoint;
      querypoint = new_query;
  }
  return true;
}

size_t query(const char* entry, size_t k) {
  results[0] = new UINT32[k];
  ds->setK(k);
  ds->batchquery(results[0], numres[0], stats, querypoint, 1, B/8);
  return k; // TODO 
}


size_t query_result(void) {
  //if (position < results_idxs.size()) {
    auto elem = results[0][position++];
    return elem - 1;
  //} else return SIZE_MAX;
}

void end_query(void) {
    delete[] dataset;
    delete[] querypoint;
    if (order != nullptr) {
        delete[] order;
    }
    delete[] results[0];
    delete[] results;
    delete[] numres[0];
    delete[] numres;
}

}
