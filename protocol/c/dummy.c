#include <stdint.h>
#include <string.h>
#include <stdbool.h>

bool configure(const char* var, const char* val) {
  return true;
}

bool end_configure(void) {
  return true;
}

bool train(const char* entry) {
  return true;
}

void end_train(void) {
}

size_t query(const char* entry, size_t k) {
  return 0;
}

size_t query_result(void) {
  return SIZE_MAX;
}

void end_query(void) {
}
