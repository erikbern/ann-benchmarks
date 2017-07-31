#include <sys/types.h>

#include <math.h>
#include <time.h>
#include <errno.h>
#include <regex.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static bool fail_transition = false;

static bool use_regexp = false;
static regex_t preg;

static int signal_to_raise_on_query = 0;

static struct timespec query_time;

bool configure(const char* var, const char* val) {
  if (strcmp(var, "raise") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 1) {
      return false;
    } else {
      raise((int)k);
      return true;
    }
  } else if (strcmp(var, "raise-on-query") == 0) {
    char* end;
    errno = 0;
    long k = strtol(val, &end, 10);
    if (errno != 0 || *val == 0 || *end != 0 || k < 0) {
      return false;
    } else {
      signal_to_raise_on_query = (int)k;
      return true;
    }
  } else if (strcmp(var, "query-time") == 0) {
    char* end;
    errno = 0;
    double d = strtod(val, &end);
    if (errno != 0 || *val == 0 || *end != 0 || d < 0.0) {
      return false;
    } else {
      double i = 0.0, f = 0.0;
      f = modf(d, &i);
      query_time.tv_sec = i;
      query_time.tv_nsec = f * 1000000000.0;
      return true;
    }
  } else if (strcmp(var, "fail-query-transition") == 0) {
    fail_transition = (strcmp(val, "1") == 0);
    return true;
  } else if (strcmp(var, "entry-regular-expression") == 0) {
    if (use_regexp)
      regfree(&preg);
    if (!val[0]) {
      use_regexp = false;
      return true;
    } else {
      use_regexp = (regcomp(&preg, val, REG_EXTENDED | REG_NOSUB) == 0);
      return use_regexp;
    }
  } else return false;
}

bool end_configure(void) {
  return !fail_transition;
}

static size_t entry_count = 0;

bool train(const char* entry) {
  bool success = (!use_regexp || regexec(&preg, entry, 0, 0, 0) == 0);
  if (success)
    entry_count++;
  return success;
}

void end_train(void) {
}

static size_t result_count = 0;
static size_t result_position = 0;

static size_t min(size_t a, size_t b) {
  return (a < b ? a : b);
}

size_t query(const char* entry, size_t k) {
  if (use_regexp && regexec(&preg, entry, 0, 0, 0) != 0) {
    return 0;
  } else if (signal_to_raise_on_query) {
    raise(signal_to_raise_on_query);
  }
  nanosleep(&query_time, NULL);
  result_count = random() % (entry_count + 1);
  result_position = entry_count - result_count;
  return result_count;
}

size_t query_result(void) {
  if (result_position < entry_count) {
    return result_position++;
  } else return SIZE_MAX;
}

void end_query(void) {
  if (use_regexp)
    regfree(&preg);
}
