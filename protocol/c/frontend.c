#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define BUF_UPDATE(Name, Type) \
  void Name(Type v, Type** buf, size_t* buflen, size_t* bufpos) { \
    if (*bufpos >= *buflen) { \
      *buflen *= 2; \
      *buf = realloc(*buf, sizeof(Type) * (*buflen)); \
    } \
    (*buf)[(*bufpos)++] = v; \
  }

BUF_UPDATE(update_buf, char)
BUF_UPDATE(update_sbuf, char*)

char** tokenise(
    const char* sequence,
    size_t sequence_len,
    size_t* token_count) {
  size_t buflen = 4;
  char* buf = malloc(buflen);
  size_t bufpos = 0;
  bool force = false;

  size_t reslen = 2;
  char** res = calloc(2, sizeof(char*));
  size_t respos = 0;

  const char* const end = sequence + sequence_len;
  const char* pos = sequence;
  int c, c1;
  while (*pos && (c = *(pos++))) {
    bool subloop = true;
    switch (c) {
      case '\\':
        /* Unconditionally write the next character to the current token, if
           there is one */
        if (pos != end)
          update_buf(*(pos++), &buf, &buflen, &bufpos);
        break;
      case '\"':
        force = true;
        while (subloop && *pos && (c1 = *(pos++))) {
          switch (c1) {
            case '\\':
              if (*pos) {
                c1 = *(pos++);
                switch (c1) {
                  case '$':
                    /* fall through */
                  case '\\':
                    /* fall through */
                  case '\"':
                    update_buf(c1, &buf, &buflen, &bufpos);
                    break;
                  default:
                    update_buf('\\', &buf, &buflen, &bufpos);
                    update_buf((char)c1, &buf, &buflen, &bufpos);
                }
              }
              break;
            case '\"':
              subloop = false;
              break;
            default:
              update_buf(c1, &buf, &buflen, &bufpos);
          }
        }
        break;
      case '\'':
        force = true;
        while (subloop && *pos && (c1 = *(pos++))) {
          switch (c1) {
            case '\'':
              subloop = false;
              break;
            default:
              update_buf(c1, &buf, &buflen, &bufpos);
          }
        }
        break;
      case ' ':
        /* fall through */
      case '\t':
        /* fall through */
      case '\n':
        /* fall through */
      case '\r':
        if (force || bufpos != 0) {
          update_buf(0, &buf, &buflen, &bufpos);
          update_sbuf(strdup(buf), &res, &reslen, &respos);
          force = false;
          bufpos = 0;
        }
        break;
      default:
        update_buf(c, &buf, &buflen, &bufpos);
    }
  }
  if (force || bufpos != 0) {
    update_buf(0, &buf, &buflen, &bufpos);
    update_sbuf(strdup(buf), &res, &reslen, &respos);
  }
  free(buf);

  if (token_count)
    *token_count = respos;
  /* Include a null terminator for tokenise_free */
  update_sbuf(NULL, &res, &reslen, &respos);

  return res;
}

void tokenise_free(char** result) {
  char** it = result;
  while (*it)
    free(*(it++));
  free(result);
}

/* Sets the configuration option @var to @val. Returns true if the change was
   successful, or false if @val did not make sense in this context. */
extern bool configure(const char* var, const char* val);
/* Called immediately before transitioning to training mode. Returns true if
   all mandatory configuration options were set, and false otherwise, which
   will cause the main program to terminate. */
extern bool end_configure(void);
/* Adds @entry as the next item of training data. Returns true if the item was
   successfully added, and false otherwise. */
extern bool train(const char* entry);
/* Called immediately before transitioning to query mode. (This is typically
   the point at which strategies should create data structures from the
   training data.) */
extern void end_train(void);
/* Searches for @count close matches to @entry, returning the number of matches
   found; the query_result function will be called this many times (but no
   more than @count times) to retrieve those matches.

   (When prepared query mode is enabled and supported, @entry will be NULL.) */
extern size_t query(const char* entry, size_t count);
/* Returns the next query result, or SIZE_MAX if there are none remaining.
   Closer matches should be returned first. */
extern size_t query_result(void);
/* Called immediately before finishing query mode and terminating the
   program. */
extern void end_query(void);

/* Support for prepared query mode. */

/* Prepares the query represented by @entry for execution. Returns true if the
   underlying strategy supports prepared query mode (even if @entry does not
   represent a valid query!), or false otherwise. */
extern bool prepare_query(const char* entry);

bool __attribute__((weak)) prepare_query(const char* entry) {
  return false;
}

static size_t min(size_t a, size_t b) {
  return (a < b ? a : b);
}

int main(int argc, const char* argv[]) {
  setvbuf(stdin, NULL, _IOLBF, 0);
  setvbuf(stdout, NULL, _IOLBF, 0);

  size_t token_count = 0;
  char* buf = NULL;
  size_t buflen = 0;
  ssize_t length = 0;

  bool prepared_queries = false;

  /* Configuration mode */
  while ((length = getline(&buf, &buflen, stdin)) > 1) {
    char** tokens = tokenise(buf, length, &token_count);
    if (token_count == 2) {
      if (configure(tokens[0], tokens[1])) {
        printf("epbprtv0 ok\n");
      } else printf("epbprtv0 fail\n");
    } else if (token_count == 3 && strcmp(tokens[0], "frontend") == 0) {
      if (strcmp(tokens[1], "prepared-queries") == 0) {
        prepared_queries = (strcmp(tokens[2], "1") == 0);
        printf("epbprtv0 ok\n");
      } else printf("epbprtv0 fail\n");
    } else printf("epbprtv0 fail\n");
    tokenise_free(tokens);
  }
  if (end_configure()) {
    printf("epbprtv0 ok\n");
  } else {
    printf("epbprtv0 fail\n");
    return 1;
  }

  /* Training mode */
  size_t success = 0, fail = 0;
  while ((length = getline(&buf, &buflen, stdin)) > 1) {
    char** tokens = tokenise(buf, length, &token_count);
    if (token_count == 1 && train(tokens[0])) {
      printf("epbprtv0 ok %u\n", success++);
    } else {
      printf("epbprtv0 fail\n");
      fail++;
    }
    tokenise_free(tokens);
  }
  end_train();
  printf("epbprtv0 ok %u", success);
  if (fail)
    printf(" fail %u", fail);
  putchar('\n');

  if (!prepared_queries) {
    /* Query mode */
    while ((length = getline(&buf, &length, stdin)) > 1) {
      char** tokens = tokenise(buf, length, &token_count);
      if (token_count == 2) {
        char* count_token = tokens[1];
        char* last;
        long int k = 0;
        errno = 0;
        k = strtol(count_token, &last, 10);
        if (errno != 0 || *count_token == 0 || *last != 0 || k < 1) {
          printf("epbprtv0 fail\n");
        } else {
          size_t result_count = 0;
          if ((result_count = query(tokens[0], k)) > 0) {
            size_t to_return = min(k, result_count);
            printf("epbprtv0 ok %u total_candidates %u\n", to_return, result_count);
            for (size_t i = 0; i < to_return; i++)
              printf("epbprtv0 %u\n", query_result());
          } else printf("epbprtv0 fail\n");
        }
      } else printf("epbprtv0 fail\n");
      tokenise_free(tokens);
    }
  } else {
    /* Prepared query mode */
    char* last_query = NULL;
    long int last_k = 0;
    while ((length = getline(&buf, &length, stdin)) > 1) {
      char** tokens = tokenise(buf, length, &token_count);
      if (token_count == 1 && strcmp(tokens[0], "query") == 0) {
        if (last_k) {
          size_t result_count = 0;
          if ((result_count = query(last_query, last_k)) > 0) {
            size_t to_return = min(last_k, result_count);
            printf("epbprtv0 ok %u total_candidates %u\n", to_return, result_count);
            for (size_t i = 0; i < to_return; i++)
              printf("epbprtv0 %u\n", query_result());
          } else printf("epbprtv0 fail\n");
        } else printf("epbprtv0 fail\n");
      } else if (token_count == 2) {
        char* count_token = tokens[1];
        char* last;
        long int k = 0;
        errno = 0;
        k = strtol(count_token, &last, 10);
        if (errno != 0 || *count_token == 0 || *last != 0 || k < 1) {
          printf("epbprtv0 fail\n");
        } else {
          if (last_query)
            free(last_query);
          bool prepared = prepare_query(tokens[0]);
          last_query = (prepared ? NULL : strdup(tokens[0]));
          last_k = k;
          printf("epbprtv0 ok prepared %s\n", (prepared ? "true" : "false"));
        }
      } else printf("epbprtv0 fail\n");
      tokenise_free(tokens);
    }
  }
  end_query();
  free(buf);
  printf("epbprtv0 ok\n");
  return 0;
}
