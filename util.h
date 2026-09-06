#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>

// TODO: replace most areas with die
static inline void g4_die(const char *msg) {
  fprintf(stderr, "Error: %s\n", msg);
  exit(1);
}

static inline void *xcalloc(size_t n, size_t size) {
  void *p = calloc(n, size);
  if (!p)
    g4_die("out of memory");
  return p;
}

static inline void *xmalloc(size_t size) {
  void *p = malloc(size);
  if (!p)
    g4_die("out of mem");
  return p;
}

#endif
