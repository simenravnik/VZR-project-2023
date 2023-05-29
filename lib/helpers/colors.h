#ifndef COLORS_H
#define COLORS_H

#include <stdio.h>

void red () {
  printf("\033[1;31m");
}

void green () {
  printf("\033[1;32m");
}

void reset () {
  printf("\033[0m");
}

#endif