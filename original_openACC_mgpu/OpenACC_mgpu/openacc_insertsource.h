#ifndef _OPENACC_SOURCE
#define _OPENACC_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void OPENACC_InsertSource(float dt, int it, int iSource,
						  float *p, float *q, float src, const int sx, const int sy, const int sz);

#endif
