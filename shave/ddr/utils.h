#ifndef __DDR_UTILS_H__
#define __DDR_UTILS_H__

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

int round_up64(int x);
void memset64_h(half *ptr, int size, half val);
void relu64_h(half *in, half *out, int size);
void relu_inplace64_h(half *ptr, int size);

#endif//__DDR_UTILS_H__