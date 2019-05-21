#ifndef __CONVOLUTION1x1s1hhhh_H__
#define __CONVOLUTION1x1s1hhhh_H__
#include <mv_types.h>

void mvcvConvolution1x1Fp16ToFp16_asm(half** in, half** out, half conv[1], u32 inWidth);

#endif