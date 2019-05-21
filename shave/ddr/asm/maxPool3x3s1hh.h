#ifndef __MAX_POOL_3x3_s1_hh_H__
#define __MAX_POOL_3x3_s1_hh_H__
#include <mv_types.h>
#include <mvcv_macro.h>

void mvcvMaxPool3x3s1hh_asm(half** src, half** dst, u32 outputWidth);

#endif