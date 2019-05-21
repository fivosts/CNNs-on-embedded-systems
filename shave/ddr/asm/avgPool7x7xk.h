#ifndef __AVG_POOL_7X7_XK_H__
#define __AVG_POOL_7X7_XK_H__
#include <mv_types.h>
#include <mvcv_macro.h>

void mvcvAvgPool7x7xk_asm(half** src, half** dst, u32 outputWidth);

#endif