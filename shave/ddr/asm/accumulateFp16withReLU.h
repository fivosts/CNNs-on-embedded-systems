#ifndef __ACCUMULATEFp16WITHRELU_H__
#define __ACCUMULATEFp16WITHRELU_H__
#include <mv_types.h>
#include <mvcv_macro.h>

void mvcvaccumulateFp16withReLU_asm(half** dst, half** src, u32 inWidth);


#endif //__ACCUMULATEFp16WITHRELU_H__