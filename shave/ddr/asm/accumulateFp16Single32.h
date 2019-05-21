#ifndef __ACCUMULATEFP16SINGLE32_H__
#define __ACCUMULATEFP16SINGLE32_H__
#include <mv_types.h>
    
//!@{
/// mvcvAccumulateFp16Single kernel computes does an addition between elements in dst and bias
/// @param[in][out] dst        - Input line that contains elements to be added and also destination for new elements
/// @param[in] bias            - Element to be added
/// @param[in] width           - Width of the input lines (multple of 8). If the width is not multiple of 8, then round up to multiple of 8 occurs
/// @return    Nothing

void mvcvAccumulateFp16Single32_asm(half* dst, half bias, u32 width);
//!@}


#endif //__ACCUMULATEFP16SINGLE32_H__