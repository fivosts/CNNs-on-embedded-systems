#ifndef __VECVECDOTPRODUCT32_H__
#define __VECVECDOTPRODUCT32_H__
#include <mv_types.h>
    
//!@{
/// mvcvvecVecDotProduct32 kernel the dot product of two vectors.
/// @param[in] vec1        - First vector containing float16.
/// @param[in] vec2        - Second vector containing float16.
/// @param[in] width       - Width of the input lines (multple of 8). If the width is not multiple of 8, then round up to multiple of 8 occurs
/// @return    The dot product in float32. All internal arithmetic operations are performed in float32

float mvcvvecVecDotProduct32_asm(half* vec1, half* vec2, u32 width);
//!@}


#endif //__VECVECDOTPRODUCT32_H__