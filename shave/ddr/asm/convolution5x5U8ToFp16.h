#ifndef __CONVOLUTION5x5U8ToFp16_H__
#define __CONVOLUTION5x5U8ToFp16_H__
#include <mv_types.h>
#include <mvcv_macro.h>


//!@{
/// This kernel performs a convolution on the u8 input image using the given 5x5 matrix
/// @param[in] in         - Input lines, 8-bits integer
/// @param[in] out        - Output line, 16-bits floating point
/// @param[in] conv       - 25 elements array with fp16 values containing the 5x5 convolution matrix
/// @param[in] inWidth    - Width of input line

void mvcvConvolution5x5U8ToFp16_asm(half** in, half** out, half conv[25], u32 inWidth);
//!@}



#endif //__CONVOLUTION5x5_H__