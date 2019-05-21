#ifndef __MAX_POOL_3x3_s2_hh_H__
#define __MAX_POOL_3x3_s2_hh_H__
#include <mv_types.h>
#include <mvcv_macro.h>

//!@{
/// This kernel computes maximum value on a 3x3 pool using an output stride of 2
///@param[in]   src     - Input lines
///@param[out]  dest    - Output line
///@param[in]   width   - Width of output line (must be multiple of 8)

void mvcvMaxPool3x3s2hh_asm(half** src, half** dst, u32 outputWidth);
//!@}

#endif //__MAX_POOL_3x3_s2_hh_H__
