#ifndef __DDR_CONV_H__
#define __DDR_CONV_H__

// 1: Includes
// ----------------------------------------------------------------------------
#include "ddr_common.h"
#include <conv_api.h>
#include <im2col_api.h>

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------
typedef struct {
    dma_context dma;
    memory_context mem;
    common_context com;

    //upgrade begin
    conv_info **info;
    //upgrade end
} conv_context;

typedef struct {
    dma_context dma;
    memory_context mem;
    common_context com;

    //upgrade begin
    im2col_info **info;
    //upgrade end
} im2col_context;


typedef void (*CONV_DDR_PTR) (u32 firstMapNo, u32 lastMapNo, conv_context *context);
typedef void (*IM2COL_DDR_PTR) (u32 firstMapNo, u32 lastMapNo, im2col_context *context);

// 3: Static Local Data
// ----------------------------------------------------------------------------

// 4:  Exported Functions (non-inline)
// ----------------------------------------------------------------------------
void conv_ddr(u32 firstMapNo, u32 lastMapNo, conv_context *context);
void im2col_ddr(u32 firstMapNo, u32 lastMapNo, im2col_context *context);


#endif//__DDR_CONV_H__