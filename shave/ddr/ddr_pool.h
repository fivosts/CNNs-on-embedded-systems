#ifndef __DDR_POOL_H__
#define __DDR_POOL_H__

// 1: Includes
// ----------------------------------------------------------------------------
#include "ddr_common.h"
#include <pool_api.h>

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------
typedef struct {
    dma_context dma;
    memory_context mem;
    common_context com;

    pool_info **info;
} pool_context;

typedef void (*POOL_DDR_PTR) (u32 firstMapNo, u32 lastMapNo, pool_context *context);

// 3: Static Local Data
// ----------------------------------------------------------------------------

// 4:  Exported Functions (non-inline)
// ----------------------------------------------------------------------------
void pool_ave_ddr(u32 firstMapNo, u32 lastMapNo, pool_context *context);
void pool_max_ddr(u32 firstMapNo, u32 lastMapNo, pool_context *context);

void mvcvAvgPool7x7xk( half** srcAddr, half** destAddr, u32 width);
void mvcvAvgPool14x14xk( half** srcAddr, half** destAddr, u32 width);

#endif//__DDR_POOL_H__