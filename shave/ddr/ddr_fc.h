#ifndef __DDR_FC_H__
#define __DDR_FC_H__

// 1: Includes
// ----------------------------------------------------------------------------
#include "ddr_common.h"
#include <fc_api.h>

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------
typedef struct {
    dma_context dma;
    memory_context mem;
    common_context com;

    fc_info *info;
} fc_context;

typedef void (*FC_DDR_PTR) (int firstMapNo, int lastMapNo, fc_context *context);

// 3: Static Local Data
// ----------------------------------------------------------------------------

// 4:  Exported Functions (non-inline)
// ----------------------------------------------------------------------------
void fc_ddr(int firstMapNo, int lastMapNo, fc_context *context);

#endif//__DDR_FC_H__