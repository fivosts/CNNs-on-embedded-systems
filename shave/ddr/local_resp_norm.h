#ifndef _LOCAL_RESP_NORM_H
#define _LOCAL_RESP_NORM_H

#include "ddr_common.h"
#include <lrn_api.h>

typedef struct {
    dma_context dma;
    memory_context mem;
    common_context com;

    lrn_info *info;
} lrn_context;

typedef void (*LRN_DDR_PTR) (u32 first_pixel, u32 last_pixel, lrn_context *context);

void lrn_ddr_generic(u32 first_pixel, u32 last_pixel, lrn_context *context);

void lrn_ddr_LR5_A0_0001_B_0_75(u32 first_pixel, u32 last_pixel, lrn_context *context);

void LRN_AcrossChannels_generic(fp16** input, fp16** output, u16 channels, u16 pixel_batch, u8 local_ratio, fp16 alpha, fp16 beta);

#endif