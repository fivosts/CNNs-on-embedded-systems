#ifndef __LRN_ACROSSCHANNELS_LR5_H__
#define __LRN_ACROSSCHANNELS_LR5_H__
#include <mv_types.h>

void LRN_AcrossChannels_LR5_A0_0001_B_0_75(fp16* input, fp16* output, u16 channels, u16 pixel_batch,
                                u8 local_ratio, fp16 alpha, fp16 beta);

#endif