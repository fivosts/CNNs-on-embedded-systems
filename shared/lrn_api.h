#ifndef __LRN_API_H__
#define __LRN_API_H__

#include <mv_types.h>

typedef struct
{
	u8 *input;
	u8 *output;

	u32 image_offset;
	u8 BPP;
	u16 channels;

	u8 ddr_function;

	u8 local_ratio;
	fp16 alpha;
	fp16 beta;

} lrn_info;

#endif