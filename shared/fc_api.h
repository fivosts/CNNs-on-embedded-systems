#ifndef __FC_API_H__
#define __FC_API_H__

#include <mv_types.h>

typedef struct
{

    u8 *weightLines;
    int weightLines_offset;

    u8 *vector;
    u8 *bias;
    u8 inputBPP;

    u8 with_relu;

    u32 linesNo;
    u8 *output;
    u8 outputBPP;
    u32 inputWidth;
} fc_info;

#endif
