#ifndef __POOLING_API_H__
#define __POOLING_API_H__

#include <mv_types.h>
// #include "ddr_functions_types.h"

typedef struct {
    int src_addr;
    int dst_addr;
    int elements;
    int buffer_elements;
} pool_buffer_info;

typedef struct {
    u8 *input;
    u16 input_channel_offset;
    u8 inputBPP;

    u8 *output;
    u16 output_channel_offset;
    u8 outputBPP;

    u16 channels;
    u16 line_width;
    u32 ddr_function;
    u8 kernel_h;
    u8 type;

    int center;
    int right;
    int bottom;
    int corner;

    int in_src_width, in_src_stride;
    int in_dst_width, in_dst_stride;

    int out_src_addr, out_src_width, out_src_stride;
    int out_dst_addr, out_dst_width, out_dst_stride;
    int out_buffer_elements, out_elements;

    u8 in_buffers_num;
    u8 splits;

    pool_buffer_info in_buffers[8];
   
} pool_info;

#endif