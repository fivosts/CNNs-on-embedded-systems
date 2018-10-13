#ifndef __CONV_API_H__
#define __CONV_API_H__

#include <mv_types.h>

// Defines and typedefs
// ----------------------------------------------------------------------------

typedef struct{
    int src_addr;
    int dst_addr;
    int elements;
    int buffer_elements;
}conv_buffer_info;

typedef struct {
    u8 *input;
    u16 input_channel_offset;
    u8 inputBPP;

    u8 *output;
    u16 output_channel_offset;
    u8 outputBPP;

    u8 *conv_weights;
    u16 conv_weights_offset;
    u8 conv_weights_channel_offset; 

    u8 *conv_biases;
    u8 kernelBPP;

   	u16 channels;
    u32 ddr_function;
    u8 kernel_h;
    u8 kernel_w;
    
    u8 coalescing_num;
    u8 with_relu;

    int in_buffer_shift;
    u16 line_width;

    int out_src_addr, out_src_width, out_src_stride;
    int out_dst_addr, out_dst_width, out_dst_stride;
    int out_buffer_elements, out_elements;

	u16 maps;
    
	u8 c_group;
	u8 splits;

    int in_src_width, in_src_stride;
    int in_dst_width, in_dst_stride;
    
    u8 in_stride;
    int in_buffers_num;
    conv_buffer_info in_buffers[8];

} conv_info;

#endif