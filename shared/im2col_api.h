#ifndef __IM2COL_API_H__
#define __IM2COL_API_H__

#include <mv_types.h>

// Defines and typedefs
// ----------------------------------------------------------------------------

typedef struct {
    u8 *input;
    int input_channel_offset;
    u8 inputBPP;

    u16 tiles;
    u8 *input_column;
    u8 *output;
    int output_channel_offset;
    u8 outputBPP;

    u8 *conv_weights;
    int conv_weights_offset;
    int conv_weights_channel_offset; 

    u8 *conv_biases;
    u8 kernelBPP;

    u16 channels;
    u8 ddr_function;
    u8 kernel_h;
    u8 kernel_w;    
    u8 with_relu;

    //upgrade begin
    u16 maps;
    u8 c_group;
    u8 in_stride;
    //upgrade end

    u16 offset;
    u16 in_col_height;
    u16 in_row_width;
    u16 weight_col_height;
    u16 weight_row_width;

    
} im2col_info;

#endif