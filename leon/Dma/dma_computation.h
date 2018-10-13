#ifndef __NETWORK_DMA_COMPUTATION_H__
#define __NETWORK_DMA_COMPUTATION_H__

#include <pool_api.h>
#include <conv_api.h>
#include <im2col_api.h>

void 
pooling_output_dim(
    int *output_height,
    int *output_width,
    int input_height,
    int input_width,
    int K_H,
    int K_W,
    int P_H,
    int P_W,
    int S_H,
    int S_W
);

void 
pool_prepare_dma(
    pool_info *const pooling_buf,
    int input_height,
    int input_width,
    int lines,
    int line_id,
    int K_H,
    int K_W,
    int P_H,
    int P_W,
    int S_H,
    int S_W,
    int alignment
);

void
pool_ave_edges(
    int *center,
    int *right,
    int *bottom,
    int *corner,

    int pooled_height_,
    int pooled_width_,
    int height_,
    int width_,
    int kernel_h_,
    int kernel_w_,
    int pad_h_,
    int pad_w_,
    int stride_h_,
    int stride_w_
);

void 
convolution_output_dim(
    int *output_height,
    int *output_width,
    int input_height,
    int input_width,
    int K_H,
    int K_W,
    int P_H,
    int P_W,
    int S_H,
    int S_W
);

void 
conv_prepare_dma(
    conv_info *const conv_buf,
    int input_height,
    int input_width,
    int lines,
    int line_id,
    int K_H,
    int K_W,
    int P_H,
    int P_W,
    int S_H,
    int S_W,
    int alignment
);

u8* NewMapWithPad(u8* pointer, int width, int height, int kernel, int stride, int padding, int channels);

u8* InputstoColumns(u8* pointer, int width, int height, int kernel, int stride, int padding, int channels);


#endif//__NETWORK_DMA_COMPUTATION_H__
