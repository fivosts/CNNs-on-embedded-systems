#ifndef __DMA_COMPUTATION_DEFINES_H__
#define __DMA_COMPUTATION_DEFINES_H__

#define max(a,b)            \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b)            \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

struct rect_plane {
    int x1, y1;
    int x2, y2;
};

struct extended_rect_plane {
    struct rect_plane plane;
    int junk;
    int last_line_width;
};

struct rect_padding {
    int top, bottom;
    int left, right;
};

struct rect_dim {
    int width;
    int height;
};

struct dma_buffer_descr {
    int start_padding_num;
    int end_padding_num;  
    int start_data_index;  
    int data_num;         
    int left_padding;
    int right_padding;
};

struct buffer_descr {
    int padding_start, padding_middle, padding_end;
    int buffer_elements;
    int start_index;
    int width;
};

struct dma_instr {
    int src_addr;
    int src_width;
    int src_stride;
    
    int dst_addr;
    int dst_width;
    int dst_stride;
    
    int elements;
};

#endif//__NETWORK_DMA_COMPUTATION_DEFINES_H__