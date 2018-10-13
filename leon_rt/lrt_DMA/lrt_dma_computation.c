#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "mv_types.h"
#include <math.h>

#include <pool_api.h>
#include <conv_api.h>

#include "dma_computation.h"
#include "dma_computation_defines.h"
#include <stdio.h>

#define INPUTCOL __attribute__((section(".ddr_direct.data"), aligned (16)))

u8 INPUTCOL data[15000000];
u8 INPUTCOL data_pad[500000];

static int 
ceil_int(
    int numerator,
    int denominator
)
{

    int div = numerator / denominator;
    if (((numerator ^ denominator) >= 0) && (numerator % denominator != 0))
        div++;

    return div;
}

static int 
floor_int(
    int numerator,
    int denominator
)
{
    int div = numerator / denominator;
    if (((numerator ^ denominator) < 0) && (numerator % denominator != 0))
        div--;

    return div;
}

static int 
intersection_exists(
    const struct rect_plane *const p1, 
    const struct rect_plane *const p2
)
{
    return (p1->x1 < p2->x2 &&
            p1->x2 > p2->x1 &&
            p1->y1 < p2->y2 &&
            p1->y2 > p2->y1);
}

static void 
get_intersection(
    struct rect_plane *const intersection, 
    const struct rect_plane *const p1, 
    const struct rect_plane *const p2
)
{   
    intersection->y1 = max(p1->y1, p2->y1);
    intersection->y2 = min(p1->y2, p2->y2);

    intersection->x1 = max(p1->x1, p2->x1);
    intersection->x2 = min(p1->x2, p2->x2);
}

static int 
is_inside(
    const struct rect_plane *const parent, 
    const struct rect_plane *const child
)
{
    return (parent->x1 <= child->x1 &&
            parent->y1 <= child->y1 &&
            parent->x2 >= child->x2 &&
            parent->y2 >= child->y2);
}

static void
create_base(
    struct rect_plane *const base,
    struct rect_plane *const image,
    const struct rect_padding *const padding,
    const struct rect_dim *const image_dim
)
{
    base->x1 = 0;
    base->y1 = 0;
    base->x2 = image_dim->width + padding->left + padding->right;
    base->y2 = image_dim->height + padding->top + padding->bottom;
    
    image->x1 = padding->left;
    image->y1 = padding->top;
    image->x2 = image_dim->width + padding->left;
    image->y2 = image_dim->height + padding->top;
}

static int
kernel_fits_rect(
    const struct rect_plane *const base,
    const int kernel_h,
    const int kernel_w
)
{
    return ((base->x2 - base->x1) >= kernel_w &&
            (base->y2 - base->y1) >= kernel_h);
}

static void
define_area(
    struct rect_plane *const area,
    const int x,
    const int y,
    const struct rect_dim *const dim
)
{
    area->x1 = x;
    area->y1 = y;
    area->x2 = dim->width + area->x1;
    area->y2 = dim->height + area->y1; 
}

static int
rect_width(
    const struct rect_plane *const p
)
{
    return (p->x2 - p->x1);
}

static int
rect_height(
    const struct rect_plane *const p
)
{
    return (p->y2 - p->y1);
}

static int 
round_up(
    const int num_to_round, 
    const int multiple
) 
{
    assert(multiple);
    return ((num_to_round + multiple - 1) / multiple) * multiple;
}

// Συμπιέζουμε το padding και ευθυγραμμίζουμε τις γραμμές ώστε να απέχουν
// μεταξύ τους πολλαπλάσιο του ορίσματος.
static void 
align_lines(
    struct extended_rect_plane *const new_base,
    const struct rect_plane *const base,
    const struct rect_plane *const image,
    const int stride_w,
    const int kernel_w,
    const int align
)
{   
    int left_padding = image->x1 - base->x1;
    int right_padding_old = base->x2 - image->x2;
    int spaces = max(left_padding, right_padding_old);

    int left_ends = rect_width(base);
    int right_begins = left_ends - min(left_padding, right_padding_old);
    int last_mask_begins = left_ends - kernel_w;
    
    int i = 0;
    
    int tmp = last_mask_begins + stride_w * i;
    while (tmp < right_begins || (tmp % align) != 0) {
        i++;
        tmp = last_mask_begins + stride_w * i;
    }
    
    int right_begins_old = right_begins;
    int right_begins_new = last_mask_begins + stride_w * i;
    
    int extra_padding = right_begins_new - right_begins_old;
    int right_padding_new = spaces + extra_padding - left_padding;

    int junk = i-1;

    new_base->plane = *base;
    new_base->plane.x2 += right_padding_new - right_padding_old;
    new_base->junk = junk;
    new_base->last_line_width = rect_width(base);
    
}

static int
align_buffer(
    const int boundary,
    const int kernel_w
)
{

    assert(kernel_w % 2 == 1);

    return round_up(kernel_w/2, boundary) - kernel_w/2;
}

static void
reverse_area_transform(
    struct rect_plane *const res,
    const struct rect_plane *const window,
    const struct rect_plane *const base_in,
    const struct rect_plane *const base_out,
    
    const int stride_h,
    const int stride_w,
    
    const int kernel_h,
    const int kernel_w
)
{
    res->x1 = stride_w*(window->x1 - base_out->x1) + base_in->x1;
    res->y1 = stride_h*(window->y1 - base_out->y1) + base_in->y1;
    
    res->x2 = res->x1 + kernel_w + stride_w * (window->x2 - window->x1 - 1);
    res->y2 = res->y1 + kernel_h + stride_h * (window->y2 - window->y1 - 1);
    
    assert(res->x2 - res->x1 >= kernel_w);
    assert(res->y2 - res->y1 >= kernel_h);
}

static int
input_buffer_size(
    const struct dma_buffer_descr *const buffer,
    const struct extended_rect_plane *const aligned_area
)
{
    int width = rect_width(&(aligned_area->plane));
    
    int lines = buffer->start_padding_num + 
                buffer->data_num + buffer->end_padding_num;
    
    return (lines - 1) * width + aligned_area->last_line_width;
}

static int
output_buffer_size(
    const struct rect_dim *const output_dim,
    const struct extended_rect_plane *const ebase
)
{
    
    return (output_dim->height - 1) * 
                         (output_dim->width + ebase->junk) + output_dim->width;
}

static void
calculate_in_dma(
    struct dma_instr *const dma,
    const struct rect_plane *const image,
    const struct rect_plane *const intersection,
    const struct dma_buffer_descr *const buffer,
    const int stride_h
)
{
    dma->src_addr = (buffer->start_data_index - image->y1) * rect_width(image) + 
                    (intersection->x1 - image->x1);
    dma->src_width = rect_width(intersection);
    dma->src_stride = stride_h * rect_width(image);

    int width = buffer->left_padding + dma->src_width + buffer->right_padding;

    dma->dst_addr = buffer->left_padding +  width * buffer->start_padding_num;
    dma->dst_width = dma->src_width;
    dma->dst_stride = width;
    
    dma->elements = rect_width(intersection) * buffer->data_num;
}
    
static void
calculate_out_dma(
    struct dma_instr *const dma,
    const struct extended_rect_plane *const ebase,
    const struct rect_plane *const result_plane,
    const struct rect_plane *const window
)
{
    dma->src_addr = 0;
    dma->src_width = rect_width(window);
    dma->src_stride = dma->src_width + ebase->junk;

    dma->dst_addr = (window->y1 - result_plane->y1) * rect_width(result_plane) + 
                    (window->x1 - result_plane->x1);
    dma->dst_width = dma->src_width;
    dma->dst_stride = rect_width(result_plane);
    
    dma->elements = rect_width(window) * rect_height(window);
}

static void
buffer_split(
    struct dma_buffer_descr *const buffer_descr,
    const struct extended_rect_plane *const aligned_area,
    const struct rect_plane *const intersection,
    const int start_offset,
    const int stride_h_
)
{
    int start, end;

    int pad_lines_start = 0;
    int pad_lines_end = 0;
    
    int data_lines_start = 0;
    int data_lines_num = 0;
    
    start = aligned_area->plane.y1 + start_offset;
    end = intersection->y1;
    
    while (start < end) {
        pad_lines_start++;
        start += stride_h_;
    }
    
    end = intersection->y2 - 1;
    data_lines_start = start;
    
    while (start <= end) {
        data_lines_num++;
        start += stride_h_;
    }
    
    end = aligned_area->plane.y2 - 1;
    
    while (start <= end) {
        pad_lines_end++;
        start += stride_h_;
    }
    

    buffer_descr->start_padding_num = pad_lines_start;
    buffer_descr->end_padding_num = pad_lines_end;
    buffer_descr->start_data_index = data_lines_start;
    buffer_descr->data_num = data_lines_num;
    buffer_descr->left_padding = intersection->x1 - aligned_area->plane.x1;
    buffer_descr->right_padding = aligned_area->plane.x2 - intersection->x2;
    
}

void
pooling_dimensions(
    struct rect_dim *const output_dim,
    struct rect_padding *const padding,
    const struct rect_dim *const image_dim,
    
    const int pad_h,
    const int pad_w,
    
    const int stride_h,
    const int stride_w,
    
    const int kernel_h,
    const int kernel_w
)
{
    int height = image_dim->height;
    int width = image_dim->width;

    int pooled_height_ = ceil_int(height + 2 * pad_h - kernel_h, stride_h) + 1;
    int pooled_width_ = ceil_int(width + 2 * pad_w - kernel_w, stride_w) + 1;
    
    if (pad_h || pad_w) {
        assert(pad_h < kernel_h);
        assert(pad_w < kernel_w);

        if ((pooled_height_ - 1) * stride_h >= height + pad_h) {
            --pooled_height_;
        }
        
        if ((pooled_width_ - 1) * stride_w >= width + pad_w) {
            --pooled_width_;
        }
        
        assert((pooled_height_ - 1) * stride_h < height + pad_h);
        assert((pooled_width_ - 1) * stride_w < width + pad_w);
    }
    
    output_dim->width = pooled_width_;
    output_dim->height = pooled_height_;
    
    padding->top = pad_h,
    padding->left = pad_w,
    padding->bottom = (pooled_height_ - 1) * stride_h + 
                                kernel_h - height - pad_h;
    
    padding->right = (pooled_width_ - 1) * stride_w + 
                                kernel_w - width - pad_w;
                                
    if (padding->bottom < 0) padding->bottom = 0;
    if (padding->right < 0) padding->right = 0;  
}

void
convolution_dimensions(
    struct rect_dim *const output_dim,
    struct rect_padding *const padding,
    const struct rect_dim *const image_dim,
    
    const int pad_h,
    const int pad_w,
    
    const int stride_h,
    const int stride_w,
    
    const int kernel_h,
    const int kernel_w
)
{
    int height = image_dim->height;
    int width = image_dim->width;

    int convolved_height_ = floor_int(height + 2*pad_h - kernel_h, stride_h) + 1;
    int convolved_width_ = floor_int(width + 2*pad_w - kernel_w, stride_w) + 1;
    
    output_dim->height = convolved_height_;
    output_dim->width = convolved_width_;
    
    
    padding->top = pad_h,
    padding->left = pad_w,
    padding->bottom = (convolved_height_ - 1) * stride_h + 
                                kernel_h - height - pad_h;
    
    padding->right = (convolved_width_ - 1) * stride_w + 
                                kernel_w - width - pad_w;
                                
    if (padding->bottom < 0) padding->bottom = 0;
    if (padding->right < 0) padding->right = 0;  
}


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
)
{
    struct rect_dim image_dim = {
        .width = input_width,
        .height = input_height
    };
    
    struct rect_dim output_dim;
    struct rect_padding pool_padding;
    
    pooling_dimensions(&output_dim, &pool_padding, &image_dim, 
                                    P_H, P_W, S_H, S_W, K_H, K_W);

    *output_height = output_dim.height;
    *output_width = output_dim.width;
}

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
)
{
    struct rect_dim image_dim = {
        .width = input_width,
        .height = input_height
    };
    
    struct rect_dim output_dim;
    struct rect_padding conv_padding;
    
    convolution_dimensions(&output_dim, &conv_padding, &image_dim, 
                                        P_H, P_W, S_H, S_W, K_H, K_W);

    *output_height = output_dim.height;
    *output_width = output_dim.width;
}

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
)
{
    struct rect_dim image_dim = {
        .width = input_width,
        .height = input_height
    };
    
    struct rect_dim window_dim;
    int window_x, window_y;

    struct rect_dim output_dim;
    struct rect_padding pool_padding;
    
    pooling_dimensions(&output_dim, &pool_padding, &image_dim, P_H, P_W, S_H, S_W, K_H, K_W);

    if(lines == 1)
    {
        window_dim.width = output_dim.width;
        window_dim.height = output_dim.height;
        window_x = 0;
        window_y = 0;
    }
    else 
    {
        if((output_dim.height % lines) == 0){            
            if(line_id==0){
                window_dim.width = output_dim.width;
                window_dim.height = output_dim.height/lines;
                window_x = 0;
                window_y = 0;
            }
            else{
                window_dim.width = output_dim.width;
                window_dim.height = (output_dim.height/lines);
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;            
            }
        }
        else{
            if (line_id == lines-1){
                window_dim.width = output_dim.width;
                window_dim.height = output_dim.height - (output_dim.height/lines)*line_id;
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;
            }   
            else{
                window_dim.width = output_dim.width;
                window_dim.height = (output_dim.height/lines);
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;       
            } 
            
        }
    }
    
    
    struct rect_plane base, image, area, intersection;
    struct extended_rect_plane aligned_area;
    
    struct rect_plane window, result_plane;
    struct dma_instr in_dma, out_dma;
    
    create_base(&base, &image, &pool_padding, &image_dim);
    assert(kernel_fits_rect(&base, K_H, K_W));
    
    define_area(&result_plane, 0, 0, &output_dim);
    define_area(&window, window_x, window_y, &window_dim);

    reverse_area_transform(&area, &window, &base, &result_plane, S_H, S_W, K_H, K_W);
    assert(is_inside(&base, &area));

    get_intersection(&intersection, &image, &area);
    assert(intersection_exists(&image, &area));
    assert(kernel_fits_rect(&area, K_H, K_W));
    
    align_lines(&aligned_area, &area, &intersection, S_W, K_W, alignment);

    int buffer_splits_num = output_dim.height > 1 ?  min(S_H, K_H) : 1;
    int buffer_splits_stride = buffer_splits_num > 1 ? S_H : 1;

    pooling_buf->in_buffers_num = buffer_splits_stride;
    
    for (int i = 0; i < buffer_splits_num; i++)
    {
        struct dma_buffer_descr buffer_descr;
        buffer_split(&buffer_descr, &aligned_area, &intersection, i, buffer_splits_stride);
        
        calculate_in_dma(&in_dma, &image, &intersection, &buffer_descr, buffer_splits_stride);
        
        pooling_buf->in_buffers[i] = (pool_buffer_info) {
            .src_addr = in_dma.src_addr,
            .dst_addr = in_dma.dst_addr,
            .elements = in_dma.elements,
            .buffer_elements = input_buffer_size(&buffer_descr, &aligned_area)
        };

        
        if (i == buffer_splits_num-1) {
            pooling_buf->line_width = rect_width(&(aligned_area.plane));
            pooling_buf->in_src_width = in_dma.src_width;
            pooling_buf->in_src_stride = in_dma.src_stride;
            pooling_buf->in_dst_width = in_dma.dst_width;
            pooling_buf->in_dst_stride = in_dma.dst_stride;
        
            calculate_out_dma(&out_dma, &aligned_area, &result_plane, &window);

            pooling_buf->out_src_addr = out_dma.src_addr;
            pooling_buf->out_src_width = out_dma.src_width;
            pooling_buf->out_src_stride = out_dma.src_stride;
            pooling_buf->out_dst_addr = out_dma.dst_addr;
            pooling_buf->out_dst_width = out_dma.dst_width;
            pooling_buf->out_dst_stride = out_dma.dst_stride;
            pooling_buf->out_buffer_elements = round_up(output_buffer_size(&window_dim, &aligned_area), 8);
            pooling_buf->out_elements = out_dma.elements;
            pooling_buf->splits = lines;
        } 
    }
    
}

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
)
{
    int BUFFER_ALIGNMENT_BOUNDARY = 16;

    struct rect_dim image_dim = {
        .width = input_width,
        .height = input_height
    };
    
    struct rect_dim window_dim;
    int window_x, window_y;

    struct rect_plane base, image;
    
    struct rect_plane area, intersection;
    struct extended_rect_plane aligned_area;
    
    struct rect_plane window, result_plane;
    struct dma_instr in_dma, out_dma;

    struct rect_dim output_dim;
    struct rect_padding conv_padding;
    
    convolution_dimensions(&output_dim, &conv_padding, 
        &image_dim, P_H, P_W, S_H, S_W, K_H, K_W);

    if(lines == 1)
    {
        window_dim.width = output_dim.width;
        window_dim.height = output_dim.height;
        window_x = 0;
        window_y = 0;
    }
    else 
    {
        if((output_dim.height % lines) == 0){            
            if(line_id==0){
                window_dim.width = output_dim.width;
                window_dim.height = output_dim.height/lines;
                window_x = 0;
                window_y = 0;
            }
            else{
                window_dim.width = output_dim.width;
                window_dim.height = (output_dim.height/lines);
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;            
            }
        }
        else{
            if (line_id == lines-1){
                window_dim.width = output_dim.width;
                window_dim.height = output_dim.height - (output_dim.height/lines)*line_id;
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;
            }   
            else{
                window_dim.width = output_dim.width;
                window_dim.height = (output_dim.height/lines);
                window_x = 0;
                window_y = (output_dim.height/lines)*line_id;       
            } 
            
        }
    }
    conv_buf->splits = lines;

    create_base(&base, &image, &conv_padding, &image_dim);

    assert(kernel_fits_rect(&base, K_H, K_W));
    
    define_area(&result_plane, 0, 0, &output_dim);
    define_area(&window, window_x, window_y, &window_dim);

    reverse_area_transform(&area, &window, &base, &result_plane, S_H, S_W, K_H, K_W);

    get_intersection(&intersection, &image, &area);

    assert(intersection_exists(&image, &area));
    assert(kernel_fits_rect(&area, K_H, K_W));

    align_lines(&aligned_area, &area, &intersection, S_W, K_W, alignment);
    int in_buffer_shift = align_buffer(BUFFER_ALIGNMENT_BOUNDARY, K_W);

    int buffer_splits_num = output_dim.height > 1 ? min(S_H, K_H) : 1;
    int buffer_splits_stride = buffer_splits_num > 1 ? S_H : 1;

    conv_buf->in_stride = buffer_splits_stride;
    conv_buf->in_buffers_num = buffer_splits_num;                        
    if (conv_buf->in_stride != 1){

        for (int i = 0; i < conv_buf->in_buffers_num; i++){
            struct dma_buffer_descr buffer_descr;
            buffer_split(&buffer_descr, &aligned_area, &intersection, i, buffer_splits_stride);
            calculate_in_dma(&in_dma, &image, &intersection, &buffer_descr, buffer_splits_stride);

            conv_buf->in_buffers[i] = (conv_buffer_info) {   
                .src_addr = in_dma.src_addr,
                .dst_addr = in_buffer_shift + in_dma.dst_addr,
                .buffer_elements = in_buffer_shift + input_buffer_size(&buffer_descr, &aligned_area),
                .elements = in_dma.elements
            };
                    
            if (i == conv_buf->in_buffers_num - 1){
                conv_buf->line_width = rect_width(&(aligned_area.plane));
                conv_buf->in_src_width = in_dma.src_width;
                conv_buf->in_src_stride = in_dma.src_stride;
                conv_buf->in_dst_width = in_dma.dst_width;
                conv_buf->in_dst_stride = in_dma.dst_stride;
                conv_buf->in_buffer_shift = in_buffer_shift;
                conv_buf->splits = lines;                

                calculate_out_dma(&out_dma, &aligned_area, &result_plane, &window);

                conv_buf->out_src_addr = out_dma.src_addr;
                conv_buf->out_src_width = out_dma.src_width;
                conv_buf->out_src_stride = out_dma.src_stride;
                conv_buf->out_dst_addr = out_dma.dst_addr;
                conv_buf->out_dst_width = out_dma.dst_width;
                conv_buf->out_dst_stride = out_dma.dst_stride;
                conv_buf->out_buffer_elements = round_up(output_buffer_size(&window_dim, &aligned_area), 8);
                conv_buf->out_elements = out_dma.elements; 
                
            }
            
        }
    }
    else{
        struct dma_buffer_descr buffer_descr;
        buffer_split(&buffer_descr, &aligned_area, &intersection, 0, 1);
        calculate_in_dma(&in_dma, &image, &intersection, &buffer_descr, 1);

        conv_buf->in_buffers[0] = (conv_buffer_info) {   
            .src_addr = in_dma.src_addr,
            .dst_addr = in_buffer_shift + in_dma.dst_addr,
            .buffer_elements = in_buffer_shift + input_buffer_size(&buffer_descr, &aligned_area),
            .elements = in_dma.elements
        };
                
        conv_buf->line_width = rect_width(&(aligned_area.plane));
        conv_buf->in_src_width = in_dma.src_width;
        conv_buf->in_src_stride = in_dma.src_stride;
        conv_buf->in_dst_width = in_dma.dst_width;
        conv_buf->in_dst_stride = in_dma.dst_stride;
        conv_buf->in_buffer_shift = in_buffer_shift;
    
        calculate_out_dma(&out_dma, &aligned_area, &result_plane, &window);

        conv_buf->out_src_addr = out_dma.src_addr;
        conv_buf->out_src_width = out_dma.src_width;
        conv_buf->out_src_stride = out_dma.src_stride;
        conv_buf->out_dst_addr = out_dma.dst_addr;
        conv_buf->out_dst_width = out_dma.dst_width;
        conv_buf->out_dst_stride = out_dma.dst_stride;
        conv_buf->out_buffer_elements = round_up(output_buffer_size(&window_dim, &aligned_area), 8);
        conv_buf->out_elements = out_dma.elements; 
    }
}


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
)
{
    int ph, pw;
    int hstart, wstart;
    int hend, wend;

    *center = kernel_h_ * kernel_w_;

    ph = 0;
    pw = pooled_width_ - 1;
    hstart = ph * stride_h_ - pad_h_;
    wstart = pw * stride_w_ - pad_w_;
    hend = min(hstart + kernel_h_, height_ + pad_h_);
    wend = min(wstart + kernel_w_, width_ + pad_w_);
    *right = (hend - hstart) * (wend - wstart);

    ph = pooled_height_ - 1;
    pw = 0;
    hstart = ph * stride_h_ - pad_h_;
    wstart = pw * stride_w_ - pad_w_;
    hend = min(hstart + kernel_h_, height_ + pad_h_);
    wend = min(wstart + kernel_w_, width_ + pad_w_);
    *bottom = (hend - hstart) * (wend - wstart);

    ph = pooled_height_ - 1;
    pw = pooled_width_ - 1;
    hstart = ph * stride_h_ - pad_h_;
    wstart = pw * stride_w_ - pad_w_;
    hend = min(hstart + kernel_h_, height_ + pad_h_);
    wend = min(wstart + kernel_w_, width_ + pad_w_);
    *corner = (hend - hstart) * (wend - wstart);
}


u8* NewMapWithPad(u8* previous, int width, int height, int kernel, int stride, int padding, int channels){

    
    int out_height = floor_int(height + 2*padding - kernel, stride) + 1;
    int out_width = floor_int(width + 2*padding - kernel, stride) + 1;
       
    int paddingbottom = (out_height - 1) * stride + kernel - height - padding;
    int paddingright = (out_width - 1) * stride + kernel - width - padding;

    int new_width = width + padding + paddingright;
    int new_height = height + padding + paddingbottom;
                         

    for (int ch = 0; ch < channels; ch++){
        for (int y_column = 0; y_column < new_height; y_column++){
            for (int x_pad = 0; x_pad < padding; x_pad ++){    
                data_pad[2 * ch * (new_width * new_height) + 2 * y_column * new_width + 2 * x_pad] = 0;
                data_pad[2 * ch * (new_width * new_height) + 2 * y_column * new_width + 2 * x_pad + 1] = 0; 
            }    
            for (int x_pad = (new_width - paddingright); x_pad < new_width; x_pad++){   
                data_pad[2 * ch * (new_width * new_height) + 2 * y_column * new_width + 2 * x_pad] = 0;
                data_pad[2 * ch * (new_width * new_height) + 2 * y_column * new_width + 2 * x_pad + 1] = 0; 
            }    
        }
        for (int x_column = padding; x_column < new_width - padding; x_column++){
            for (int y_pad = 0; y_pad < padding; y_pad++){
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column] = 0;
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column + 1] = 0; 
            }    
            for (int y_pad = (new_height - paddingbottom); y_pad < new_height; y_pad++){
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column] = 0;
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column + 1] = 0; 
            }    
        }
        for (int y_pad = (padding); y_pad < (new_height - paddingbottom); y_pad++){
            for (int x_column = (padding); x_column < (new_width - paddingright); x_column++){
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column] = previous[2 * ch * (width*height) + 2 * (y_pad * width - padding * width) + 2 * (x_column - padding)];
                data_pad[2 * ch * (new_width * new_height) + 2 * y_pad * new_width + 2 * x_column + 1] = previous[2 * ch * (width*height) + 2 * (y_pad * width - padding * width) + 2 * (x_column - padding) + 1];
            }
        }    
    }

    for (int ch = 0; ch < channels; ch++){
        for(int y_kernel = 0; y_kernel < kernel; y_kernel++){
            for(int x_kernel = 0; x_kernel < kernel; x_kernel++){
                for(int y_stride = 0; y_stride < out_height; y_stride++){
                    for(int x_stride=0; x_stride < out_width; x_stride++){
                        data[(2*ch*(kernel*kernel*out_width*out_height))+(2*y_kernel*(kernel*out_height*out_width))+(2*x_kernel*(out_width*out_height))+(y_stride*2*out_width)+(2*x_stride)] = data_pad[(x_stride*2*stride+x_kernel*2)+(stride*2*y_stride*new_width)+(y_kernel*new_height*2)+(ch*(new_width*new_height)*2)];
                        data[(2*ch*(kernel*kernel*out_width*out_height))+(2*y_kernel*(kernel*out_height*out_width))+(2*x_kernel*(out_width*out_height))+(y_stride*2*out_width)+(2*x_stride)+1] = data_pad[(x_stride*2*stride+x_kernel*2)+(stride*2*y_stride*new_width)+(y_kernel*new_height*2)+(ch*(new_width*new_height)*2)+1]; 
                    }
                }
            }
        }
    }

    return data;
}

u8* InputstoColumns(u8* pointer, int width, int height, int kernel, int stride, int padding, int channels){

    int out_height = floor_int(height + 2 * padding - kernel, stride) + 1;
    int out_width = floor_int(width + 2 * padding - kernel, stride) + 1;
    
    
    for (int ch = 0; ch < channels; ch++){
        for(int y_kernel = 0; y_kernel < kernel; y_kernel++){
            for(int x_kernel = 0; x_kernel < kernel; x_kernel++){
                for(int y_stride = 0; y_stride < out_height; y_stride++){
                    for(int x_stride=0; x_stride < out_width; x_stride++){
                        data[(2*ch*(kernel*kernel*out_width*out_height))+(2*y_kernel*(kernel*out_height*out_width))+(2*x_kernel*(out_width*out_height))+(y_stride*2*out_width)+(2*x_stride)] = pointer[(x_stride*2*stride+x_kernel*2)+(stride*2*y_stride*width)+(y_kernel*height*2)+(ch*(width*height)*2)];
                        data[(2*ch*(kernel*kernel*out_width*out_height))+(2*y_kernel*(kernel*out_height*out_width))+(2*x_kernel*(out_width*out_height))+(y_stride*2*out_width)+(2*x_stride)+1] = pointer[(x_stride*2*stride+x_kernel*2)+(stride*2*y_stride*width)+(y_kernel*height*2)+(ch*(width*height)*2)+1]; 
                    }
                }
            }
        }
    }
    return data;
    
}
