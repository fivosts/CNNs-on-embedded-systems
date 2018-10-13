#ifndef _DDR_FUNCTIONS_H_
#define _DDR_FUNCTIONS_H_

// 1: Includes
// ----------------------------------------------------------------------------
#include "ddr_functions_types.h"
#include "stddef.h"
// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------

typedef void (*FUNCPTR_T) (void *);
typedef FUNCPTR_T (*J_FUNCPTR_T)(s32);

typedef void (*ACCPTR_T) (half **dst,
                          half **src,
                          u32 width);

typedef void (*ACCSINGLEPTR_T) (half *dst,
                                half src,
                                u32 width);

typedef void (*CONVPTR_T) (void **a,
                           void **b,
                           void  *c,
                           u32 width);

typedef void (*POOLPTR_T) (void **a,
                           void **b,
                           u32 width);

typedef float (*VECDOTPTR_T) (half *vec1,
                              half *vec2,
                              u32 width);

typedef float (*IM2COLPTR_T) (half *vec1,
                              half *vec2,
                              u32 width);

typedef void (*LRNPTRGENERIC_T) (fp16** input, fp16** output, 
                              u16 channels, u16 pixel_batch, 
                                  u8 local_ratio, fp16 alpha, fp16 beta);

typedef void (*LRNSPECIFICPTR_T) (fp16** input, fp16** output, 
                                    u16 channels, u16 pixel_batch);

typedef int (*PRINTF_PTR) (char *str, ...); 

typedef void (*BZERO_PTR) (void *s, size_t n);

enum lib_func_cat {func_common, func_conv, func_pool, func_acc, func_fc, func_lrn, func_im2col};

struct lib_conv {
    const char *name;
    enum data4_datatype type;
    enum kernel_size kernel_size;
    enum kernel_stride stride;
    FUNCPTR_T func;
};

struct lib_pool {
    const char *name;
    enum data2_datatype type;
    enum kernel_size kernel_size;
    enum kernel_stride stride;
    enum pooling_type ptype;
    FUNCPTR_T func;
};

struct lib_acc {
    const char *name;
    enum data2_datatype type;
    FUNCPTR_T func;
};

struct lib_fc {
    const char *name;
    enum data4_datatype type;
    FUNCPTR_T func;
};

struct lib_lrn {
	const char *name;
	FUNCPTR_T func;
};

struct lib_common {
    const char *name;
    FUNCPTR_T func;
};


struct lib_im2col {
    const char *name;
    FUNCPTR_T func;
};

struct lib_function {
    enum lib_func_cat category;
    union {
        struct lib_conv conv;
        struct lib_pool pool;
        struct lib_acc acc;
        struct lib_fc fc;
        struct lib_lrn lrn;
        struct lib_common cm;
        struct lib_im2col im2col;
    } cat;

};

// Convolution 1x1
#define MV_conv1x1s1hhhh 0

// Convolution 3x3
#define MV_conv3x3s1hhhh 1

#define MV_conv3x3s2hhhh 2

#define MV_conv3x3s3hhhh 3

#define MV_conv3x3s4hhhh 4

#define MV_conv3x3s8hhhh 5

// Convolution 5x5
#define MV_conv5x5s1hhhh 6

#define MV_conv5x5s2hhhh 7

#define MV_conv5x5s3hhhh 8

#define MV_conv5x5s4hhhh 9

#define MV_conv5x5s8hhhh 10

// Convolution 7x7
#define MV_conv7x7s1hhhh 11

#define MV_conv7x7s2hhhh 12

#define MV_conv7x7s4hhhh 13

#define MV_conv7x7s8hhhh 14

// Convolution 9x9
#define MV_conv9x9s1hhhh 15

#define MV_conv9x9s2hhhh 16

#define MV_conv9x9s3hhhh 17

#define MV_conv9x9s4hhhh 18

#define MV_conv9x9s8hhhh 19

// Convolution 11x11
#define MV_conv11x11s1hhhh 20

#define MV_conv11x11s2hhhh 21

#define MV_conv11x11s4hhhh 22

#define MV_conv11x11s8hhhh 23

// Accumulation
#define MV_accumulate_hh_single32 24

#define MV_accumulate_hh_single64 25

#define MV_accumulate_hh 26

#define MV_accumulate_hh_withReLU 27

// Pooling
#define MV_avgPool3x3s2hh 28

#define MV_avgPool7x7s1hh 29
  
#define MV_avgPool14x14s1hh 30

#define MV_maxPool2x2s2hh 31

#define MV_maxPool3x3s1hh 32

#define MV_maxPool3x3s2hh 33    

#define MV_maxPool3x3s2hh_withReLU 34

//Inner Product
#define MV_vecvecmul_hhff 35

//LRN
#define FT_LRN_AcrossChannels_generic 36

#define FT_LRN_AcrossChannels_LR5_A0_0001_B_0_75 37

// Common Functions
#define CM_printf 38

#define CM_bzero 39

#define CM_conv_ddr 40

#define CM_pool_ave_ddr 41

#define CM_pool_max_ddr 42

#define CM_fc_ddr 43

#define CM_lrn_ddr_generic 44

#define CM_lrn_ddr_LR5_A0_0001_B_0_75 45

#define FT_im_col 46

#endif