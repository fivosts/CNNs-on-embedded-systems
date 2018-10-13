#ifndef __DDR_FUNCTIONS_EXPORTS_H__
#define __DDR_FUNCTIONS_EXPORTS_H__

// 1: Includes
// ----------------------------------------------------------------------------
#ifdef __MOVICOMPILE__
#include <mv_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include <accumulateFp16.h>

#include <../shave/ddr/asm/convolution1x1s1hhhh.h>   //googlenet

#include <convolution3x3Fp16ToFp16.h>
#include <convolution5x5Fp16ToFp16.h>
#include <convolution7x7Fp16ToFp16.h>
#include <convolution9x9Fp16ToFp16.h>

#include <convolution3x3s2hhhh.h>
#include <convolution3x3s3hhhh.h>
#include <convolution3x3s4hhhh.h>
#include <convolution3x3s8hhhh.h>

#include <convolution5x5s2hhhh.h>
#include <convolution5x5s3hhhh.h>
#include <convolution5x5s4hhhh.h>
#include <convolution5x5s8hhhh.h>

#include <convolution7x7s2hhhh.h>
#include <convolution7x7s4hhhh.h>
#include <convolution7x7s8hhhh.h>

#include <convolution9x9s2hhhh.h>
#include <convolution9x9s3hhhh.h>
#include <convolution9x9s4hhhh.h>
#include <convolution9x9s8hhhh.h>

#include <convolution11x11s1hhhh.h>
#include <convolution11x11s2hhhh.h>
#include <convolution11x11s4hhhh.h>
#include <convolution11x11s8hhhh.h>

#include "../shave/ddr/asm/accumulateFp16withReLU.h"
#include "../shave/ddr/asm/avgPool3x3s2hh.h"
#include "../shave/ddr/asm/convolution5x5U8ToFp16.h"
#include "../shave/ddr/asm/convolution5x5U8fToFp16.h"

#include <maxPool2x2s2hh.h>
#include "../shave/ddr/asm/maxPool3x3s2hh.h"
#include "../shave/ddr/asm/maxPool3x3s1hh.h"
#include "../shave/ddr/asm/maxPool3x3s2hhwithReLU.h"
// #include "../shave/ddr/asm/avgPool7x7xk.h"

#include "../shave/ddr/asm/accumulateFp16Single32.h"
#include "../shave/ddr/asm/accumulateFp16Single64.h"
#include "../shave/ddr/asm/vecVecDotProduct32.h"
#include "../shave/ddr/asm/Lrn_AcrossChannels_LR5.h"

// Routines that organize the computation with DMA
#include "../shave/ddr/ddr_conv.h"
#include "../shave/ddr/ddr_pool.h"
#include "../shave/ddr/ddr_fc.h"

//#include "../shave/ddr/ddr_lrn.h"
#include "../shave/ddr/local_resp_norm.h"

#endif //__MOVICOMPILE__

#include "ddr_functions.h"

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------

#ifdef __MOVICOMPILE__
#define LIB_FUNC_CONV(f_name, f_type, f_k_size, f_stride, f_func)              \
    {.category = func_conv,                                                    \
     .cat = {                                                                  \
        .conv = {                                                              \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .kernel_size = f_k_size,                                           \
            .stride = f_stride,                                                \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_POOL(f_name, f_type, f_k_size, f_stride, f_ptype, f_func)     \
    {.category = func_pool,                                                    \
     .cat = {                                                                  \
        .pool = {                                                              \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .kernel_size = f_k_size,                                           \
            .stride = f_stride,                                                \
            .ptype = f_ptype,                                                  \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_ACC(f_name, f_type, f_func)                                   \
    {.category = func_acc,                                                     \
     .cat = {                                                                  \
        .acc = {                                                               \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_FC(f_name, f_type, f_func)                                    \
    {.category = func_fc,                                                      \
     .cat = {                                                                  \
        .fc = {                                                                \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_LRN(f_name, f_func)                                   		   \
    {.category = func_lrn,                                                     \
     .cat = {                                                                  \
        .lrn = {                                                               \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_CM(f_name, f_func)                                            \
    {.category = func_common,                                                  \
     .cat = {                                                                  \
        .cm = {                                                                \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)f_func                                          \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_IM2COL(f_name, f_func)                                            \
    {.category = func_im2col,                                                  \
     .cat = {                                                                  \
        .cm = {                                                                \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#else

#define LIB_FUNC_CONV(f_name, f_type, f_k_size, f_stride, f_func)              \
    {.category = func_conv,                                                    \
     .cat = {                                                                  \
        .conv = {                                                              \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .kernel_size = f_k_size,                                           \
            .stride = f_stride,                                                \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_POOL(f_name, f_type, f_k_size, f_stride, f_ptype, f_func)     \
    {.category = func_pool,                                                    \
     .cat = {                                                                  \
        .pool = {                                                              \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .kernel_size = f_k_size,                                           \
            .stride = f_stride,                                                \
            .ptype = f_ptype,                                                  \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_ACC(f_name, f_type, f_func)                                   \
    {.category = func_acc,                                                     \
     .cat = {                                                                  \
        .acc = {                                                               \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_FC(f_name, f_type, f_func)                                    \
    {.category = func_fc,                                                      \
     .cat = {                                                                  \
        .fc = {                                                                \
            .name = #f_name,                                                   \
            .type = f_type,                                                    \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_LRN(f_name, f_func)                               		       \
    {.category = func_lrn,                                                     \
     .cat = {                                                                  \
        .lrn = {                                                               \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_CM(f_name, f_func)                                            \
    {.category = func_common,                                                  \
     .cat = {                                                                  \
        .cm = {                                                                \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#define LIB_FUNC_IM2COL(f_name, f_func)                                            \
    {.category = func_im2col,                                                  \
     .cat = {                                                                  \
        .cm = {                                                                \
            .name = #f_name,                                                   \
            .func = (FUNCPTR_T)NULL                                            \
        }                                                                      \
     }                                                                         \
    }

#endif

struct lib_function lib[] = {

    // Convolution 1x1
    LIB_FUNC_CONV(MV_conv1x1s1hhhh, hhhh, k1x1, stride1, mvcvConvolution1x1Fp16ToFp16_asm),

   // Convolution 3x3
    LIB_FUNC_CONV(MV_conv3x3s1hhhh, hhhh, k3x3, stride1, mvcvConvolution3x3Fp16ToFp16_asm),
    
    LIB_FUNC_CONV(MV_conv3x3s2hhhh, hhhh, k3x3, stride2, mvcvConvolution3x3s2hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv3x3s3hhhh, hhhh, k3x3, stride3, mvcvConvolution3x3s3hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv3x3s4hhhh, hhhh, k3x3, stride4, mvcvConvolution3x3s4hhhh_asm),

    LIB_FUNC_CONV(MV_conv3x3s8hhhh, hhhh, k3x3, stride8, mvcvConvolution3x3s8hhhh_asm),
    
    // Convolution 5x5

    LIB_FUNC_CONV(MV_conv5x5s1hhhh, hhhh, k5x5, stride1, mvcvConvolution5x5Fp16ToFp16_asm),
    
    LIB_FUNC_CONV(MV_conv5x5s2hhhh, hhhh, k5x5, stride2, mvcvConvolution5x5s2hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv5x5s3hhhh, hhhh, k5x5, stride3, mvcvConvolution5x5s3hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv5x5s4hhhh, hhhh, k5x5, stride4, mvcvConvolution5x5s4hhhh_asm),

    LIB_FUNC_CONV(MV_conv5x5s8hhhh, hhhh, k5x5, stride8, mvcvConvolution5x5s8hhhh_asm),

    // Convolution 7x7
    LIB_FUNC_CONV(MV_conv7x7s1hhhh, hhhh, k7x7, stride1, mvcvConvolution7x7Fp16ToFp16_asm),
    
    LIB_FUNC_CONV(MV_conv7x7s2hhhh, hhhh, k7x7, stride2, mvcvConvolution7x7s2hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv7x7s4hhhh, hhhh, k7x7, stride4, mvcvConvolution7x7s4hhhh_asm),
    
    LIB_FUNC_CONV(MV_conv7x7s8hhhh, hhhh, k7x7, stride8, mvcvConvolution7x7s8hhhh_asm),

    // Convolution 9x9
    LIB_FUNC_CONV(MV_conv9x9s1hhhh, hhhh, k9x9, stride1, mvcvConvolution9x9Fp16ToFp16_asm),

    LIB_FUNC_CONV(MV_conv9x9s2hhhh, hhhh, k9x9, stride2, mvcvConvolution9x9s2hhhh_asm),

    LIB_FUNC_CONV(MV_conv9x9s3hhhh, hhhh, k9x9, stride3, mvcvConvolution9x9s3hhhh_asm),

    LIB_FUNC_CONV(MV_conv9x9s4hhhh, hhhh, k9x9, stride4, mvcvConvolution9x9s4hhhh_asm),

    LIB_FUNC_CONV(MV_conv9x9s8hhhh, hhhh, k9x9, stride8, mvcvConvolution9x9s8hhhh_asm),

    // Convolution 11x11
    LIB_FUNC_CONV(MV_conv11x11s1hhhh, hhhh, k11x11, stride1, mvcvConvolution11x11s1hhhh_asm),

    LIB_FUNC_CONV(MV_conv11x11s2hhhh, hhhh, k11x11, stride2, mvcvConvolution11x11s2hhhh_asm),

    LIB_FUNC_CONV(MV_conv11x11s4hhhh, hhhh, k11x11, stride4, mvcvConvolution11x11s4hhhh_asm),

    LIB_FUNC_CONV(MV_conv11x11s8hhhh, hhhh, k11x11, stride8, mvcvConvolution11x11s8hhhh_asm),

    // Accumulation
    LIB_FUNC_ACC(MV_accumulate_hh_single32, hh, mvcvAccumulateFp16Single32_asm),

	LIB_FUNC_ACC(MV_accumulate_hh_single64, hh, mvcvAccumulateFp16Single64_asm),

	LIB_FUNC_ACC(MV_accumulate_hh, hh, mvcvAccumulateFp16_asm),

    LIB_FUNC_ACC(MV_accumulate_hh_withReLU, hh, mvcvaccumulateFp16withReLU_asm),

    // Pooling
    LIB_FUNC_POOL(MV_avgPool3x3s2hh, hh, k3x3, stride2, pooling_AVE, mvcvAvgPool3x3s2hh_asm),
    
    LIB_FUNC_POOL(MV_avgPool7x7s1hh, hh, k7x7, stride1, pooling_AVE, mvcvAvgPool7x7xk),
    // LIB_FUNC_POOL(MV_avgPool7x7s1hh, hh, k7x7, stride1, pooling_AVE, mvcvAvgPool7x7xk_asm),

    LIB_FUNC_POOL(MV_avgPool14x14s1hh, hh, k14x14, stride1, pooling_AVE, mvcvAvgPool14x14xk),
    
    LIB_FUNC_POOL(MV_maxPool2x2s2hh, hh, k2x2, stride2, pooling_MAX, mvcvMaxPool2x2s2hh_asm),

    LIB_FUNC_POOL(MV_maxPool3x3s1hh, hh, k3x3, stride1, pooling_MAX, mvcvMaxPool3x3s1hh_asm),

    LIB_FUNC_POOL(MV_maxPool3x3s2hh, hh, k3x3, stride2, pooling_MAX, mvcvMaxPool3x3s2hh_asm),    

    LIB_FUNC_POOL(MV_maxPool3x3s2hh_withReLU, hh, k3x3, stride2, pooling_MAX, mvcvMaxPool3x3s2hhwithReLU_asm),

    //Inner Product
    LIB_FUNC_FC(MV_vecvecmul_hhff, hhff, mvcvvecVecDotProduct32_asm),

    //LRN
    LIB_FUNC_LRN(FT_LRN_AcrossChannels_generic, LRN_AcrossChannels_generic),

    LIB_FUNC_LRN(FT_LRN_AcrossChannels_LR5_A0_0001_B_0_75, LRN_AcrossChannels_LR5_A0_0001_B_0_75),

    // Common Functions
    LIB_FUNC_CM(CM_printf, printf),

    LIB_FUNC_CM(CM_bzero, bzero),

    LIB_FUNC_CM(CM_conv_ddr, conv_ddr),

    LIB_FUNC_CM(CM_pool_ave_ddr, pool_ave_ddr),

    LIB_FUNC_CM(CM_pool_max_ddr, pool_max_ddr),

    LIB_FUNC_CM(CM_fc_ddr, fc_ddr),

    LIB_FUNC_CM(CM_lrn_ddr_generic, lrn_ddr_generic),

    LIB_FUNC_CM(CM_lrn_ddr_LR5_A0_0001_B_0_75, lrn_ddr_LR5_A0_0001_B_0_75),

    LIB_FUNC_CM(FT_im_col, im2col_ddr),
};

#endif