// 1: Includes
// ----------------------------------------------------------------------------
#include <svuCommonShave.h>
#include <swcCdma.h>
#include <swcWhoAmI.h>
#include <stddef.h>
#include <mv_types.h>
#include <stdio.h>

#include <moviVectorUtils.h>

#include "../ddr/ddr_conv.h"
#include "../ddr/ddr_pool.h"
#include "../ddr/ddr_fc.h"
#include "../ddr/local_resp_norm.h"

#include <ddr_functions.h>
#include "memory.h"

#define DMA_TASKS_MAX 3
#define DMA_REFS_MAX 3

dmaTransactionList_t __attribute__((section(".cmx.cdmaDescriptors"))) 
task[DMA_TASKS_MAX], *ref[DMA_REFS_MAX];

void shave_conv(conv_info **info, u32 firstMapNo, u32 lastMapNo, J_FUNCPTR_T jumpTable){

    int shaveId = swcWhoAmI() - PROCESS_SHAVE0;
    conv_context context = {
        .dma = (dma_context){
            .dmaInitRequester = dmaInitRequester,
            .dmaCreateTransactionFullOptions = dmaCreateTransactionFullOptions,
            .dmaStartListTask = dmaStartListTask,
            .dmaWaitTask = dmaWaitTask,
            .task = task,
            .ref = ref
        },

        .com = (common_context){
            .shaveId = shaveId,
            .jumpTable = jumpTable
        },

        .mem = (memory_context){
            .setAlignedMem = setAlignedMem,
            .getAlignedMem = getAlignedMem
        },

        .info = info,
    };

    CONV_DDR_PTR conv_ddr = (CONV_DDR_PTR) jumpTable(CM_conv_ddr);
    conv_ddr(firstMapNo, lastMapNo, &context);

    SHAVE_HALT;
}

void shave_pool(pool_info **info, u32 firstMapNo, u32 lastMapNo, J_FUNCPTR_T jumpTable){
    int shaveId = swcWhoAmI() - PROCESS_SHAVE0;
    pool_context context = {
        .dma = (dma_context){
            .dmaInitRequester = dmaInitRequester,
            .dmaCreateTransactionFullOptions = dmaCreateTransactionFullOptions,
            .dmaStartListTask = dmaStartListTask,
            .dmaWaitTask = dmaWaitTask,
            .task = task,
            .ref = ref
        },

        .com = (common_context){
            .shaveId = shaveId,
            .jumpTable = jumpTable
        },

        .mem = (memory_context){
            .setAlignedMem = setAlignedMem,
            .getAlignedMem = getAlignedMem
        },

        .info = info,
    };

    POOL_DDR_PTR pool_max_ddr = (POOL_DDR_PTR) jumpTable(CM_pool_max_ddr);
    POOL_DDR_PTR pool_ave_ddr = (POOL_DDR_PTR) jumpTable(CM_pool_ave_ddr);

    switch (info[0]->type){

        case pooling_AVE:
            pool_ave_ddr(firstMapNo, lastMapNo, &context);
            break;
        case pooling_MAX:
            pool_max_ddr(firstMapNo, lastMapNo, &context);
            break;
        case pooling_STOCHASTIC:
            if (shaveId == 0){
                PRINTF_PTR printf = (PRINTF_PTR) jumpTable(CM_printf);
                printf("\nStochastic pooling not supported!\n");
            }
            break;
        case pooling_INVALID:
            if (shaveId == 0){
                PRINTF_PTR printf = (PRINTF_PTR) jumpTable(CM_printf);
                printf("\nInvalid pooling type!\n");
            }
            break;
    }
    SHAVE_HALT;
}

void shave_fc(fc_info *info, int firstLineNo, int lastLineNo, J_FUNCPTR_T jumpTable){

    int shaveId = swcWhoAmI() - PROCESS_SHAVE0;
    fc_context context = {
        .dma = (dma_context){
            .dmaInitRequester = dmaInitRequester,
            .dmaCreateTransaction = dmaCreateTransaction,
            .dmaStartListTask = dmaStartListTask,
            .dmaWaitTask = dmaWaitTask,
            .task = task,
            .ref = ref
        },

        .com = (common_context){
            .shaveId = shaveId,
            .jumpTable = jumpTable
        },

        .mem = (memory_context){
            .setAlignedMem = setAlignedMem,
            .getAlignedMem = getAlignedMem
        },

        .info = info,
    };

    FC_DDR_PTR fc_ddr = (FC_DDR_PTR) jumpTable(CM_fc_ddr);
    fc_ddr(firstLineNo, lastLineNo, &context);
   
    SHAVE_HALT;
}
//TODO #if LRN
void shave_lrn(lrn_info *info, u32 first_pixel, u32 last_pixel, J_FUNCPTR_T jumpTable){

    int shaveId = swcWhoAmI() - PROCESS_SHAVE0;
    lrn_context context = {
        .dma = (dma_context){
            .dmaInitRequester = dmaInitRequester,
            .dmaCreateTransactionFullOptions = dmaCreateTransactionFullOptions,
            .dmaStartListTask = dmaStartListTask,
            .dmaWaitTask = dmaWaitTask,
            .task = task,
            .ref = ref
        },

        .com = (common_context){
            .shaveId = shaveId,
            .jumpTable = jumpTable
        },

        .mem = (memory_context){
            .setAlignedMem = setAlignedMem,
            .getAlignedMem = getAlignedMem
        },

        .info = info,
    };

    switch(info->ddr_function){

        case 35:{
            LRN_DDR_PTR lrn_ddr = (LRN_DDR_PTR) jumpTable(CM_lrn_ddr_generic);
            lrn_ddr(first_pixel, last_pixel, &context);
            break;
        }
        case 36:{

            LRN_DDR_PTR lrn_ddr = (LRN_DDR_PTR) jumpTable(CM_lrn_ddr_LR5_A0_0001_B_0_75);
            lrn_ddr(first_pixel, last_pixel, &context);
            break;
        }
        default:{
            if (shaveId == 0){
                PRINTF_PTR printf = (PRINTF_PTR) jumpTable(CM_printf);
                printf("\nInvalid normalization type!\n");
            }
            break;
        } 
    }
    SHAVE_HALT;
}

void shave_im2col(im2col_info **info, int firstMapNo, int lastMapNo, J_FUNCPTR_T jumpTable){
    int shaveId = swcWhoAmI() - PROCESS_SHAVE0;
    im2col_context context = {
        .dma = (dma_context){
            .dmaInitRequester = dmaInitRequester,
            .dmaCreateTransactionFullOptions = dmaCreateTransactionFullOptions,
            .dmaStartListTask = dmaStartListTask,
            .dmaWaitTask = dmaWaitTask,
            .task = task,
            .ref = ref
        },

        .com = (common_context){
            .shaveId = shaveId,
            .jumpTable = jumpTable
        },

        .mem = (memory_context){
            .setAlignedMem = setAlignedMem,
            .getAlignedMem = getAlignedMem
        },

        .info = info,
    };

    IM2COL_DDR_PTR im2col_ddr = (IM2COL_DDR_PTR) jumpTable(FT_im_col);
    im2col_ddr(firstMapNo, lastMapNo, &context);

    SHAVE_HALT;
}