#include "ddr_pool.h"
#include "utils.h"

void mvcvAvgPool7x7xk(half** srcAddr, half** destAddr, u32 width)
{
    (void)width;
    u32 i, j;
    float sum;
    const float f49 = (float)0.02040816326530612;

    for( j = 0; j < 7; j++)
    {
        sum = 0;
        for( i = 0; i < 49; i++)
            sum += srcAddr[j][i];
        destAddr[0][j] = (half) (sum * f49);
    }
}

void mvcvAvgPool14x14xk(half** srcAddr, half** destAddr, u32 width)
{
    (void)width;
    u32 i, j;
    float sum;
    const float f = (float)0.0051020408163265;

    for( j = 0; j < 14; j++)
    {
        sum = 0;
        for( i = 0; i < 14*14; i++)
            sum += srcAddr[j][i];
        destAddr[0][j] = (half) (sum * f);
    }
}

void 
pooling_ave_edges64_h(
    half *array,    

    int true_width, 
    int width,
    int height,

    int elements, 
    
    int center, 
    int right, 
    int bottom, 
    int corner
)
{
    half center_over_r = center / (half)right;
    half center_over_b = center / (half)bottom;
    half center_over_c = center / (half)corner;

    int index = true_width - 1;
    for (int i = 0; i < height-1; i++) {
        array[index] *= center_over_r;
        index += width;
    }

    index = width * (height-1);
    for (int j = 0; j < true_width-1; j++) {
        array[index] *= center_over_b;
        index++;
    }

    array[index] *= center_over_c;

    half d = 1.0/center;

    half8 denom = {d, d, d, d, d, d, d, d};
    half8 *output = (half8 *)array;

    int j;
    for (j = 0; j < elements/8; j += 8) {
        output[j]   = __builtin_shave_vau_mul_f16_rr(output[j],   denom);
        output[j+1] = __builtin_shave_vau_mul_f16_rr(output[j+1], denom);
        output[j+2] = __builtin_shave_vau_mul_f16_rr(output[j+2], denom);
        output[j+3] = __builtin_shave_vau_mul_f16_rr(output[j+3], denom);
        output[j+4] = __builtin_shave_vau_mul_f16_rr(output[j+4], denom);
        output[j+5] = __builtin_shave_vau_mul_f16_rr(output[j+5], denom);
        output[j+6] = __builtin_shave_vau_mul_f16_rr(output[j+6], denom);
        output[j+7] = __builtin_shave_vau_mul_f16_rr(output[j+7], denom);
    }
}

void 
pool_ave_ddr(
   u32 firstMapNo, 
   u32 lastMapNo, 
   pool_context *context
)
{
    int shaveId = context->com.shaveId;
    pool_info **info = context->info;
    J_FUNCPTR_T jumpTable = context->com.jumpTable;

    SETALIGNEDMEM_PTR setAlignedMem = context->mem.setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem = context->mem.getAlignedMem;

    DMAINITREQUESTER_PTR dmaInitRequester = context->dma.dmaInitRequester;
    DMACREATETRANSACTIONFULLOPTIONS_PTR dmaCreateTransactionFullOptions = 
                                context->dma.dmaCreateTransactionFullOptions;
    DMASTARTLISTTASK_PTR dmaStartListTask = context->dma.dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask = context->dma.dmaWaitTask;
    dmaTransactionList_t *task = context->dma.task;
    dmaTransactionList_t **ref = context->dma.ref;


    POOLPTR_T pool = (POOLPTR_T) jumpTable(info[0]->ddr_function);

    int kernel_h = info[0]->kernel_h;

    int out_buf_2d_elements = round_up64(info[0]->out_buffer_elements);

    for(int p=0;p<info[0]->splits;p++){

        setAlignedMem( (firstMapNo == 0 ? shaveId : -1), jumpTable);

        u32 outputBufferBytes = info[p]->outputBPP * out_buf_2d_elements;

        u8 **inputPtr = getAlignedMem(32, info[p]->kernel_h * sizeof(*inputPtr));

        u8 **group = getAlignedMem(32, info[p]->in_buffers_num * sizeof(*group));
        for (u8 i = 0; i < info[p]->in_buffers_num; i++) {

            int len = round_up64(info[p]->in_buffers[i].buffer_elements);
            group[i] = getAlignedMem(32, info[p]->inputBPP * len);
        
            memset64_h((half *)(group[i]), len, 0.0);
        }

        u8 *localOutput = getAlignedMem(32, outputBufferBytes);

        u32 id = dmaInitRequester(shaveId);

        for (u16 j = 0; j < info[p]->in_buffers_num; j++) {
            for (u16 i = j, k = 0; i < kernel_h; i += info[p]->in_buffers_num, k++) {
                inputPtr[i] = (u8*)group[j] + k * info[p]->line_width * info[p]->inputBPP;
            }
        }

        u32 mapNo = firstMapNo;
        int offset_increment_input = info[p]->inputBPP * info[p]->input_channel_offset;
        int offset_increment_output = info[p]->outputBPP * info[p]->output_channel_offset;

        u8* inputAddr = info[p]->input + mapNo * offset_increment_input;
        u8* outputAddr = info[p]->output + mapNo * offset_increment_output;

        while (mapNo < lastMapNo) {  
            for (u8 i = 0; i < info[p]->in_buffers_num; i++) {

                if (info[p]->in_buffers[i].elements == 0)
                    continue;

                ref[0] = dmaCreateTransactionFullOptions(
                    id,
                    &task[0],
                    inputAddr + info[p]->in_buffers[i].src_addr * info[p]->inputBPP,      // src
                    group[i] + info[p]->in_buffers[i].dst_addr * info[p]->inputBPP,       // dst
                    info[p]->in_buffers[i].elements * info[p]->inputBPP,                  // byte length
                    info[p]->in_src_width * info[p]->inputBPP,                            // src line width
                    info[p]->in_dst_width * info[p]->inputBPP,                            // dst line width
                    info[p]->in_src_stride * info[p]->inputBPP,                           // src stride
                    info[p]->in_dst_stride * info[p]->inputBPP);                          // dst stride
                dmaStartListTask(ref[0]);
                dmaWaitTask(ref[0]);
            }

            pool((void**)inputPtr, (void**)&localOutput, info[p]->out_buffer_elements);

            ref[1] = dmaCreateTransactionFullOptions(
                id,
                &task[1],
                localOutput + info[p]->out_src_addr * info[p]->outputBPP,                 // src
                outputAddr + info[p]->out_dst_addr * info[p]->outputBPP,                  // dst
                info[p]->out_elements * info[p]->outputBPP,                               // byte length
                info[p]->out_src_width * info[p]->outputBPP,                              // src line width
                info[p]->out_dst_width * info[p]->outputBPP,                              // dst line width
                info[p]->out_src_stride * info[p]->outputBPP,                             // src stride
                info[p]->out_dst_stride * info[p]->outputBPP);                            // dst stride

            dmaStartListTask(ref[1]);
            dmaWaitTask(ref[1]);

            mapNo++;
            inputAddr += offset_increment_input;
            outputAddr += offset_increment_output;
        }
    }    
}

void 
pool_max_ddr(
   u32 firstMapNo, 
   u32 lastMapNo, 
   pool_context *context
)
{
    int shaveId = context->com.shaveId;
    pool_info **info = context->info;
    J_FUNCPTR_T jumpTable = context->com.jumpTable;

    SETALIGNEDMEM_PTR setAlignedMem = context->mem.setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem = context->mem.getAlignedMem;

    DMAINITREQUESTER_PTR dmaInitRequester = context->dma.dmaInitRequester;
    DMACREATETRANSACTIONFULLOPTIONS_PTR dmaCreateTransactionFullOptions = 
                                context->dma.dmaCreateTransactionFullOptions;
    DMASTARTLISTTASK_PTR dmaStartListTask = context->dma.dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask = context->dma.dmaWaitTask;
    dmaTransactionList_t *task = context->dma.task;
    dmaTransactionList_t **ref = context->dma.ref;


    POOLPTR_T pool = (POOLPTR_T) jumpTable(info[0]->ddr_function);

    int kernel_h = info[0]->kernel_h;

    // -Inf in fp16 format
    int m_inf_bin = 0x0000FC00;
    half m_inf = *((half *) &m_inf_bin);

    for(int p=0;p<info[0]->splits;p++){

        setAlignedMem( (firstMapNo == 0 ? shaveId : -1), jumpTable);

        u32 outputBufferBytes = info[p]->outputBPP * 
                                        round_up64(info[p]->out_buffer_elements);

        u8 **inputPtr = getAlignedMem(32, info[p]->kernel_h * sizeof(*inputPtr));

        u8 **group = getAlignedMem(32, info[p]->in_buffers_num * sizeof(*group));
        for (u8 i = 0; i < info[p]->in_buffers_num; i++) {

            int len = round_up64(info[p]->in_buffers[i].buffer_elements);
            group[i] = getAlignedMem(32, info[p]->inputBPP * len);
        
            memset64_h((half *)(group[i]), len, m_inf);
        }

        u8 *localOutput = getAlignedMem(32, outputBufferBytes);

        u32 id = dmaInitRequester(shaveId);

        for (u16 j = 0; j < info[p]->in_buffers_num; j++) {
            for (u16 i = j, k = 0; i < kernel_h; i += info[p]->in_buffers_num, k++) {
                inputPtr[i] = (u8*)group[j] + k * info[p]->line_width * info[p]->inputBPP;
            }
        }

        u32 mapNo = firstMapNo;
        int offset_increment_input = info[p]->inputBPP * info[p]->input_channel_offset;
        int offset_increment_output = info[p]->outputBPP * info[p]->output_channel_offset;

        u8* inputAddr = info[p]->input + mapNo * offset_increment_input;
        u8* outputAddr = info[p]->output + mapNo * offset_increment_output;

        while (mapNo < lastMapNo) {  
            for (u8 i = 0; i < info[p]->in_buffers_num; i++) {

                if (info[p]->in_buffers[i].elements == 0)
                    continue;

                ref[0] = dmaCreateTransactionFullOptions(
                    id,
                    &task[0],
                    inputAddr + info[p]->in_buffers[i].src_addr * info[p]->inputBPP,    
                    group[i] + info[p]->in_buffers[i].dst_addr * info[p]->inputBPP,  
                    info[p]->in_buffers[i].elements * info[p]->inputBPP,            
                    info[p]->in_src_width * info[p]->inputBPP,                         
                    info[p]->in_dst_width * info[p]->inputBPP,                            
                    info[p]->in_src_stride * info[p]->inputBPP,                     
                    info[p]->in_dst_stride * info[p]->inputBPP);                   
                dmaStartListTask(ref[0]);
                dmaWaitTask(ref[0]);
            }

            pool((void**)inputPtr, (void**)&localOutput, info[p]->out_buffer_elements);
            
            ref[1] = dmaCreateTransactionFullOptions(
                id,
                &task[1],
                localOutput + info[p]->out_src_addr * info[p]->outputBPP,           
                outputAddr + info[p]->out_dst_addr * info[p]->outputBPP,              
                info[p]->out_elements * info[p]->outputBPP,                        
                info[p]->out_src_width * info[p]->outputBPP,                        
                info[p]->out_dst_width * info[p]->outputBPP,                          
                info[p]->out_src_stride * info[p]->outputBPP,                  
                info[p]->out_dst_stride * info[p]->outputBPP);                   

            dmaStartListTask(ref[1]);
            dmaWaitTask(ref[1]);

            mapNo++;
            inputAddr += offset_increment_input;
            outputAddr += offset_increment_output;
        }
    }       
}