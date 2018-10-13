#include "ddr_fc.h"
#include "utils.h"

void fc_ddr(int firstLineNo, int lastLineNo, fc_context *context){
	
    int shaveId = context->com.shaveId;
    fc_info *info = context->info;
    J_FUNCPTR_T jumpTable = context->com.jumpTable;

    SETALIGNEDMEM_PTR setAlignedMem = context->mem.setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem = context->mem.getAlignedMem;

    DMAINITREQUESTER_PTR dmaInitRequester = context->dma.dmaInitRequester;
    DMACREATETRANSACTION_PTR dmaCreateTransaction = context->dma.dmaCreateTransaction;
    DMASTARTLISTTASK_PTR dmaStartListTask = context->dma.dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask = context->dma.dmaWaitTask;
    dmaTransactionList_t *task = context->dma.task;
    dmaTransactionList_t **ref = context->dma.ref;

    BZERO_PTR bzero   = (BZERO_PTR) jumpTable(CM_bzero);
    VECDOTPTR_T dot = (VECDOTPTR_T) jumpTable(MV_vecvecmul_hhff);

    int inputWidth = info->inputWidth;  
    int seg_width = 14*1024;  


    int line_diff = lastLineNo - firstLineNo;
    int line_diff64 = round_up64(line_diff);

    int inputBufferBytes = info->inputBPP * round_up64(seg_width);
    int outputBufferBytes = info->outputBPP * line_diff64;

    int true_seg_width;    
                          
    int true_seg_width8;  
    int true_input_bytes; 
    u8* inputAddr;

    setAlignedMem( (firstLineNo == 0 ? shaveId : -1), jumpTable);

    u8 *vector  = getAlignedMem(128, inputBufferBytes);

    u8 *localInput  = getAlignedMem(128, inputBufferBytes);
    u8 *localInput1 = getAlignedMem(128, inputBufferBytes);

    half *localOutput = getAlignedMem(128, outputBufferBytes);
    memset64_h(localOutput, line_diff64, 0.0f);

    half *vectorBuffer = (half *)vector;
    half *prefetchBuffer, *currentBuffer, *toggleInputBuffer;

    u32 id = dmaInitRequester(shaveId);

    int offset_increment = info->weightLines_offset * info->inputBPP;

    for (int width_start = 0; width_start < inputWidth; width_start += seg_width) {

        int offset = firstLineNo * info->weightLines_offset * info->inputBPP;

        prefetchBuffer = (half *)localInput;
        currentBuffer = (half *)localInput1;

        true_seg_width = MIN(seg_width, inputWidth-width_start);
        true_input_bytes = info->inputBPP * true_seg_width;


        ref[0] = dmaCreateTransaction(
            id,
            &task[0],
            &info->vector[width_start * info->inputBPP], 
            (u8 *)vectorBuffer,                           
            true_input_bytes);                  
        dmaStartListTask(ref[0]);

        true_seg_width8 = ((true_seg_width + 7) / 8) * 8;

        dmaWaitTask(ref[0]);


        inputAddr = info->weightLines + offset + width_start * info->inputBPP;
        ref[0] = dmaCreateTransaction(
            id,
            &task[0],
            inputAddr,              // src
            (u8*)prefetchBuffer,    // dst
            true_input_bytes);      // byte length
        dmaStartListTask(ref[0]);

        int residue = info->inputBPP * (true_seg_width8 - true_seg_width);

        if (residue)
            bzero(&(vectorBuffer[width_start * info->inputBPP]) + 
                                                     true_input_bytes, residue);
        dmaWaitTask(ref[0]);

        for (int lineNo = firstLineNo + 1; lineNo < lastLineNo; lineNo++) {
            offset += offset_increment;

            inputAddr = info->weightLines + offset + width_start * info->inputBPP;

            ref[0] = dmaCreateTransaction(
                id,
                &task[0],
                inputAddr,             
                (u8*)currentBuffer,  
                true_input_bytes); 
            dmaStartListTask(ref[0]);

            localOutput[lineNo - firstLineNo - 1] += 
                (half)dot(vectorBuffer, prefetchBuffer, true_seg_width8);

            toggleInputBuffer = prefetchBuffer;
            prefetchBuffer = currentBuffer;
            currentBuffer = toggleInputBuffer;

            dmaWaitTask(ref[0]);
        }

        localOutput[line_diff - 1] += 
            (half)dot(vectorBuffer, prefetchBuffer, true_seg_width8);
    }

    if (info->bias != NULL) {
        half *bias = (half *)(info->bias);
        for (int i = 0; i < lastLineNo - firstLineNo; i++) {
            localOutput[i] += bias[i + firstLineNo];
        }
    }

    if (info->with_relu) {
        relu_inplace64_h(localOutput, line_diff64);
    }

    ref[0] = dmaCreateTransaction(
        id,
        &task[0],
        (u8 *)localOutput,                       
        info->output + firstLineNo * info->outputBPP,  
        (lastLineNo-firstLineNo) * info->outputBPP); 
    dmaStartListTask(ref[0]);
    dmaWaitTask(ref[0]);
}