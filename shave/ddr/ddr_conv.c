#include "ddr_conv.h"
#include "utils.h"
#include <stdio.h>

half dot_product(half* in, half* weight, u32 length){
    half array = 0;
    
    for (u32 i=0;i<length;i++){
        array += in[i]*weight[i];
        
    }

    return array;
}

void 
conv_ddr( 
   u32 firstMapNo, 
   u32 lastMapNo, 
   conv_context *context
)
{
    int shaveId = context->com.shaveId;

    
    conv_info **info = context->info;
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

    CONVPTR_T convolve = (CONVPTR_T) jumpTable(info[0]->ddr_function);
    
    ACCPTR_T accumulateFp16 = (ACCPTR_T) jumpTable(MV_accumulate_hh);
    ACCPTR_T accumulateFp16withReLU = 
                            (ACCPTR_T) jumpTable(MV_accumulate_hh_withReLU);
    ACCPTR_T accumulate = NULL;
    ACCSINGLEPTR_T accumulateFp16Single = 
                        (ACCSINGLEPTR_T) jumpTable(MV_accumulate_hh_single64);

    int input_src_width = info[0]->in_src_width * info[0]->inputBPP;
    int input_src_stride = info[0]->in_src_stride * info[0]->inputBPP;
    int input_dst_width = info[0]->in_dst_width * info[0]->inputBPP;
    int input_dst_stride = info[0]->in_dst_stride * info[0]->inputBPP;

    int output_src_width = info[0]->out_src_width * info[0]->outputBPP;
    int output_src_stride = info[0]->out_src_stride * info[0]->outputBPP;

    int output_dst_width = info[0]->out_dst_width * info[0]->outputBPP;
    int output_dst_stride = info[0]->out_dst_stride * info[0]->outputBPP;

    u32 actual_coalescing;
    int input_offset = info[0]->input_channel_offset * info[0]->inputBPP;

    int times = 1;
    u32 previousMap, firstChannel = 0;
                    
    if (info[0]->in_stride!=1){
        for (u8 p=0;p<info[0]->splits;p++){
            
            int output_src_addr = info[p]->out_src_addr * info[p]->outputBPP;
            
            int output_dst_addr = info[p]->out_dst_addr * info[p]->outputBPP;

            int output_bytes = info[p]->out_elements * info[p]->outputBPP;
            
            int conv_width =  info[0]->in_stride * info[p]->out_buffer_elements;
            u32 out_buf_sz = round_up64(info[p]->out_buffer_elements * info[p]->in_buffers_num);

            setAlignedMem( (shaveId == 0 ? shaveId : -1), jumpTable);

            u8 **group = getAlignedMem(128, info[p]->in_buffers_num * sizeof(*group));
            for (int i = 0; i < info[p]->in_buffers_num; i++){


                int len = round_up64(info[p]->in_buffers[i].buffer_elements);
                group[i] = getAlignedMem(128, info[p]->inputBPP * len);
                memset64_h((half *)(group[i]), len, 0.0);
            }

            u8 **inputPtr = getAlignedMem(128, info[p]->kernel_h * sizeof(*inputPtr));
            
            u8 **localOutput = getAlignedMem(128, info[p]->coalescing_num * 
                                                    sizeof(*localOutput));
            
            for (u32 i = 0; i < info[p]->coalescing_num; i++) {
                localOutput[i] = getAlignedMem(128, out_buf_sz * info[p]->outputBPP);
            }

            u8 *localOutputTmp = getAlignedMem(128, out_buf_sz * info[p]->outputBPP);

            u32 id = dmaInitRequester(shaveId);
            int start = (info[p]->in_buffer_shift + (info[p]->kernel_w/2)) * info[p]->inputBPP;

            for (u16 j = 0; j < info[p]->in_stride; j++ ){
                for (u16 i = j, k = 0; i < info[p]->kernel_h; i+= info[p]->in_stride, k++) {

                    inputPtr[i] = &group[j][start +  (k * info[p]->line_width) * info[p]->inputBPP];
                }
            }
            
            if(info[p]->c_group!=1){
                if((firstMapNo < (info[p]->maps/info[p]->c_group))&&(lastMapNo > (info[p]->maps/info[p]->c_group))){
                    previousMap = lastMapNo;
                    lastMapNo = info[p]->maps/info[p]->c_group;
                    times = 2;
                }
                else if((firstMapNo >= (info[p]->maps/info[p]->c_group))&&(lastMapNo > (info[p]->maps/info[p]->c_group))){
                    firstChannel = info[p]->channels;
                }
                else{
                    firstChannel = 0;
                }
            }

            u32 mapNo = firstMapNo;    
            
            for (int k = 0; k < times; k++){
                if (k == 1){
                    firstMapNo = info[p]->maps/info[p]->c_group;
                    lastMapNo = previousMap;
                    firstChannel = info[p]->channels;
                }
                while (mapNo < lastMapNo) {
                    accumulate = accumulateFp16;
                    actual_coalescing = MIN(info[p]->coalescing_num, lastMapNo - mapNo);

                    u8* inputAddr = info[p]->input ;
                    for (int j = 0; j < info[p]->in_buffers_num ; j++){
                        if (info[p]->in_buffers[j].elements == 0)
                            continue;
                                        
                        ref[0] = dmaCreateTransactionFullOptions(
                            id,
                            &task[0],
                            inputAddr + info[p]->in_buffers[j].src_addr * info[p]->inputBPP, 
                            group[j] + info[p]->in_buffers[j].dst_addr * info[p]->inputBPP,  
                            info[p]->in_buffers[j].elements * info[p]->inputBPP,            
                            input_src_width,                
                            input_dst_width,                 
                            input_src_stride,            
                            input_dst_stride);           
                        dmaStartListTask(ref[0]);
                        dmaWaitTask(ref[0]);
                        
                    }
                    
                    for (u32 channelIdx = firstChannel; channelIdx < firstChannel + info[p]->channels-1; channelIdx++) {
                        for (u32 i = 0; i < actual_coalescing; i++) {
                            
                            u32 taps_offset = info[p]->kernelBPP * (
                                info[p]->conv_weights_offset*(mapNo+i) + 
                                (channelIdx - firstChannel) * info[p]->conv_weights_channel_offset);

                            if (channelIdx == firstChannel) {
                                convolve((void**)inputPtr, (void**)&localOutput[i], 
                                        info[p]->conv_weights + taps_offset, conv_width);
                             }
                            else {
                                convolve((void**)inputPtr, (void**)&localOutputTmp, 
                                        info[p]->conv_weights + taps_offset, conv_width);
                                accumulate((half**)&localOutput[i], 
                                                (half**)&localOutputTmp, conv_width);
                            } 
                        }
                        inputAddr += input_offset;

                        for (int j = 0; j<info[p]->in_buffers_num; j++){
                            if (info[p]->in_buffers[j].elements == 0)
                                continue;

                            ref[0] = dmaCreateTransactionFullOptions(
                                id,
                                &task[0],
                                inputAddr + info[p]->in_buffers[j].src_addr * info[p]->inputBPP,  
                                group[j] + info[p]->in_buffers[j].dst_addr * info[p]->inputBPP, 
                                info[p]->in_buffers[j].elements * info[p]->inputBPP,          
                                input_src_width,                  
                                input_dst_width,                 
                                input_src_stride,                
                                input_dst_stride);            
                            dmaStartListTask(ref[0]);
                            dmaWaitTask(ref[0]);
                        }       

                    } 

                    if (info[p]->with_relu) {
                            accumulate = accumulateFp16withReLU;
                        }

                    u32 taps_offset = info[p]->kernelBPP * (
                            info[p]->conv_weights_offset*(mapNo+1) -
                                info[p]->conv_weights_channel_offset);

                    convolve((void **) inputPtr, (void **) &localOutputTmp, 
                                    info[p]->conv_weights + taps_offset, conv_width);

                    if (info[p]->conv_biases != NULL) {
                        half b = ((half *) (info[p]->conv_biases))[mapNo];
                        accumulateFp16Single((half *) localOutputTmp, b, conv_width); 
                    }

                    if (info[p]->channels == 1) {
                       memset64_h((half *) (localOutput[0]), out_buf_sz, 0.0);
                    }
                    accumulate((half **) &localOutput[0], 
                               (half **) &localOutputTmp, conv_width);

                    
                    int output_offset_step = info[p]->outputBPP * info[p]->output_channel_offset;
                    int output_offset = mapNo * output_offset_step;

                    for (u32 i = 1; i < actual_coalescing; i++) {
                        ref[1] = dmaCreateTransactionFullOptions(
                            id,
                            &task[1],
                            localOutput[i-1] + output_src_addr,                
                            info[p]->output + output_offset + output_dst_addr,   
                            output_bytes,                      
                            output_src_width,                     
                            output_dst_width,                     
                            output_src_stride,               
                            output_dst_stride);                
                        dmaStartListTask(ref[1]);
                        dmaWaitTask(ref[1]);

                        u32 taps_offset = info[p]->kernelBPP * (
                                info[p]->conv_weights_offset*(mapNo+i+1) -
                                    info[p]->conv_weights_channel_offset);

                        convolve((void**)inputPtr, (void**)&localOutputTmp, 
                                    info[p]->conv_weights + taps_offset, conv_width);
                        if (info[p]->conv_biases != NULL) {
                            half b = ((half*)(info[p]->conv_biases))[mapNo + i];
                            accumulateFp16Single((half*)localOutputTmp, b, conv_width);
                        }

                        if (info[p]->channels == 1) {
                            memset64_h((half *) (localOutput[i]), out_buf_sz, 0.0);
                        }
                        accumulate((half **) &localOutput[i], 
                                   (half **) &localOutputTmp, conv_width);

                        output_offset += output_offset_step;

                    }

                    ref[1] = dmaCreateTransactionFullOptions(
                        id,
                        &task[1],
                        localOutput[actual_coalescing-1] + output_src_addr,      
                        info[p]->output + output_offset + output_dst_addr,        
                        output_bytes,                
                        output_src_width,              
                        output_dst_width,                  
                        output_src_stride,                 
                        output_dst_stride);           

                    dmaStartListTask(ref[1]);
                    dmaWaitTask(ref[1]);

               
                    mapNo += actual_coalescing;
                }
            }    
        }    
    }
    else{

        for (u8 p = 0;p < info[0]->splits; p++ ){
             // INPUT DMA
            int input_src_addr = info[p]->in_buffers[0].src_addr * info[p]->inputBPP;


            int input_dst_addr = info[p]->in_buffers[0].dst_addr * info[p]->inputBPP;


            int input_bytes = info[p]->in_buffers[0].elements * info[p]->inputBPP;

            int output_src_addr = info[p]->out_src_addr * info[p]->outputBPP;

            int output_dst_addr = info[p]->out_dst_addr * info[p]->outputBPP;
            u32 out_buf_sz = round_up64(info[p]->out_buffer_elements * info[p]->in_buffers_num);
            
            int output_bytes = info[p]->out_elements * info[p]->outputBPP;

            int conv_width = info[p]->out_buffer_elements;

            setAlignedMem( (shaveId == 0 ? shaveId : -1), jumpTable);
    
            u32 in_buf_sz  = round_up64(info[p]->in_buffers[0].buffer_elements);

            u8 **inputPtr, **toggleInputPtr;
            u8 **inputPtr1 = getAlignedMem(128, info[p]->kernel_h * sizeof(*inputPtr));
            u8 **inputPtr2 = getAlignedMem(128, info[p]->kernel_h * sizeof(*inputPtr));
            u8 *localInput  = getAlignedMem(128, in_buf_sz * info[p]->inputBPP);
            u8 *localInput1 = getAlignedMem(128, in_buf_sz * info[p]->inputBPP);
            
            u8 **localOutput = getAlignedMem(128, info[p]->coalescing_num * 
                                                    sizeof(*localOutput));
            for (u32 i = 0; i < info[p]->coalescing_num; i++) {
                localOutput[i] = getAlignedMem(128, out_buf_sz * info[p]->outputBPP);
            }

            u8 *localOutputTmp = getAlignedMem(128, out_buf_sz * info[p]->outputBPP);

            memset64_h((half *)localInput, in_buf_sz, 0.0);
            memset64_h((half *)localInput1, in_buf_sz, 0.0);

            u8 *prefetchBuffer;
            u8 *toggleInputBuffer;

            u32 id = dmaInitRequester(shaveId);

            for (u8 i = 0; i < info[p]->kernel_h; i++) {

                int start = (info[p]->in_buffer_shift + info[p]->kernel_w/2) * info[p]->inputBPP;
                int offset = start + i * info[p]->line_width * info[p]->inputBPP;

                inputPtr1[i] = &localInput[offset];
                inputPtr2[i] = &localInput1[offset];
            }

            toggleInputBuffer = (void *) ((u32)localInput1 ^ (u32)localInput);
            toggleInputPtr = (void *) ((u32)inputPtr1 ^ (u32)inputPtr2);

            u32 actual_coalescing;

            int times = 1;
            int input_offset = info[p]->input_channel_offset * info[p]->inputBPP;
            u32 previousMap, firstChannel = 0;
            
            if(info[p]->c_group!=1){
                if((firstMapNo < (info[p]->maps/info[p]->c_group))&&(lastMapNo > (info[p]->maps/info[p]->c_group))){
                    previousMap = lastMapNo;
                    lastMapNo = info[p]->maps/info[p]->c_group;
                    times = 2;
                }
                else if((firstMapNo >= (info[p]->maps/info[p]->c_group))&&(lastMapNo > (info[p]->maps/info[p]->c_group))){
                    firstChannel = info[p]->channels;
                }
                else{
                    firstChannel = 0;
                }
            }
            u32 mapNo = firstMapNo;
            
            for (int k = 0; k < times; k++){
                if (k == 1){
                    firstMapNo = info[p]->maps/info[p]->c_group;
                    lastMapNo = previousMap;
                    firstChannel = info[p]->channels;
                }
                mapNo = firstMapNo;
                while (mapNo < lastMapNo) { 
                    prefetchBuffer = localInput;
                    inputPtr = inputPtr2;

                    accumulate = accumulateFp16;
                    actual_coalescing = MIN(info[p]->coalescing_num, lastMapNo - mapNo);

                    u8* inputAddr = info[p]->input + (firstChannel*(input_offset));

                    ref[0] = dmaCreateTransactionFullOptions(
                        id,
                        &task[0],
                        inputAddr + input_src_addr,       
                        prefetchBuffer + input_dst_addr,   
                        input_bytes,                      
                        input_src_width,                  
                        input_dst_width,        
                        input_src_stride,               
                        input_dst_stride);                
                    dmaStartListTask(ref[0]);

                    for (u32 channelIdx = firstChannel; channelIdx < firstChannel + info[p]->channels-1; channelIdx++) {
                        dmaWaitTask(ref[0]);

                        prefetchBuffer = (void *) ((u32)prefetchBuffer ^ (u32)toggleInputBuffer);

                        inputAddr += input_offset;

                        ref[0] = dmaCreateTransactionFullOptions(
                            id,
                            &task[0],
                            inputAddr + input_src_addr,  
                            prefetchBuffer + input_dst_addr,  
                            input_bytes,                
                            input_src_width,              
                            input_dst_width,             
                            input_src_stride,              
                            input_dst_stride);      

                        dmaStartListTask(ref[0]);


                        inputPtr = (void *) ((u32)inputPtr ^ (u32)toggleInputPtr);

                        for (u32 i = 0; i < actual_coalescing; i++) {
                            
                            u32 taps_offset = info[p]->kernelBPP * (
                                info[p]->conv_weights_offset*(mapNo+i) + 
                                (channelIdx-firstChannel) * info[p]->conv_weights_channel_offset);

                            if (channelIdx == firstChannel) {
                                convolve((void**)inputPtr, (void**)&localOutput[i], 
                                        info[p]->conv_weights + taps_offset, conv_width);
                            }
                            else {
                                convolve((void**)inputPtr, (void**)&localOutputTmp, 
                                        info[p]->conv_weights + taps_offset, conv_width);
                                accumulate((half**)&localOutput[i], 
                                                (half**)&localOutputTmp, conv_width);
                            }
                        }
                    } 

                    dmaWaitTask(ref[0]); 

                    if (info[p]->with_relu) {
                        accumulate = accumulateFp16withReLU;
                    }

                    inputPtr = (void *) ((u32)inputPtr ^ (u32)toggleInputPtr);

                    u32 taps_offset = info[p]->kernelBPP * (
                            info[p]->conv_weights_offset*(mapNo+1) -
                                info[p]->conv_weights_channel_offset);

                    convolve((void **) inputPtr, (void **) &localOutputTmp, 
                                    info[p]->conv_weights + taps_offset, conv_width);

                    if (info[p]->conv_biases != NULL) {
                        half b = ((half *) (info[p]->conv_biases))[mapNo];
                        accumulateFp16Single((half *) localOutputTmp, b, conv_width); 
                    }

                    if (info[p]->channels == 1) {
                       memset64_h((half *) (localOutput[0]), out_buf_sz, 0.0);
                    }
                    accumulate((half **) &localOutput[0], 
                               (half **) &localOutputTmp, conv_width);

                    int output_offset_step = info[p]->outputBPP * info[p]->output_channel_offset;
                    int output_offset = mapNo * output_offset_step;
                    for (u32 i = 1; i < actual_coalescing; i++) {
                        ref[1] = dmaCreateTransactionFullOptions(
                            id,
                            &task[1],
                            localOutput[i-1] + output_src_addr,                
                            info[p]->output + output_offset + output_dst_addr,  
                            output_bytes,                      
                            output_src_width,               
                            output_dst_width,           
                            output_src_stride,    
                            output_dst_stride);        
                        dmaStartListTask(ref[1]);

                        u32 taps_offset = info[p]->kernelBPP * (
                                info[p]->conv_weights_offset*(mapNo+i+1) -
                                    info[p]->conv_weights_channel_offset);

                        convolve((void**)inputPtr, (void**)&localOutputTmp, 
                                    info[p]->conv_weights + taps_offset, conv_width);
                        if (info[p]->conv_biases != NULL) {
                            half b = ((half*)(info[p]->conv_biases))[mapNo + i];
                            accumulateFp16Single((half*)localOutputTmp, b, conv_width);
                        }

                        if (info[p]->channels == 1) {
                            memset64_h((half *) (localOutput[i]), out_buf_sz, 0.0);
                        }
                        accumulate((half **) &localOutput[i], 
                                   (half **) &localOutputTmp, conv_width);

                        output_offset += output_offset_step;
                        dmaWaitTask(ref[1]);
                    }

                    ref[1] = dmaCreateTransactionFullOptions(
                        id,
                        &task[1],
                        localOutput[actual_coalescing-1] + output_src_addr,      
                        output_bytes,                          
                        output_src_width,                     
                        output_dst_width,                    
                        output_src_stride,                    
                        output_dst_stride);                   

                    dmaStartListTask(ref[1]);
                    dmaWaitTask(ref[1]);

                    mapNo += actual_coalescing;
                }
            }
        }
    }
}

void 
im2col_ddr( 
   u32 firstMapNo, 
   u32 lastMapNo, 
   im2col_context *context
)
{
    int shaveId = context->com.shaveId;
    
    im2col_info **info = context->info;
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

    VECDOTPTR_T dot = (VECDOTPTR_T) jumpTable(MV_vecvecmul_hhff);

    setAlignedMem( (firstMapNo == 0 ? shaveId : -1), jumpTable);

    u32 id = dmaInitRequester(shaveId);

    u8 **input_column = getAlignedMem(128, (info[0]->in_row_width/info[0]->tiles) * sizeof(*input_column));
    for (u8 i = 0; i < (info[0]->in_row_width/info[0]->tiles); i++){


                input_column[i] = getAlignedMem(128, info[0]->in_col_height * info[0]->inputBPP);
    }
    half **localOutput = getAlignedMem(128, (info[0]->in_row_width/info[0]->tiles) *sizeof(*localOutput));
    for (u8 i = 0; i < (info[0]->in_row_width/info[0]->tiles); i++){


                localOutput[i] = getAlignedMem(128, info[0]->weight_col_height * info[0]->inputBPP);
    }
    
    for(u8 gr = 0; gr < info[0]->c_group; gr++){
	    for (u16 i = 0; i < info[0]->tiles ; i++){       
	        for (u8 mapNo = firstMapNo; mapNo < lastMapNo; mapNo++){    
	            ref[0] = dmaCreateTransactionFullOptions(
	                id,
	                &task[0],
	                (info[0]->input_column + (gr * info[0]->in_col_height * info[0]->in_row_width * info[0]->inputBPP)) +  i * (info[0]->in_row_width/info[0]->tiles) * info[0]->inputBPP + mapNo * info[0]->inputBPP,  
	                input_column[mapNo],  
	                info[0]->in_col_height * info[0]->inputBPP, 
	                info[0]->inputBPP, 
	                info[0]->in_col_height * info[0]->inputBPP,  
	                info[0]->inputBPP * info[0]->in_row_width,     
	                info[0]->inputBPP);  
	            dmaStartListTask(ref[0]);
	            dmaWaitTask(ref[0]);
	            if (info[0]->c_group == 1){
	                for (int weight_height = 0; weight_height < info[0]->weight_col_height; weight_height++){
	                    localOutput[mapNo][weight_height] = (half)dot((half*)(info[0]->conv_weights + (weight_height*info[0]->weight_row_width*info[0]->kernelBPP)), (half*)input_column[mapNo], info[0]->in_col_height);
	                }    
	            }
	            else{
	                if(gr == 0){
	                    for (int weight_height = 0; weight_height < info[0]->weight_col_height; weight_height++){
	                        localOutput[mapNo][weight_height] = (half)dot((half*)(info[0]->conv_weights + (weight_height*info[0]->weight_row_width*info[0]->kernelBPP)), (half*)input_column[mapNo], info[0]->in_col_height);
	                    }
	                }
	                else{
	                	for (int weight_height = info[0]->weight_col_height; weight_height < info[0]->c_group * info[0]->weight_col_height; weight_height++){
	                        localOutput[mapNo][weight_height - info[0]->weight_col_height] = (half)dot((half*)(info[0]->conv_weights + (weight_height * info[0]->weight_row_width * info[0]->kernelBPP)), (half*)input_column[mapNo], info[0]->in_col_height);
	                    }    
	                }
	            }    
	            if (info[0]->conv_biases != NULL) {
		        	half *bias = (half *)(info[0]->conv_biases);
		        	if(gr == 0){
		                for (int k = 0; k < info[0]->weight_col_height; k++) {
		                    localOutput[mapNo][k] += bias[k];
		                }
		            }

	                	else{
			                for (int k = info[0]->weight_col_height; k < info[0]->c_group * info[0]->weight_col_height; k++) {
			                    localOutput[mapNo][k - info[0]->weight_col_height] += bias[k];
			                }
	                	} 
	            }
	            if (info[0]->with_relu) {
	                relu_inplace64_h(localOutput[mapNo], info[0]->weight_col_height);
	            }    
	            ref[0] = dmaCreateTransactionFullOptions(
	                id,
	                &task[0],
	                (u8 *)(localOutput[mapNo]), 
	                info[0]->output + mapNo * info[0]->outputBPP + i * (info[0]->in_row_width/info[0]->tiles) * info[0]->outputBPP + (gr * info[0]->in_row_width * info[0]->weight_col_height * info[0]->outputBPP),
	                info[0]->weight_col_height * info[0]->outputBPP,  
	                info[0]->outputBPP * info[0]->weight_col_height,
	                info[0]->outputBPP,
	                info[0]->outputBPP,             
	                info[0]->in_row_width * info[0]->outputBPP);               
	            dmaStartListTask(ref[0]);
	            dmaWaitTask(ref[0]);                
	        }        
	    }    
	}
}    
     