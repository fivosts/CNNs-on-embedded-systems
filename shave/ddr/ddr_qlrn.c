#include "local_resp_norm.h"
#include "utils.h"
#include "math.h"

void LRN_AcrossChannels_generic(fp16** input, fp16** output, u16 channels, u16 pixel_batch,
                                u8 local_ratio, fp16 alpha, fp16 beta){

    int low_margin = 0;
    int up_margin = 0;

    float alpha_f = alpha;
    float beta_f = beta;
    float squares = 0;
    float sum = 0;
    float current_input_pixel = 0;
    float current_output_pixel = 0;

    for (u32 pix = 0; pix < pixel_batch; pix++){
        for (u32 ch = 0; ch < channels; ch++){
            sum = 0;

            low_margin = ch-(local_ratio/2);
            low_margin = ( low_margin < 0 ? 0 : low_margin);

            up_margin = ( ((channels - 1) < (ch + (local_ratio / 2) + (local_ratio % 2) - 1)) ? (channels - 1) : ch + (local_ratio / 2) + (local_ratio % 2) - 1 );

            current_input_pixel = input[0][ch * pixel_batch + pix];

            for(int l = low_margin; l < (up_margin + 1); l++){
                squares = input[0][(pixel_batch)*(l) + pix ];
                squares *= squares;
                sum += squares;
            }
            current_output_pixel = current_input_pixel / pow((1 + (alpha_f / local_ratio) * sum), beta_f);
            output[0][ch * pixel_batch + pix] = current_output_pixel;
        }
    }
    return;
}

void lrn_ddr_generic(u32 first_pixel, u32 last_pixel, lrn_context *context){

    int shaveId = context->com.shaveId;
    lrn_info *info = context->info;
    J_FUNCPTR_T jumpTable = context->com.jumpTable;

    SETALIGNEDMEM_PTR setAlignedMem = context->mem.setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem = context->mem.getAlignedMem;

    DMAINITREQUESTER_PTR dmaInitRequester = context->dma.dmaInitRequester;
    DMACREATETRANSACTIONFULLOPTIONS_PTR dmaCreateTransactionFullOptions = context->dma.dmaCreateTransactionFullOptions;
    DMASTARTLISTTASK_PTR dmaStartListTask = context->dma.dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask = context->dma.dmaWaitTask;
    dmaTransactionList_t *task = context->dma.task;
    dmaTransactionList_t **ref = context->dma.ref;

    LRNPTRGENERIC_T lrn_across_channels = (LRNPTRGENERIC_T) jumpTable(FT_LRN_AcrossChannels_generic);

    setAlignedMem( (first_pixel == 0 ? shaveId : -1), jumpTable);

    u32 pixels_per_batch = 0;
    u32 batch_to_allocate = 0;
    u32 pixel_remainder = 0;
    u32 pixel_offset = first_pixel;
    u8 buffer_switch = 0;
    u8 dual = 0;

    u8 *inputAddr = info->input;
    u8 *outputAddr = info->output;
    u8 *input_pointer[2];

    if (((MEMORY_POOL - 4 * 1024)/(info->channels * 2 * 2)) < (last_pixel - first_pixel)){
    	pixels_per_batch = ((MEMORY_POOL - (18 * info->channels)) / (info->channels * 2 * 3));
        batch_to_allocate = (pixels_per_batch % 4 ? ((pixels_per_batch / 4) + 1) * 4 : pixels_per_batch);
	    buffer_switch = 1;
        dual = 1;
	    pixel_remainder = (((last_pixel - first_pixel) % pixels_per_batch) != 0 ? ((last_pixel - first_pixel) % pixels_per_batch) : pixels_per_batch);
    }
    else {
        pixels_per_batch = last_pixel - first_pixel;
        batch_to_allocate = (pixels_per_batch % 4 ? ((pixels_per_batch/ 4) + 1) * 4 : pixels_per_batch);
    }

    u32 buffer_size = round_up64(info->BPP * batch_to_allocate * info->channels);

	input_pointer[0] = getAlignedMem(32, buffer_size);
    u8 *local_output = getAlignedMem(32, buffer_size);

    u32 id = dmaInitRequester(shaveId);

	if (dual){

        input_pointer[1] = getAlignedMem(32, buffer_size);
    	ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[0], pixel_remainder * info->channels * info->BPP,
    												pixel_remainder * info->BPP, pixel_remainder * info->BPP, info->image_offset * info->BPP, pixel_remainder * info->BPP);
        dmaStartListTask(ref[0]);
    	dmaWaitTask(ref[0]);
    	pixel_offset += pixel_remainder;
	}

	ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[buffer_switch % 2], pixels_per_batch * info->channels * info->BPP,
													pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP, pixels_per_batch * info->BPP);
    dmaStartListTask(ref[0]);
    pixel_remainder = (buffer_switch ? pixel_remainder : pixels_per_batch);
    dmaWaitTask(ref[0]);


	lrn_across_channels((fp16**)&input_pointer[0], (fp16**)&local_output, info->channels, pixel_remainder, info->local_ratio, info->alpha, info->beta);

    ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + first_pixel * info->BPP, pixel_remainder * info->channels * info->BPP,
                                                        pixel_remainder * info->BPP, pixel_remainder * info->BPP, pixel_remainder * info->BPP, info->image_offset * info->BPP);
    dmaStartListTask(ref[1]);
    dmaWaitTask(ref[1]);

    if (dual){

        pixel_offset += pixels_per_batch;
        buffer_switch = (buffer_switch + 1) % 2;    

        while (pixel_offset < last_pixel){

            ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[buffer_switch % 2], pixels_per_batch * info->channels * info->BPP,
                                                          pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP, pixels_per_batch * info->BPP);
            dmaStartListTask(ref[0]);


            lrn_across_channels((fp16**)&input_pointer[(buffer_switch + 1) % 2], (fp16**)&local_output, info->channels, pixels_per_batch, info->local_ratio, info->alpha, info->beta);

            ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + info->BPP*(pixel_offset - pixels_per_batch), pixels_per_batch * info->channels * 2, //TODO offset to outaddress
                                                            pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP);
            dmaStartListTask(ref[1]);
            pixel_offset += pixels_per_batch;
            buffer_switch = (buffer_switch + 1) % 2;
            dmaWaitTask(ref[0]);
            dmaWaitTask(ref[1]);
        }


        lrn_across_channels((fp16**)&input_pointer[(buffer_switch + 1) % 2], (fp16**)&local_output, info->channels, pixels_per_batch, info->local_ratio, info->alpha, info->beta);


        ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + info->BPP*(pixel_offset - pixels_per_batch), pixels_per_batch * info->channels * 2, //TODO offset to outaddress
                                                        pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP);
        dmaStartListTask(ref[1]);
        dmaWaitTask(ref[1]);
    }
    return;
}

void lrn_ddr_LR5_A0_0001_B_0_75(u32 first_pixel, u32 last_pixel, lrn_context *context){

    int shaveId = context->com.shaveId;
    lrn_info *info = context->info;
    J_FUNCPTR_T jumpTable = context->com.jumpTable;

    SETALIGNEDMEM_PTR setAlignedMem = context->mem.setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem = context->mem.getAlignedMem;

    DMAINITREQUESTER_PTR dmaInitRequester = context->dma.dmaInitRequester;
    DMACREATETRANSACTIONFULLOPTIONS_PTR dmaCreateTransactionFullOptions = context->dma.dmaCreateTransactionFullOptions;
    DMASTARTLISTTASK_PTR dmaStartListTask = context->dma.dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask = context->dma.dmaWaitTask;
    dmaTransactionList_t *task = context->dma.task;
    dmaTransactionList_t **ref = context->dma.ref;

    LRNSPECIFICPTR_T lrn_across_channels = (LRNSPECIFICPTR_T) jumpTable(FT_LRN_AcrossChannels_LR5_A0_0001_B_0_75);

    setAlignedMem( (first_pixel == 0 ? shaveId : -1), jumpTable);

    u32 pixels_per_batch = 0;
    u32 batch_to_allocate = 0;
    u32 pixel_remainder = 0;
    u32 pixel_offset = first_pixel;
    u8 buffer_switch = 0;
    u8 dual = 0;

    u8 *inputAddr = info->input;
    u8 *outputAddr = info->output;
    u8 *input_pointer[2];

    if (((MEMORY_POOL - (4 * 1024))/(info->channels * 2 * 2)) < (last_pixel - first_pixel)){
        pixels_per_batch = ((MEMORY_POOL - (18 * info->channels)) / (info->channels * 2 * 3));
        batch_to_allocate = (pixels_per_batch % 4 ? ((pixels_per_batch / 4) + 1) * 4 : pixels_per_batch);
        buffer_switch = 1;
        dual = 1;
        pixel_remainder = (((last_pixel - first_pixel) % pixels_per_batch) != 0 ? ((last_pixel - first_pixel) % pixels_per_batch) : pixels_per_batch);
    }
    else {
        pixels_per_batch = last_pixel - first_pixel;
        batch_to_allocate = (pixels_per_batch % 4 ? ((pixels_per_batch/ 4) + 1) * 4 : pixels_per_batch);
    }

    u32 buffer_size = round_up64(info->BPP * batch_to_allocate * info->channels);

    input_pointer[0] = getAlignedMem(32, buffer_size);
    u8 *local_output = getAlignedMem(32, buffer_size);

    u32 id = dmaInitRequester(shaveId);

    if (dual){

        input_pointer[1] = getAlignedMem(32, buffer_size);
        ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[0], pixel_remainder * info->channels * info->BPP,
                                                    pixel_remainder * info->BPP, pixel_remainder * info->BPP, info->image_offset * info->BPP, pixel_remainder * info->BPP);
        dmaStartListTask(ref[0]);
        dmaWaitTask(ref[0]);
        pixel_offset += pixel_remainder;
    }

    ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[buffer_switch % 2], pixels_per_batch * info->channels * info->BPP,
                                                    pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP, pixels_per_batch * info->BPP);
    dmaStartListTask(ref[0]);
    pixel_remainder = (buffer_switch ? pixel_remainder : pixels_per_batch);
    dmaWaitTask(ref[0]);


    lrn_across_channels((fp16**)&input_pointer[0], (fp16**)&local_output, info->channels, pixel_remainder);


    ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + first_pixel * info->BPP, pixel_remainder * info->channels * info->BPP,
                                                        pixel_remainder * info->BPP, pixel_remainder * info->BPP, pixel_remainder * info->BPP, info->image_offset * info->BPP);
    dmaStartListTask(ref[1]);
    dmaWaitTask(ref[1]);

    if (dual){

        pixel_offset += pixels_per_batch;
        buffer_switch = (buffer_switch + 1) % 2;    

        while (pixel_offset < last_pixel){

            ref[0] = dmaCreateTransactionFullOptions(id, &task[0], inputAddr + pixel_offset * info->BPP, input_pointer[buffer_switch % 2], pixels_per_batch * info->channels * info->BPP,
                                                          pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP, pixels_per_batch * info->BPP);
            dmaStartListTask(ref[0]);

            lrn_across_channels((fp16**)&input_pointer[(buffer_switch + 1) % 2], (fp16**)&local_output, info->channels, pixels_per_batch);

            ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + info->BPP*(pixel_offset - pixels_per_batch), pixels_per_batch * info->channels * 2, //TODO offset to outaddress
                                                            pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP);
            dmaStartListTask(ref[1]);
            pixel_offset += pixels_per_batch;
            buffer_switch = (buffer_switch + 1) % 2;
            dmaWaitTask(ref[0]);
            dmaWaitTask(ref[1]);
        }


        lrn_across_channels((fp16**)&input_pointer[(buffer_switch + 1) % 2], (fp16**)&local_output, info->channels, pixels_per_batch);

        ref[1] = dmaCreateTransactionFullOptions(id, &task[1], local_output, outputAddr + info->BPP*(pixel_offset - pixels_per_batch), pixels_per_batch * info->channels * 2, //TODO offset to outaddress
                                                        pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, pixels_per_batch * info->BPP, info->image_offset * info->BPP);
        dmaStartListTask(ref[1]);
        dmaWaitTask(ref[1]);
    }
    return;
}