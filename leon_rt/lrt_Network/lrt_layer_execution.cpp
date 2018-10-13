//-------------------Includes----------------------------//
#include <cassert>
#include "mv_types.h"
#include "lrt_caffe_layers.h"
#include "network_defines.h"
#include "ddr_functions.h"
#include "ddr_functions_exports.h"
#include <stdio.h>

//------------Execution Initialization Includes----------//
extern "C"{
#include "lrt_dma_computation.h"
#include "lrt_dma_computation_defines.h"
}
#if CONVOLUTION
    #include "conv_api.h"
    #include "im2col_api.h"
#endif

#if POOLING
    #include "pool_api.h"
#endif

#if INNERPRODUCT
    #include "fc_api.h"
#endif

#if LRN
    #include "lrn_api.h"
#endif

//-------------Shave Execution Includes-----------//
#include <swcShaveLoader.h>
#include <DrvShaveL2Cache.h>
#include <DrvLeonL2C.h>
#include <DrvSvu.h>
#include <DrvCpr.h>
// #ifdef DUAL_CPU
    #include <DrvTimer.h> //uncomment for time measuring
// #endif

//---------------------------------Global Definitions-------------------------------//
#ifdef PROFILE
u8 __attribute__((section(".ddr_direct.data"))) sampling_semaphore;
u8 __attribute__((section(".ddr_direct.data"))) setting_semaphore;
#endif

#if CONVOLUTION
static conv_info __attribute__((section(".cmx_direct.data"), aligned (16))) Convolution_object[4];
static conv_info __attribute__((section(".cmx_direct.data"), aligned (16))) *convolution_object_table[4];
static im2col_info __attribute__((section(".cmx_direct.data"), aligned (16))) Im2Col_object[841];
static im2col_info __attribute__((section(".cmx_direct.data"), aligned (16))) *im2col_object_table[841];
#endif

#if POOLING
static pool_info __attribute__((section(".cmx_direct.data"), aligned (16))) Pooling_object[4];
static pool_info __attribute__((section(".cmx_direct.data"), aligned (16))) *pooling_object_table[4];
#endif

#if INNERPRODUCT
static fc_info __attribute__((section(".cmx_direct.data"), aligned (16))) InnerProduct_object;
#endif

#if LRN
static lrn_info __attribute__((section(".cmx_direct.data"), aligned (16))) LRN_object;
#endif

extern u32 jumpTableAddr;

//--------------Shave Entrypoints for execution--------------------//
#if CONVOLUTION
extern u32 RTCNN0_shave_conv;
extern u32 RTCNN1_shave_conv;
extern u32 RTCNN2_shave_conv;
extern u32 RTCNN3_shave_conv;
extern u32 RTCNN4_shave_conv;
extern u32 RTCNN5_shave_conv;
extern u32 RTCNN6_shave_conv;
extern u32 RTCNN7_shave_conv;
extern u32 RTCNN8_shave_conv;
extern u32 RTCNN9_shave_conv;
extern u32 RTCNN10_shave_conv;
extern u32 RTCNN11_shave_conv;

extern u32 RTCNN0_shave_im2col;
extern u32 RTCNN1_shave_im2col;
extern u32 RTCNN2_shave_im2col;
extern u32 RTCNN3_shave_im2col;
extern u32 RTCNN4_shave_im2col;
extern u32 RTCNN5_shave_im2col;
extern u32 RTCNN6_shave_im2col;
extern u32 RTCNN7_shave_im2col;
extern u32 RTCNN8_shave_im2col;
extern u32 RTCNN9_shave_im2col;
extern u32 RTCNN10_shave_im2col;
extern u32 RTCNN11_shave_im2col;
#endif

#if POOLING
extern u32 RTCNN0_shave_pool;
extern u32 RTCNN1_shave_pool;
extern u32 RTCNN2_shave_pool;
extern u32 RTCNN3_shave_pool;
extern u32 RTCNN4_shave_pool;
extern u32 RTCNN5_shave_pool;
extern u32 RTCNN6_shave_pool;
extern u32 RTCNN7_shave_pool;
extern u32 RTCNN8_shave_pool;
extern u32 RTCNN9_shave_pool;
extern u32 RTCNN10_shave_pool;
extern u32 RTCNN11_shave_pool;
#endif

#if INNERPRODUCT
extern u32 RTCNN0_shave_fc;
extern u32 RTCNN1_shave_fc;
extern u32 RTCNN2_shave_fc;
extern u32 RTCNN3_shave_fc;
extern u32 RTCNN4_shave_fc;
extern u32 RTCNN5_shave_fc;
extern u32 RTCNN6_shave_fc;
extern u32 RTCNN7_shave_fc;
extern u32 RTCNN8_shave_fc;
extern u32 RTCNN9_shave_fc;
extern u32 RTCNN10_shave_fc;
extern u32 RTCNN11_shave_fc;
#endif

#if LRN
extern u32 RTCNN0_shave_lrn;
extern u32 RTCNN1_shave_lrn;
extern u32 RTCNN2_shave_lrn;
extern u32 RTCNN3_shave_lrn;
extern u32 RTCNN4_shave_lrn;
extern u32 RTCNN5_shave_lrn;
extern u32 RTCNN6_shave_lrn;
extern u32 RTCNN7_shave_lrn;
extern u32 RTCNN8_shave_lrn;
extern u32 RTCNN9_shave_lrn;
extern u32 RTCNN10_shave_lrn;
extern u32 RTCNN11_shave_lrn;
#endif

//----------------------------Entrypoints Tables---------------------//
#if CONVOLUTION
static u32 startShave_conv[] =
{
    (u32) RTCNN0_shave_conv,
    (u32) RTCNN1_shave_conv,
    (u32) RTCNN2_shave_conv,
    (u32) RTCNN3_shave_conv,
    (u32) RTCNN4_shave_conv,
    (u32) RTCNN5_shave_conv,
    (u32) RTCNN6_shave_conv,
    (u32) RTCNN7_shave_conv,
    (u32) RTCNN8_shave_conv,
    (u32) RTCNN9_shave_conv,
    (u32) RTCNN10_shave_conv,
    (u32) RTCNN11_shave_conv,
};

static u32 startShave_im2col[] =
{
    (u32) RTCNN0_shave_im2col,
    (u32) RTCNN1_shave_im2col,
    (u32) RTCNN2_shave_im2col,
    (u32) RTCNN3_shave_im2col,
    (u32) RTCNN4_shave_im2col,
    (u32) RTCNN5_shave_im2col,
    (u32) RTCNN6_shave_im2col,
    (u32) RTCNN7_shave_im2col,
    (u32) RTCNN8_shave_im2col,
    (u32) RTCNN9_shave_im2col,
    (u32) RTCNN10_shave_im2col,
    (u32) RTCNN11_shave_im2col,
};
#endif

#if POOLING
static u32 startShave_pool[] =
{
    (u32) RTCNN0_shave_pool,
    (u32) RTCNN1_shave_pool,
    (u32) RTCNN2_shave_pool,
    (u32) RTCNN3_shave_pool,
    (u32) RTCNN4_shave_pool,
    (u32) RTCNN5_shave_pool,
    (u32) RTCNN6_shave_pool,
    (u32) RTCNN7_shave_pool,
    (u32) RTCNN8_shave_pool,
    (u32) RTCNN9_shave_pool,
    (u32) RTCNN10_shave_pool,
    (u32) RTCNN11_shave_pool,
};
#endif

#if INNERPRODUCT
static u32 startShave_InnerProduct[] =
{
    (u32) RTCNN0_shave_fc,
    (u32) RTCNN1_shave_fc,
    (u32) RTCNN2_shave_fc,
    (u32) RTCNN3_shave_fc,
    (u32) RTCNN4_shave_fc,
    (u32) RTCNN5_shave_fc,
    (u32) RTCNN6_shave_fc,
    (u32) RTCNN7_shave_fc,
    (u32) RTCNN8_shave_fc,
    (u32) RTCNN9_shave_fc,
    (u32) RTCNN10_shave_fc,
    (u32) RTCNN11_shave_fc,
};
#endif

#if LRN
static u32 startShave_LRN[] =
{
    (u32) RTCNN0_shave_lrn,
    (u32) RTCNN1_shave_lrn,
    (u32) RTCNN2_shave_lrn,
    (u32) RTCNN3_shave_lrn,
    (u32) RTCNN4_shave_lrn,
    (u32) RTCNN5_shave_lrn,
    (u32) RTCNN6_shave_lrn,
    (u32) RTCNN7_shave_lrn,
    (u32) RTCNN8_shave_lrn,
    (u32) RTCNN9_shave_lrn,
    (u32) RTCNN10_shave_lrn,
    (u32) RTCNN11_shave_lrn,
};
#endif

//-------------------------------Layer Execution Implementations-------------------------//
#if CONVOLUTION
u64 Convolution::execute(){

    if (ddr_function<=23){
        u8 lines = 1;

        if ((this->bottom_input_height * this->bottom_input_width) >= 224 * 224){
            lines = 4;
        }
        else if ((this->bottom_input_height * this->bottom_input_width) >= 120 * 120){
            lines = 2;
        }

        // printf("\nedw0\n");
        for (u8 i = 0; i < lines; i++){

            convolution_object_table[i] = &(Convolution_object[i]);

            convolution_object_table[i]->input = this->bottom_output_buffer;
            convolution_object_table[i]->output = this->output_buffer;
            convolution_object_table[i]->conv_weights = (u8*)((u32)(this->weight_pointer) & (u32)0x8fffffff);
            convolution_object_table[i]->conv_biases = (u8*)((u32)(this->bias_pointer) & (u32)0x8fffffff);

            convolution_object_table[i]->c_group = this->group;
            convolution_object_table[i]->channels = this->bottom_channels / this->group;
            convolution_object_table[i]->output_channel_offset = this->input_height * this->input_width;
            convolution_object_table[i]->kernel_h = this->kernel_size;
            convolution_object_table[i]->kernel_w = this->kernel_size;
            convolution_object_table[i]->with_relu = this->ReLU_flag;
            convolution_object_table[i]->conv_weights_channel_offset = this->kernel_size * this->kernel_size;
            convolution_object_table[i]->in_stride = this->stride;

            convolution_object_table[i]->input_channel_offset = this->bottom_input_height * this->bottom_input_width;
            convolution_object_table[i]->conv_weights_offset = this->kernel_size * this->kernel_size * (this->bottom_channels/this->group);

            convolution_object_table[i]->splits = lines;
            convolution_object_table[i]->maps = channels;

            convolution_object_table[i]->inputBPP = 2;
            convolution_object_table[i]->outputBPP = 2;
            convolution_object_table[i]->kernelBPP = 2;
            convolution_object_table[i]->coalescing_num = 1;

            while (convolution_object_table[i]->coalescing_num * ((this->input_height * this->input_width)/lines) * 2 < 20000){
                    convolution_object_table[i]->coalescing_num ++;                
                    if((convolution_object_table[i]->coalescing_num * ((this->input_height * this->input_width)/lines) * 2 > 20000)||(convolution_object_table[i]->coalescing_num > channels/10)){
                        break;
                    }
            }
            // convolution_object_table[i]->coalescing_num = 6;

            // while (convolution_object_table[i]->coalescing_num * ((this->input_height * this->input_width)/lines) * 2 < 20000){
            //     convolution_object_table[i]->coalescing_num ++;                
            // }
            
            convolution_object_table[i]->ddr_function = ddr_function;

            conv_prepare_dma((convolution_object_table[i]), this->bottom_input_height, this->bottom_input_width, lines, i, this->kernel_size, this->kernel_size, this->pad, this->pad, this->stride, this->stride, 1);
        }
        // printf("\nedw1\n");
        u8 first_shave = 12 - this->shaves_used;
        u8 last_shave = 11;

        u16 number_of_outputs_per_shave = channels / this->shaves_used;
        u16 remained_outputs = channels % this->shaves_used;

        u16 first_map = 0;
        u16 last_map = 0;

        #ifdef PROFILE
                    setting_semaphore = 1;
            while(!sampling_semaphore){DrvTimerSleepMs(20);}
            while (sampling_semaphore){
        #elif defined(DUAL_CPU)
            tyTimeStamp clock_ticks;
            u64 cpu_cycles = 0;
            s32 sc;
            sc = DrvTimerStartTicksCount(&clock_ticks);
            assert(!sc);
        #endif
            // printf("\nedw2\n");
            u16 remained_outputs_iter = remained_outputs;
            first_map = 0;
            last_map = 0;

            DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

            for(u8 shave_index = 0; shave_index < this->shaves_used; shave_index++){
                last_map += number_of_outputs_per_shave;
                if(remained_outputs_iter > 0){
                    last_map++;
                    remained_outputs_iter--;
                }
                swcResetShave(first_shave + shave_index);
                swcSetAbsoluteDefaultStack(first_shave + shave_index);

                swcStartShaveCC(first_shave + shave_index,
                                (u32) startShave_conv[shave_index + first_shave],
                                "iiii",
                                convolution_object_table, first_map, last_map, jumpTableAddr);
                
                first_map = last_map;
                if(last_map > channels){
                    break;
                }
            }
            for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
                swcWaitShave(shave_index);
            }
        #ifdef PROFILE
        }
        #endif
        for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
            DrvSvutStop(shave_index);
        }
        DrvCprTurnOffShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

        #ifdef DUAL_CPU
            sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
            assert(!sc);
            return cpu_cycles;
        #elif defined(PROFILE)
            return 0;
        #endif
    }
    else{
        u16 tiles = 1;
        int array_size = 0;
        u16 width_size =this->input_width * this->input_height;
        array_size = this->input_height * this->input_width * 2 * bottom_channels * kernel_size * kernel_size;
        while (true){
            tiles ++;
            if((array_size/tiles<50000)&&((width_size)%tiles==0))
                break;
        }
        
        


        //περιπτωσεις δικτυων που δεν πιανονται παραπανω
        if ((array_size == 2709504)||(array_size == 14450688)){
            tiles = 512;
        } 
        if (array_size < 85000){
            tiles = 1;
            if ((array_size == 50176)&&(width_size==784)){
                tiles = 4;
            }
            else if ((array_size == 25088)&&(width_size==196))
                tiles = 4;
            else if ((array_size == 73728)&&(width_size==36))
                tiles = 6;
        }
        else if ((array_size <210000)&&(width_size!=3136)){
            tiles = 4;
            if ((array_size == 200704)&&(width_size==196)){
                tiles = 8;
            }
        }
        else if ((array_size < 402000)&&(width_size==748)){
            tiles = 4;
        }   
        else if ((array_size == 100352)&&(width_size==3136)){
            tiles = 14;
            // printf("\n14\n");
        } 
        else if ((array_size == 559872)&&(width_size==2916)){
            tiles = 27;
        }
        
        while ((width_size/tiles)<shaves_used){
            tiles--;
            if (((width_size/tiles)>shaves_used))
                break;
        }
        for (u16 i = 0; i < tiles; i++){

            im2col_object_table[i] = &(Im2Col_object[i]);
            im2col_object_table[i]->input = (u8*)((u32)(bottom_output_buffer) & (u32)0x8fffffff);

            im2col_object_table[i]->input = bottom_output_buffer;
            im2col_object_table[i]->output = this->output_buffer;
            im2col_object_table[i]->conv_weights = (u8*)((u32)(this->weight_pointer) & (u32)0x8fffffff);
            im2col_object_table[i]->conv_biases = (u8*)((u32)(this->bias_pointer) & (u32)0x8fffffff);
            im2col_object_table[i]->tiles = tiles;

            im2col_object_table[i]->c_group = this->group;
            im2col_object_table[i]->channels = bottom_channels / this->group;
            im2col_object_table[i]->output_channel_offset = this->input_height * this->input_width;
            im2col_object_table[i]->kernel_h = this->kernel_size;
            im2col_object_table[i]->kernel_w = this->kernel_size;
            im2col_object_table[i]->with_relu = this->ReLU_flag;
            im2col_object_table[i]->conv_weights_channel_offset = this->kernel_size * this->kernel_size;
            im2col_object_table[i]->in_stride = this->stride;

            im2col_object_table[i]->input_channel_offset = bottom_input_height * bottom_input_width;
            im2col_object_table[i]->conv_weights_offset = this->kernel_size * this->kernel_size * (bottom_channels/this->group);

            im2col_object_table[i]->maps = channels;

            im2col_object_table[i]->inputBPP = 2;
            im2col_object_table[i]->outputBPP = 2;
            im2col_object_table[i]->kernelBPP = 2;
            
            im2col_object_table[i]->ddr_function = ddr_function;
            im2col_object_table[i]->weight_col_height = channels/this->group;
            im2col_object_table[i]->weight_row_width = im2col_object_table[i]->channels * this->kernel_size * this->kernel_size; 
            im2col_object_table[i]->in_col_height = (bottom_channels / this->group) * this->kernel_size * this->kernel_size;
            im2col_object_table[i]->in_row_width = width_size;
            
            // to offset toy kathe kommatioy tha einai  
            im2col_object_table[i]->offset = ((im2col_object_table[0]->in_row_width)/(u32)tiles)*im2col_object_table[0]->in_col_height*im2col_object_table[i]->inputBPP;
            
        }
            
        // an exw padding vazw mhdenika se kathe input channel stis gwnies prin steilw thn eisodo gia metatroph
        if (this->pad != 0){
            im2col_object_table[0]->input_column = (u8*)((u32)NewMapWithPad(bottom_output_buffer, bottom_input_width, bottom_input_height, this->kernel_size, this->stride, this->pad, bottom_channels) & (u32)0x8fffffff);
        }
        else{
            im2col_object_table[0]->input_column = (u8*)((u32)InputstoColumns(bottom_output_buffer, bottom_input_width, bottom_input_height, this->kernel_size, this->stride, this->pad, bottom_channels) & (u32)0x8fffffff);
        }
        u16 first_shave = 12 - this->shaves_used;
        u16 last_shave = 11;
        
        //to kathe kommati poy xwraei stous shave to spaw se oso to plhthos twn shave gia na analavei o kathenas
        u16 number_of_outputs_per_shave =  (im2col_object_table[0]->in_row_width/tiles)/ shaves_used;
        u16 remained_outputs = (im2col_object_table[0]->in_row_width/tiles) % shaves_used;

        u16 first_map = 0;
        u16 last_map = 0;

        #ifdef PROFILE
                    setting_semaphore = 1;
            while(!sampling_semaphore){}
            while (sampling_semaphore){
        #elif defined(DUAL_CPU)
            tyTimeStamp clock_ticks;
            u64 cpu_cycles = 0;
            s32 sc;
            sc = DrvTimerStartTicksCount(&clock_ticks);
            assert(!sc);
        #endif

            u16 remained_outputs_iter = remained_outputs;
            first_map = 0;
            last_map = 0;

            DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

            
            for(u16 shave_index = 0; shave_index < shaves_used; shave_index++)
            {
                last_map += number_of_outputs_per_shave;
                if(remained_outputs_iter > 0)
                {
                    last_map++;
                    remained_outputs_iter--;
                }
                    
                swcResetShave(first_shave + shave_index);
                swcSetAbsoluteDefaultStack(first_shave + shave_index);
                
                    // printf("\n%d %d\n", first_map, last_map);
                swcStartShaveCC(first_shave + shave_index, 
                            (u32) startShave_im2col[shave_index + first_shave],
                            "iiii", 
                            im2col_object_table, first_map, last_map, jumpTableAddr);
                
                first_map = last_map;
                if(last_map > (im2col_object_table[0]->in_row_width/tiles)){
                    break;
                }
            }
            for(u16 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
                swcWaitShave(shave_index);
            }
        #ifdef PROFILE
        }
        #endif
        for (u16 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
                DrvSvutStop(shave_index);
        }
        DrvCprTurnOffShaveMask((1 << (last_shave + 1)) - (1 << first_shave));     
        
        #ifdef DUAL_CPU
            sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
            assert(!sc);
            return cpu_cycles;
        #elif defined(PROFILE)
            return 0;
        #endif
    }
}
#endif

#if POOLING
u64 Pooling::execute(){

    u8 lines = 1;

    if ((this->bottom_input_height * this->bottom_input_width) >= 224 * 224){
        lines = 4;
    }
    else if ((this->bottom_input_height * this->bottom_input_width) >= 120 * 120){
        lines = 2;
    }

    for (u8 i = 0; i < lines; i++){

        pooling_object_table[i] = &(Pooling_object[i]);

        pooling_object_table[i]->input = this->bottom_output_buffer;
        pooling_object_table[i]->output = this->output_buffer;

        pooling_object_table[i]->output_channel_offset = this->input_height * this->input_width;
        pooling_object_table[i]->line_width = this->channels;
        pooling_object_table[i]->kernel_h = this->kernel_size;
        pooling_object_table[i]->type = this->pooling_method;

        pooling_object_table[i]->input_channel_offset = this->bottom_input_height * this->bottom_input_width;
        pooling_object_table[i]->channels = this->bottom_channels;

        pooling_object_table[i]->inputBPP = 2;
        pooling_object_table[i]->outputBPP = 2;
        pooling_object_table[i]->in_buffers_num = 2;
        pooling_object_table[i]->splits = lines;

        pooling_object_table[i]->ddr_function = ddr_function;
        pool_prepare_dma(pooling_object_table[i], this->bottom_input_height, this->bottom_input_width, lines, i, this->kernel_size, this->kernel_size, this->pad, this->pad, this->stride, this->stride, 1);
    }

    u8 first_shave = 12 - this->shaves_used;
    u8 last_shave = 11;

    u16 number_of_outputs_per_shave = channels / shaves_used;
    u16 remained_outputs = channels % shaves_used;

    u16 first_map = 0;
    u16 last_map = 0;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef PROFILE
                setting_semaphore = 1;
        while(!sampling_semaphore){DrvTimerSleepMs(20);}
        while (sampling_semaphore){
    #elif defined(DUAL_CPU)
        tyTimeStamp clock_ticks;
        u64 cpu_cycles = 0;
        s32 sc;
        sc = DrvTimerStartTicksCount(&clock_ticks);
        assert(!sc);
    #endif
        u16 remained_outputs_iter = remained_outputs;
        first_map = 0;
        last_map = 0;

        for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){
            last_map += number_of_outputs_per_shave;
            if(remained_outputs_iter > 0){
                last_map++;
                remained_outputs_iter--;
            }

            swcResetShave(first_shave + shave_index);
            swcSetAbsoluteDefaultStack(first_shave + shave_index);

            swcStartShaveCC(first_shave + shave_index,
                            (u32) startShave_pool[shave_index + first_shave],
                            "iiii",
                            pooling_object_table, first_map, last_map, jumpTableAddr);

            first_map = last_map;
            if(last_map > channels)
                break;
        }
        for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
            swcWaitShave(shave_index);
        }
    #ifdef PROFILE
    }
    #endif
    for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
        DrvSvutStop(shave_index);
    }
    DrvCprTurnOffShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef DUAL_CPU
        sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
        assert(!sc);
        return cpu_cycles;
    #elif defined(PROFILE)
        return 0;
    #endif
}
#endif

#if INNERPRODUCT
u64 InnerProduct::execute(){

    fc_info *execute_InnerProduct = &InnerProduct_object;

    execute_InnerProduct->vector = this->bottom_output_buffer;
    execute_InnerProduct->output = this->output_buffer;
    execute_InnerProduct->bias = (u8*)((u32)(this->bias_pointer) & 0x8fffffff);
    execute_InnerProduct->weightLines = (u8*)((u32)(this->weight_pointer) & 0x8fffffff);

    execute_InnerProduct->linesNo = this->input_width;
    execute_InnerProduct->with_relu = this->ReLU_flag;

    execute_InnerProduct->inputWidth = this->bottom_input_width * this->bottom_input_height * this->bottom_channels;
    execute_InnerProduct->weightLines_offset = this->bottom_input_width * this->bottom_input_height * this->bottom_channels;

    execute_InnerProduct->inputBPP = 2;
    execute_InnerProduct->outputBPP = 2;

    shaves_used = (input_width < shaves_used ? input_width : shaves_used);

    u8 first_shave = 12 - this->shaves_used;
    u8 last_shave = 11;

    u16 number_of_outputs_per_shave = input_width / shaves_used;
    u16 remained_outputs = input_width % shaves_used;

    u16 first_map = 0;
    u16 last_map = 0;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef PROFILE
                setting_semaphore = 1;
        while(!sampling_semaphore){DrvTimerSleepMs(20);}
        while (sampling_semaphore){
    #elif defined(DUAL_CPU)
        tyTimeStamp clock_ticks;
        u64 cpu_cycles = 0;
        s32 sc;
        sc = DrvTimerStartTicksCount(&clock_ticks);
        assert(!sc);
    #endif
        u16 remained_outputs_iter = remained_outputs;
        first_map = 0;
        last_map = 0;

        for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

            last_map += number_of_outputs_per_shave;
            if(remained_outputs_iter > 0){

                last_map++;
                remained_outputs_iter--;
            }

            swcResetShave(first_shave + shave_index);
            swcSetAbsoluteDefaultStack(first_shave + shave_index);

            swcStartShaveCC(first_shave + shave_index,
                            (u32) startShave_InnerProduct[shave_index + first_shave],
                            "iiii",
                            execute_InnerProduct, first_map, last_map, jumpTableAddr);
            first_map = last_map;
            if(last_map > input_width){
                break;
            }
        }
        for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
            swcWaitShave(shave_index);
        }
    #ifdef PROFILE
    }
    #endif
    for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
        DrvSvutStop(shave_index);
    }
    DrvCprTurnOffShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef DUAL_CPU
        sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
        assert(!sc);
        return cpu_cycles;
    #elif defined(PROFILE)
        return 0;
    #endif
}
#endif

#if LRN
u64 Lrn::execute(){

    lrn_info *execute_lrn = &LRN_object;

    this->channels = this->bottom_channels;
    this->input_height = this->bottom_input_height;
    this->input_width = this->bottom_input_width;

    execute_lrn->input = this->bottom_output_buffer;
    execute_lrn->output = this->output_buffer;
    execute_lrn->image_offset = this->bottom_input_width * this->bottom_input_height;

    execute_lrn->BPP = 2;

    execute_lrn->channels = this->channels;
    execute_lrn->ddr_function = this->ddr_function;

    execute_lrn->local_ratio = this->local_size;
    execute_lrn->alpha = this->alpha;
    execute_lrn->beta = this->beta;

    u8 first_shave = 12 - this->shaves_used;
    u8 last_shave = 11;

    u32 pixels_per_shave = (this->input_width * this->input_height) / (shaves_used);
    u16 remaining_pixels = (this->input_width * this->input_height) % (shaves_used);

    u32 first_pixel = 0;
    u32 last_pixel = 0;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef PROFILE
                setting_semaphore = 1;
        while(!sampling_semaphore){DrvTimerSleepMs(20);}
        while (sampling_semaphore){
    #elif defined(DUAL_CPU)
        tyTimeStamp clock_ticks;
        u64 cpu_cycles = 0;
        s32 sc;
        sc = DrvTimerStartTicksCount(&clock_ticks);
        assert(!sc);
    #endif
        u16 remaining_pixels_iter = remaining_pixels;
        first_pixel = 0;
        last_pixel = 0;

        for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

            last_pixel += pixels_per_shave;
            if(remaining_pixels_iter){
                last_pixel++;
                remaining_pixels_iter--;
            }
            swcResetShave(first_shave + shave_index);
            swcSetAbsoluteDefaultStack(first_shave + shave_index);
            swcStartShaveCC(first_shave + shave_index, (u32) startShave_LRN[shave_index + first_shave], "iiii", execute_lrn, first_pixel, last_pixel, jumpTableAddr);
            first_pixel = last_pixel;
        }

        for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
            swcWaitShave(shave_index);
        }
    #ifdef PROFILE
    }
    #endif
    for (u8 shave_index = first_shave; shave_index < last_shave + 1; shave_index++){
        DrvSvutStop(shave_index);
    }
    DrvCprTurnOffShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    #ifdef DUAL_CPU
        sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
        assert(!sc);
        return cpu_cycles;
    #elif defined(PROFILE)
        return 0;
    #endif
}
#endif
