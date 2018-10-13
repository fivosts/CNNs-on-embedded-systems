//-------------------Includes----------------------------//
#include <cassert>
#include "mv_types.h"
#include "caffe_layers.h"
#include "network_defines.h"
#include "ddr_functions.h"
#include "ddr_functions_exports.h"
//------------Execution Initialization Includes----------//
extern "C"{
#include "dma_computation.h"
#include "dma_computation_defines.h"
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
#include <DrvTimer.h>

#include <stdio.h>

//---------------------------------Global Definitions-------------------------------//
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

// #if !(LINEAR)
// static u8 __attribute__((section(".ddr_direct.data"), aligned (16))) 
// shave_execution_flag[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};    //TODO when parallelization techniques
// #endif

extern u32 jumpTable;
u32 jumpTableAddr = (u32)&jumpTable;

//--------------Shave Entrypoints for execution--------------------//
#if CONVOLUTION

extern u32 CNN0_shave_conv;
extern u32 CNN1_shave_conv;
extern u32 CNN2_shave_conv;
extern u32 CNN3_shave_conv;
extern u32 CNN4_shave_conv;
extern u32 CNN5_shave_conv;
extern u32 CNN6_shave_conv;
extern u32 CNN7_shave_conv;
extern u32 CNN8_shave_conv;
extern u32 CNN9_shave_conv;
extern u32 CNN10_shave_conv;
extern u32 CNN11_shave_conv;

extern u32 CNN0_shave_im2col;
extern u32 CNN1_shave_im2col;
extern u32 CNN2_shave_im2col;
extern u32 CNN3_shave_im2col;
extern u32 CNN4_shave_im2col;
extern u32 CNN5_shave_im2col;
extern u32 CNN6_shave_im2col;
extern u32 CNN7_shave_im2col;
extern u32 CNN8_shave_im2col;
extern u32 CNN9_shave_im2col;
extern u32 CNN10_shave_im2col;
extern u32 CNN11_shave_im2col;

#endif

#if POOLING

extern u32 CNN0_shave_pool;
extern u32 CNN1_shave_pool;
extern u32 CNN2_shave_pool;
extern u32 CNN3_shave_pool;
extern u32 CNN4_shave_pool;
extern u32 CNN5_shave_pool;
extern u32 CNN6_shave_pool;
extern u32 CNN7_shave_pool;
extern u32 CNN8_shave_pool;
extern u32 CNN9_shave_pool;
extern u32 CNN10_shave_pool;
extern u32 CNN11_shave_pool;

#endif

#if INNERPRODUCT

extern u32 CNN0_shave_fc;
extern u32 CNN1_shave_fc;
extern u32 CNN2_shave_fc;
extern u32 CNN3_shave_fc;
extern u32 CNN4_shave_fc;
extern u32 CNN5_shave_fc;
extern u32 CNN6_shave_fc;
extern u32 CNN7_shave_fc;
extern u32 CNN8_shave_fc;
extern u32 CNN9_shave_fc;
extern u32 CNN10_shave_fc;
extern u32 CNN11_shave_fc;

#endif

#if LRN

extern u32 CNN0_shave_lrn;
extern u32 CNN1_shave_lrn;
extern u32 CNN2_shave_lrn;
extern u32 CNN3_shave_lrn;
extern u32 CNN4_shave_lrn;
extern u32 CNN5_shave_lrn;
extern u32 CNN6_shave_lrn;
extern u32 CNN7_shave_lrn;
extern u32 CNN8_shave_lrn;
extern u32 CNN9_shave_lrn;
extern u32 CNN10_shave_lrn;
extern u32 CNN11_shave_lrn;

#endif

//----------------------------Entrypoints Tables---------------------//
#if CONVOLUTION
static u32 startShave_conv[] =
{
    (u32) &CNN0_shave_conv,
    (u32) &CNN1_shave_conv,
    (u32) &CNN2_shave_conv,
    (u32) &CNN3_shave_conv,
    (u32) &CNN4_shave_conv,
    (u32) &CNN5_shave_conv,
    (u32) &CNN6_shave_conv,
    (u32) &CNN7_shave_conv,
    (u32) &CNN8_shave_conv,
    (u32) &CNN9_shave_conv,
    (u32) &CNN10_shave_conv,
    (u32) &CNN11_shave_conv,
};

static u32 startShave_im2col[] =
{
    (u32) &CNN0_shave_im2col,
    (u32) &CNN1_shave_im2col,
    (u32) &CNN2_shave_im2col,
    (u32) &CNN3_shave_im2col,
    (u32) &CNN4_shave_im2col,
    (u32) &CNN5_shave_im2col,
    (u32) &CNN6_shave_im2col,
    (u32) &CNN7_shave_im2col,
    (u32) &CNN8_shave_im2col,
    (u32) &CNN9_shave_im2col,
    (u32) &CNN10_shave_im2col,
    (u32) &CNN11_shave_im2col,
};

#endif

#if POOLING
static u32 startShave_pool[] =
{
    (u32) &CNN0_shave_pool,
    (u32) &CNN1_shave_pool,
    (u32) &CNN2_shave_pool,
    (u32) &CNN3_shave_pool,
    (u32) &CNN4_shave_pool,
    (u32) &CNN5_shave_pool,
    (u32) &CNN6_shave_pool,
    (u32) &CNN7_shave_pool,
    (u32) &CNN8_shave_pool,
    (u32) &CNN9_shave_pool,
    (u32) &CNN10_shave_pool,
    (u32) &CNN11_shave_pool,
};

#endif

#if INNERPRODUCT
static u32 startShave_InnerProduct[] =
{
    (u32) &CNN0_shave_fc,
    (u32) &CNN1_shave_fc,
    (u32) &CNN2_shave_fc,
    (u32) &CNN3_shave_fc,
    (u32) &CNN4_shave_fc,
    (u32) &CNN5_shave_fc,
    (u32) &CNN6_shave_fc,
    (u32) &CNN7_shave_fc,
    (u32) &CNN8_shave_fc,
    (u32) &CNN9_shave_fc,
    (u32) &CNN10_shave_fc,
    (u32) &CNN11_shave_fc,
};

#endif

#if LRN
static u32 startShave_LRN[] =
{
    (u32) &CNN0_shave_lrn,
    (u32) &CNN1_shave_lrn,
    (u32) &CNN2_shave_lrn,
    (u32) &CNN3_shave_lrn,
    (u32) &CNN4_shave_lrn,
    (u32) &CNN5_shave_lrn,
    (u32) &CNN6_shave_lrn,
    (u32) &CNN7_shave_lrn,
    (u32) &CNN8_shave_lrn,
    (u32) &CNN9_shave_lrn,
    (u32) &CNN10_shave_lrn,
    (u32) &CNN11_shave_lrn,
};

#endif

//-------------------------------Layer Execution Implementations-------------------------//
// #if CONVOLUTION
// u64 Convolution::execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
//                             u16 &bottom_input_height, u16 &bottom_input_width){

//     u8 lines = 1;

//     if ((bottom_input_height * bottom_input_width) >= 224 * 224){
//         lines = 4;
//     }
//     else if ((bottom_input_height * bottom_input_width) >= 120 * 120){
//         lines = 2;
//     }

//     for (u8 i = 0; i < lines; i++){

//         convolution_object_table[i] = &(Convolution_object[i]);

//         convolution_object_table[i]->input = bottom_output_buffer;
//         convolution_object_table[i]->output = this->output_buffer;
//         convolution_object_table[i]->conv_weights = (u8*)((u32)(this->weight_pointer) & (u32)0x8fffffff);
//         convolution_object_table[i]->conv_biases = (u8*)((u32)(this->bias_pointer) & (u32)0x8fffffff);

//         convolution_object_table[i]->c_group = this->group;
//         convolution_object_table[i]->channels = bottom_channels / this->group;
//         convolution_object_table[i]->output_channel_offset = this->input_height * this->input_width;
//         convolution_object_table[i]->kernel_h = this->kernel_size;
//         convolution_object_table[i]->kernel_w = this->kernel_size;
//         convolution_object_table[i]->with_relu = this->ReLU_flag;
//         convolution_object_table[i]->conv_weights_channel_offset = this->kernel_size * this->kernel_size;
//         convolution_object_table[i]->in_stride = this->stride;

//         convolution_object_table[i]->input_channel_offset = bottom_input_height * bottom_input_width;
//         convolution_object_table[i]->conv_weights_offset = this->kernel_size * this->kernel_size * (bottom_channels/this->group);

//         convolution_object_table[i]->splits = lines;
//         convolution_object_table[i]->maps = channels;

//         convolution_object_table[i]->inputBPP = 2;
//         convolution_object_table[i]->outputBPP = 2;
//         convolution_object_table[i]->kernelBPP = 2;
//         convolution_object_table[i]->coalescing_num = 6;

//         while (convolution_object_table[i]->coalescing_num * ((this->input_height * this->input_width)/lines) * 2 < 20000){
//                 convolution_object_table[i]->coalescing_num ++;                
//                 if((convolution_object_table[i]->coalescing_num * ((this->input_height * this->input_width)/lines) * 2 > 20000)){
//                     break;
//                 }
//         }

//         convolution_object_table[i]->ddr_function = ddr_function;

//         conv_prepare_dma((convolution_object_table[i]), bottom_input_height, bottom_input_width, lines, i, this->kernel_size, this->kernel_size, this->pad, this->pad, this->stride, this->stride, 1);
//     }

//     tyTimeStamp clock_ticks;
//     u64 cpu_cycles = 0;
//     s32 sc;
    
//     u8 first_shave = 0;
//     u8 last_shave = shaves_used - 1;
//     u8 current_shave = 0;

//     u16 number_of_outputs_per_shave = channels / shaves_used;
//     u16 remained_outputs = channels % shaves_used;

//     u16 first_map = 0;
//     u16 last_map = 0;

//     DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));
    
//     sc = DrvTimerStartTicksCount(&clock_ticks);
//     assert(!sc);

//     for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

//         last_map += number_of_outputs_per_shave;
//         if(remained_outputs > 0){
//             last_map++;
//             remained_outputs--;
//         }
//         swcResetShave(first_shave + shave_index);
//         swcSetAbsoluteDefaultStack(first_shave + shave_index);

//         swcStartShaveCC(first_shave + shave_index, 
//                         (u32) startShave_conv[shave_index + first_shave],
//                         "iiii", 
//                         convolution_object_table, first_map, last_map, jumpTableAddr);
//         first_map = last_map;
//         current_shave++;
//         if(last_map > channels){
//             break;
//         }
//     }

//     for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
//         swcWaitShave(first_shave + shave_index);
//     }

//     for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
//         DrvSvutStop(first_shave + shave_index);
//     }

//     DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

//     sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
//     assert(!sc);

//     return cpu_cycles;
// }
// #endif

#if CONVOLUTION
u64 Convolution::execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
                            u16 &bottom_input_height, u16 &bottom_input_width){

    if (ddr_function<=23){
        u8 lines = 1;
        
        if ((bottom_input_height * bottom_input_width) >= 224 * 224){
            lines = 4;
        }
        else if ((bottom_input_height * bottom_input_width) >= 120 * 120){
            lines = 2;
        }

        for (u8 i = 0; i < lines; i++){

            convolution_object_table[i] = &(Convolution_object[i]);

            convolution_object_table[i]->input = bottom_output_buffer;
            convolution_object_table[i]->output = this->output_buffer;
            convolution_object_table[i]->conv_weights = (u8*)((u32)(this->weight_pointer) & (u32)0x8fffffff);
            convolution_object_table[i]->conv_biases = (u8*)((u32)(this->bias_pointer) & (u32)0x8fffffff);

            convolution_object_table[i]->c_group = this->group;
            convolution_object_table[i]->channels = bottom_channels / this->group;
            convolution_object_table[i]->output_channel_offset = this->input_height * this->input_width;
            convolution_object_table[i]->kernel_h = this->kernel_size;
            convolution_object_table[i]->kernel_w = this->kernel_size;
            convolution_object_table[i]->with_relu = this->ReLU_flag;
            convolution_object_table[i]->conv_weights_channel_offset = this->kernel_size * this->kernel_size;
            convolution_object_table[i]->in_stride = this->stride;

            convolution_object_table[i]->input_channel_offset = bottom_input_height * bottom_input_width;
            convolution_object_table[i]->conv_weights_offset = this->kernel_size * this->kernel_size * (bottom_channels/this->group);

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
            
            convolution_object_table[i]->ddr_function = ddr_function;

           
            conv_prepare_dma((convolution_object_table[i]), bottom_input_height, bottom_input_width, lines, i, this->kernel_size, this->kernel_size, this->pad, this->pad, this->stride, this->stride, 1);
        }

       
        tyTimeStamp clock_ticks;
        u64 cpu_cycles = 0;
        s32 sc;
        
        u8 first_shave = 0;
        u8 last_shave = shaves_used - 1;
        u8 current_shave = 0;

        u16 number_of_outputs_per_shave = channels / shaves_used;
        u16 remained_outputs = channels % shaves_used;

        u16 first_map = 0;
        u16 last_map = 0;

        DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));
        
        sc = DrvTimerStartTicksCount(&clock_ticks);
        assert(!sc);

        for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

            last_map += number_of_outputs_per_shave;
            if(remained_outputs > 0){
                last_map++;
                remained_outputs--;
            }
            swcResetShave(first_shave + shave_index);
            swcSetAbsoluteDefaultStack(first_shave + shave_index);

            swcStartShaveCC(first_shave + shave_index, 
                            (u32) startShave_conv[shave_index + first_shave],
                            "iiii", 
                            convolution_object_table, first_map, last_map, jumpTableAddr);
            first_map = last_map;
            current_shave++;
            if(last_map > channels){
                break;
            }
        }

        for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
            swcWaitShave(first_shave + shave_index);
        }

        for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
            DrvSvutStop(first_shave + shave_index);
        }

        DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

        sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
        assert(!sc);

        return cpu_cycles;
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
                
        //εξασφαλιζω οτι δεν θα καθεται καποιος shave 
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

            im2col_object_table[i]->offset = ((im2col_object_table[0]->in_row_width)/(u32)tiles)*im2col_object_table[0]->in_col_height*im2col_object_table[i]->inputBPP;
            
        }

        if (this->pad != 0){
        	im2col_object_table[0]->input_column = (u8*)((u32)NewMapWithPad(bottom_output_buffer, bottom_input_width, bottom_input_height, this->kernel_size, this->stride, this->pad, bottom_channels) & (u32)0x8fffffff);
        }
        else{
        	im2col_object_table[0]->input_column = (u8*)((u32)InputstoColumns(bottom_output_buffer, bottom_input_width, bottom_input_height, this->kernel_size, this->stride, this->pad, bottom_channels) & (u32)0x8fffffff);
        }

        u16 first_shave = 0;
        u16 last_shave = shaves_used + first_shave - 1;
        u16 current_shave = 0;

        u16 number_of_outputs_per_shave =  (im2col_object_table[0]->in_row_width/tiles)/ shaves_used;
        u16 remained_outputs = (im2col_object_table[0]->in_row_width/tiles) % shaves_used;

        u16 first_map = 0;
        u16 last_map = 0;

        tyTimeStamp clock_ticks;
        u64 cpu_cycles = 0;
        s32 sc;

        DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

        sc = DrvTimerStartTicksCount(&clock_ticks);
        assert(!sc);

        for(int shave_index = 0; shave_index < shaves_used; shave_index++)
        {
            last_map += number_of_outputs_per_shave;
            if(remained_outputs > 0)
            {
                last_map++;
                remained_outputs--;
            }
            
            swcResetShave(first_shave + shave_index);
            swcSetAbsoluteDefaultStack(first_shave + shave_index);
        
            swcStartShaveCC(first_shave + shave_index, 
                        (u32) startShave_im2col[shave_index + first_shave],
                        "iiii", 
                        im2col_object_table, first_map, last_map, jumpTableAddr);
            first_map = last_map;
            current_shave++;
            if(last_map > (im2col_object_table[0]->in_row_width/tiles))
                break;
        }

        for (u8 shave_index = 0; shave_index < current_shave; shave_index++)
            swcWaitShave(first_shave + shave_index);

        for (u8 shave_index = 0; shave_index < current_shave; shave_index++)
            DrvSvutStop(first_shave + shave_index);

        DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

        sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
        assert(!sc);

        return cpu_cycles;

    }     
}
#endif


#if POOLING
u64 Pooling::execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
                            u16 &bottom_input_height, u16 &bottom_input_width){

    u8 lines = 1;

    if ((bottom_input_height * bottom_input_width) >= 224 * 224){
        lines = 4;
    }
    else if ((bottom_input_height * bottom_input_width) >= 120 * 120){
        lines = 2;
    }

    for (u8 i = 0; i < lines; i++){

        pooling_object_table[i] = &(Pooling_object[i]);
        
        pooling_object_table[i]->input = bottom_output_buffer;
        pooling_object_table[i]->output = this->output_buffer;

        pooling_object_table[i]->output_channel_offset = this->input_height * this->input_width;
        pooling_object_table[i]->line_width = this->channels;
        pooling_object_table[i]->kernel_h = this->kernel_size;
        pooling_object_table[i]->type = this->pooling_method;

        pooling_object_table[i]->input_channel_offset = bottom_input_height * bottom_input_width;
        pooling_object_table[i]->channels = bottom_channels; 

        pooling_object_table[i]->inputBPP = 2;
        pooling_object_table[i]->outputBPP = 2;
        pooling_object_table[i]->in_buffers_num = 2;
        pooling_object_table[i]->splits = lines;

        pooling_object_table[i]->ddr_function = ddr_function;    
        pool_prepare_dma(pooling_object_table[i], bottom_input_height, bottom_input_width, lines, i, this->kernel_size, this->kernel_size, this->pad, this->pad, this->stride, this->stride, 1);
    }  

    u8 first_shave = 0;
    u8 last_shave = shaves_used + first_shave - 1;
    u8 current_shave = 0;

    u16 number_of_outputs_per_shave = channels / shaves_used;
    u16 remained_outputs = channels % shaves_used;

    u16 first_map = 0;
    u16 last_map = 0;

    tyTimeStamp clock_ticks;
    u64 cpu_cycles = 0;
    s32 sc;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    sc = DrvTimerStartTicksCount(&clock_ticks);
    assert(!sc);

    for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

        last_map += number_of_outputs_per_shave;
        if(remained_outputs > 0){
            last_map++;
            remained_outputs--;
        }
        
        swcResetShave(first_shave + shave_index);
        swcSetAbsoluteDefaultStack(first_shave + shave_index);
        
        swcStartShaveCC(first_shave + shave_index, 
                        (u32) startShave_pool[shave_index + first_shave],
                        "iiii", 
                        pooling_object_table, first_map, last_map, jumpTableAddr);

        first_map = last_map;
        current_shave++;
        if(last_map > channels)
            break;
    }
    for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
        swcWaitShave(first_shave + shave_index);
    }

    for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
        DrvSvutStop(first_shave + shave_index);
    }

    DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

    sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
    assert(!sc);
    
    return cpu_cycles;
}
#endif

#if INNERPRODUCT
u64 InnerProduct::execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
                            u16 &bottom_input_height, u16 &bottom_input_width){

	fc_info *execute_InnerProduct = &InnerProduct_object;
	
	execute_InnerProduct->vector = bottom_output_buffer;
	execute_InnerProduct->output = this->output_buffer;
	execute_InnerProduct->bias = (u8*)((u32)(this->bias_pointer) & 0x8fffffff);
	execute_InnerProduct->weightLines = (u8*)((u32)(this->weight_pointer) & 0x8fffffff);

	execute_InnerProduct->linesNo = this->input_width;
	execute_InnerProduct->with_relu = this->ReLU_flag;

	execute_InnerProduct->inputWidth = bottom_input_width * bottom_input_height * bottom_channels;
	execute_InnerProduct->weightLines_offset = bottom_input_width * bottom_input_height * bottom_channels;

	execute_InnerProduct->inputBPP = 2;
	execute_InnerProduct->outputBPP = 2;

    shaves_used = (input_width < shaves_used ? input_width : shaves_used);

    u8 first_shave = 0;
    u8 last_shave = shaves_used + first_shave - 1;
    u8 current_shave = 0;

    u16 number_of_outputs_per_shave = input_width / shaves_used;
    u16 remained_outputs = input_width % shaves_used;

    u16 first_map = 0;
    u16 last_map = 0;

	tyTimeStamp clock_ticks;
	u64 cpu_cycles = 0;
    s32 sc;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    sc = DrvTimerStartTicksCount(&clock_ticks);
    assert(!sc);

    for(u8 shave_index = 0; shave_index < shaves_used; shave_index++)
    {
        last_map += number_of_outputs_per_shave;
        if(remained_outputs > 0)
        {
            last_map++;
            remained_outputs--;
        }
        
        swcResetShave(first_shave + shave_index);
        swcSetAbsoluteDefaultStack(first_shave + shave_index);
        
        swcStartShaveCC(first_shave + shave_index, 
                        (u32) startShave_InnerProduct[shave_index + first_shave],
                        "iiii", 
                        execute_InnerProduct, first_map, last_map, jumpTableAddr);
        first_map = last_map;
        current_shave++;
        if(last_map > input_width)
            break;
    }

    for (u8 shave_index = 0; shave_index < current_shave; shave_index++)
        swcWaitShave(first_shave + shave_index);

    for (u8 shave_index = 0; shave_index < current_shave; shave_index++)
        DrvSvutStop(first_shave + shave_index);

    DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

	sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
    assert(!sc);

	return cpu_cycles;
}
#endif

#if LRN
u64 Lrn::execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
                            u16 &bottom_input_height, u16 &bottom_input_width){

    lrn_info *execute_lrn = &LRN_object;

    this->channels = bottom_channels;
    this->input_height = bottom_input_height;
    this->input_width = bottom_input_width;

    execute_lrn->input = bottom_output_buffer;
    execute_lrn->output = this->output_buffer;
    execute_lrn->image_offset = bottom_input_width * bottom_input_height;

    execute_lrn->BPP = 2;

    execute_lrn->channels = this->channels;
    execute_lrn->ddr_function = this->ddr_function;

    execute_lrn->local_ratio = this->local_size;
    execute_lrn->alpha = this->alpha;
    execute_lrn->beta = this->beta;

    u8 first_shave = 0;
    u8 last_shave = shaves_used + first_shave - 1;
    u8 current_shave = 0;

    u32 pixels_per_shave = (this->input_width * this->input_height) / (shaves_used);
    u16 remaining_pixels = (this->input_width * this->input_height) % (shaves_used);

    u32 first_pixel = 0;
    u32 last_pixel = 0;

	tyTimeStamp clock_ticks;
	u64 cpu_cycles = 0;
    s32 sc;

    DrvCprTurnOnShaveMask((1 << (last_shave + 1)) - (1 << first_shave));

    sc = DrvTimerStartTicksCount(&clock_ticks);
    assert(!sc);

    for(u8 shave_index = 0; shave_index < shaves_used; shave_index++){

        last_pixel += pixels_per_shave;
        if(remaining_pixels){
            last_pixel++;
            remaining_pixels--;
        }
        swcResetShave(first_shave + shave_index);
        swcSetAbsoluteDefaultStack(first_shave + shave_index);
        swcStartShaveCC(first_shave + shave_index, (u32) startShave_LRN[shave_index + first_shave], "iiii", execute_lrn, first_pixel, last_pixel, jumpTableAddr);
        first_pixel = last_pixel;
        current_shave++;
    }

    for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
        swcWaitShave(first_shave + shave_index);
    }
    for (u8 shave_index = 0; shave_index < current_shave; shave_index++){
        DrvSvutStop(first_shave + shave_index);
    }
    DrvCprTurnOffShaveMask((1 << current_shave) - (1 << first_shave));    

	sc = DrvTimerGetElapsedTicks(&clock_ticks, &cpu_cycles);
    assert(!sc);

	return cpu_cycles;
}
#endif
