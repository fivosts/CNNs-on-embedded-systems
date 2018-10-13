// Includes
// ----------------------------------------------------------------------------
#include <mv_types.h>
#include <stdio.h>
#include "Fp16Convert.h"
#include "network_manager.h"
#include "network_defines.h"
#include "caffe_layers.h"
#include <DrvTimer.h>
#include <VcsHooksApi.h>

#ifdef PROFILE
#include <VcsHooksApi.h>
#include <DrvLeon.h>
#include <brdMv0198.h>
#include <rtems.h>
#endif

// Global Definitions
// ----------------------------------------------------------------------------
#if (defined(PROFILE) || defined(DUAL_CPU))

    extern u32 jumpTable;
    u32 lrt_jumpTableAddr = (u32)&jumpTable;

    extern u32 *lrt_start;

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

    u32 lrt_RTCNN0_shave_conv = (u32)&CNN0_shave_conv;
    u32 lrt_RTCNN1_shave_conv = (u32)&CNN1_shave_conv;
    u32 lrt_RTCNN2_shave_conv = (u32)&CNN2_shave_conv;
    u32 lrt_RTCNN3_shave_conv = (u32)&CNN3_shave_conv;
    u32 lrt_RTCNN4_shave_conv = (u32)&CNN4_shave_conv;
    u32 lrt_RTCNN5_shave_conv = (u32)&CNN5_shave_conv;
    u32 lrt_RTCNN6_shave_conv = (u32)&CNN6_shave_conv;
    u32 lrt_RTCNN7_shave_conv = (u32)&CNN7_shave_conv;
    u32 lrt_RTCNN8_shave_conv = (u32)&CNN8_shave_conv;
    u32 lrt_RTCNN9_shave_conv = (u32)&CNN9_shave_conv;
    u32 lrt_RTCNN10_shave_conv = (u32)&CNN10_shave_conv;
    u32 lrt_RTCNN11_shave_conv = (u32)&CNN11_shave_conv;
    
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

    u32 lrt_RTCNN0_shave_im2col = (u32)&CNN0_shave_im2col;
    u32 lrt_RTCNN1_shave_im2col = (u32)&CNN1_shave_im2col;
    u32 lrt_RTCNN2_shave_im2col = (u32)&CNN2_shave_im2col;
    u32 lrt_RTCNN3_shave_im2col = (u32)&CNN3_shave_im2col;
    u32 lrt_RTCNN4_shave_im2col = (u32)&CNN4_shave_im2col;
    u32 lrt_RTCNN5_shave_im2col = (u32)&CNN5_shave_im2col;
    u32 lrt_RTCNN6_shave_im2col = (u32)&CNN6_shave_im2col;
    u32 lrt_RTCNN7_shave_im2col = (u32)&CNN7_shave_im2col;
    u32 lrt_RTCNN8_shave_im2col = (u32)&CNN8_shave_im2col;
    u32 lrt_RTCNN9_shave_im2col = (u32)&CNN9_shave_im2col;
    u32 lrt_RTCNN10_shave_im2col = (u32)&CNN10_shave_im2col;
    u32 lrt_RTCNN11_shave_im2col = (u32)&CNN11_shave_im2col;
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

    u32 lrt_RTCNN0_shave_pool = (u32)&CNN0_shave_pool;
    u32 lrt_RTCNN1_shave_pool = (u32)&CNN1_shave_pool;
    u32 lrt_RTCNN2_shave_pool = (u32)&CNN2_shave_pool;
    u32 lrt_RTCNN3_shave_pool = (u32)&CNN3_shave_pool;
    u32 lrt_RTCNN4_shave_pool = (u32)&CNN4_shave_pool;
    u32 lrt_RTCNN5_shave_pool = (u32)&CNN5_shave_pool;
    u32 lrt_RTCNN6_shave_pool = (u32)&CNN6_shave_pool;
    u32 lrt_RTCNN7_shave_pool = (u32)&CNN7_shave_pool;
    u32 lrt_RTCNN8_shave_pool = (u32)&CNN8_shave_pool;
    u32 lrt_RTCNN9_shave_pool = (u32)&CNN9_shave_pool;
    u32 lrt_RTCNN10_shave_pool = (u32)&CNN10_shave_pool;
    u32 lrt_RTCNN11_shave_pool = (u32)&CNN11_shave_pool;
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

    u32 lrt_RTCNN0_shave_fc = (u32)&CNN0_shave_fc;
    u32 lrt_RTCNN1_shave_fc = (u32)&CNN1_shave_fc;
    u32 lrt_RTCNN2_shave_fc = (u32)&CNN2_shave_fc;
    u32 lrt_RTCNN3_shave_fc = (u32)&CNN3_shave_fc;
    u32 lrt_RTCNN4_shave_fc = (u32)&CNN4_shave_fc;
    u32 lrt_RTCNN5_shave_fc = (u32)&CNN5_shave_fc;
    u32 lrt_RTCNN6_shave_fc = (u32)&CNN6_shave_fc;
    u32 lrt_RTCNN7_shave_fc = (u32)&CNN7_shave_fc;
    u32 lrt_RTCNN8_shave_fc = (u32)&CNN8_shave_fc;
    u32 lrt_RTCNN9_shave_fc = (u32)&CNN9_shave_fc;
    u32 lrt_RTCNN10_shave_fc = (u32)&CNN10_shave_fc;
    u32 lrt_RTCNN11_shave_fc = (u32)&CNN11_shave_fc;
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

    u32 lrt_RTCNN0_shave_lrn = (u32)&CNN0_shave_lrn;
    u32 lrt_RTCNN1_shave_lrn = (u32)&CNN1_shave_lrn;
    u32 lrt_RTCNN2_shave_lrn = (u32)&CNN2_shave_lrn;
    u32 lrt_RTCNN3_shave_lrn = (u32)&CNN3_shave_lrn;
    u32 lrt_RTCNN4_shave_lrn = (u32)&CNN4_shave_lrn;
    u32 lrt_RTCNN5_shave_lrn = (u32)&CNN5_shave_lrn;
    u32 lrt_RTCNN6_shave_lrn = (u32)&CNN6_shave_lrn;
    u32 lrt_RTCNN7_shave_lrn = (u32)&CNN7_shave_lrn;
    u32 lrt_RTCNN8_shave_lrn = (u32)&CNN8_shave_lrn;
    u32 lrt_RTCNN9_shave_lrn = (u32)&CNN9_shave_lrn;
    u32 lrt_RTCNN10_shave_lrn = (u32)&CNN10_shave_lrn;
    u32 lrt_RTCNN11_shave_lrn = (u32)&CNN11_shave_lrn;
    #endif
#endif


#ifdef PROFILE
    volatile double total_energy;
    static u32 iterator;

    char __attribute__((section(".ddr_direct.data"))) profile_output_buffer[2*290 * 84 + 144]; //TODO size

    extern I2CM_Device *lrt_i2c2Bus;

    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_sampling_semaphore;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_setting_semaphore;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_object_type;

    extern u8 __attribute__((section(".ddr_direct.data"))) *lrt_output_buffer;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_shaves_used;

    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_input_height;
    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_input_width;
    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_channels;

    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_ddr_function;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_kernel_size;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_stride;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_pad;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_group;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_ReLU_flag;
    extern u8 __attribute__((section(".ddr_direct.data"))) *lrt_weight_pointer;
    extern u8 __attribute__((section(".ddr_direct.data"))) *lrt_bias_pointer;

    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_local_size;
    extern fp16 __attribute__((section(".ddr_direct.data"))) lrt_alpha;
    extern fp16 __attribute__((section(".ddr_direct.data"))) lrt_beta;

    extern pooling_type __attribute__((section(".ddr_direct.data"))) lrt_pooling_method;

    extern u8 __attribute__((section(".ddr_direct.data"))) *lrt_bottom_output_buffer;
    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_bottom_channels;
    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_bottom_input_height;
    extern u16 __attribute__((section(".ddr_direct.data"))) lrt_bottom_input_width;

#elif defined(DUAL_CPU)
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_execution_semaphore_lrt;
    extern u8 __attribute__((section(".ddr_direct.data"))) lrt_execution_semaphore_los;
    extern u64 __attribute__((section(".ddr_direct.data"))) lrt_round_execution_cycles;

#endif

char __attribute__((section(".ddr_direct.data"))) net_output[10000];


// Execution implementation
// ----------------------------------------------------------------------------
void Network_Manager::execute(){

    #ifndef DUAL_CPU
        #if LINEAR
            for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){
                network_cycles += (*layer)->execute((*(layer - 1))->output_buffer, (*(layer - 1))->channels, (*(layer - 1))->input_height, (*(layer - 1))->input_width);
            }
        #else
            u16 test_iter = 0;
            u64 temp = 0;
            for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){
                // printf("\nlayer: %u\n", test_iter);
                test_iter++;
                temp = (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                        (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_width);
                network_cycles += temp;
            printf("layer execution time: %f ms\n", DrvTimerTicksToMs(temp));
            }
        #endif
    #else
        u64 round_cycles = 0;
        // u64 round_time = 0;
        u16 layer_flag = 0;
        for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){

            layer_flag++;
            lrt_execution_semaphore_los = (*layer)->event_handler;
            switch ((*layer)->event_handler){

            	case 0:
                     network_cycles += (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                    (*(network_map.begin() + (*layer)->bottom_node))->input_width);                                         	
                    break;               
                case 1:
                    round_cycles += (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                    (*(network_map.begin() + (*layer)->bottom_node))->input_width); 
                    // printf("\nFinished first layer\n");              
                    break;
                case 2:
                    // printf("Before : layer %u LOS: %f  LRT: %f ms\n", layer_flag, DrvTimerTicksToMs(round_cycles), DrvTimerTicksToMs(lrt_round_execution_cycles));
                    round_cycles += (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                    (*(network_map.begin() + (*layer)->bottom_node))->input_width);
                    // printf("\nBefore stalling\n");
                    while(lrt_execution_semaphore_lrt != 5 && (lrt_round_execution_cycles == 0)){DrvTimerSleepMs(20);}
                    // while(round_cycles == 0){DrvTimerSleepMs(20);}
                    lrt_execution_semaphore_lrt = 0;
                    printf("round execution time: layer %u LOS: %f  LRT: %f ms\n", layer_flag, DrvTimerTicksToMs(round_cycles), DrvTimerTicksToMs(lrt_round_execution_cycles));
                    // round_time = (round_cycles > lrt_round_execution_cycles ? round_cycles : lrt_round_execution_cycles);
                    // printf("\nRound time: %f\n", DrvTimerTicksToMs(round_time));
                    network_cycles += (round_cycles > lrt_round_execution_cycles ? round_cycles : lrt_round_execution_cycles);
                    round_cycles = 0;
                    lrt_round_execution_cycles = 0;
                    lrt_execution_semaphore_los = 5;
                    DrvTimerSleepMs(20);
                    break;
                case 3:
                    // printf("Alone before : layer %u LOS: %f  LRT: %f ms\n", layer_flag, DrvTimerTicksToMs(round_cycles), DrvTimerTicksToMs(lrt_round_execution_cycles));
                    round_cycles += (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                    (*(network_map.begin() + (*layer)->bottom_node))->input_width);
                    while(lrt_execution_semaphore_lrt != 5){DrvTimerSleepMs(20);}
                    // printf("\nI locked LRT! %u\n", layer_flag);
                    // round_time = (round_cycles > lrt_round_execution_cycles ? round_cycles : lrt_round_execution_cycles);
                    // printf("\nRound time: %f\n", DrvTimerTicksToMs(round_time));
                    network_cycles += (round_cycles > lrt_round_execution_cycles ? round_cycles : lrt_round_execution_cycles);
                    printf("Alone round execution time: layer %u LOS: %f  LRT: %f ms\n",  layer_flag, DrvTimerTicksToMs(round_cycles), DrvTimerTicksToMs(lrt_round_execution_cycles));
                    round_cycles = 0;
                    lrt_round_execution_cycles = 0;
                    DrvTimerSleepMs(20);
                    break;
                default:
                    network_cycles += (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                    (*(network_map.begin() + (*layer)->bottom_node))->input_width);    
                    lrt_execution_semaphore_lrt = 0;               
                    DrvTimerSleepMs(40);
                    break;
            }
        }
        // printf("\nFinished execution\n");
        while(lrt_execution_semaphore_lrt != 10){DrvTimerSleepMs(20);}
    #endif
    return;
}

void Network_Manager::network_output(){

    std::vector<Layers *>::iterator output = network_map.end() - 1;

    u16 tab_iterator = 1;
    for (u16 i = 0; i < (*output)->channels; i++){
        for (u16 j = 0; j < (*output)->input_height; j++){
            for (u16 k = 0; k < (*output)->input_width; k++){
                printf("%f\t", f16Tof32(((fp16*)&(*(*output)->output_buffer))[(*output)->input_height * (*output)->input_width * i + (*output)->input_width * j + k] & 0x0000ffff));
                if (!(tab_iterator % 4)){
                    printf("\n");
                }
                tab_iterator++;
            }
            printf("\n");
        }
     }

    printf("Network execution time: %f ms\n", DrvTimerTicksToMs(network_cycles));

    u32 length = 0;
    for (u32 it = 0; it < (*output)->channels * (*output)->input_height * (*output)->input_width; it++){
        length += sprintf(net_output + length, "%f\t", (f16Tof32(((fp16*)&(*(*output)->output_buffer))[it] & 0x0000ffff )));
        if (!((it + 1) % 4)){
            length += sprintf(net_output + length, "\n");
        }
    }

    saveMemoryToFile((u32)&net_output, length, "./network_output.log");
    return;
}

// Profile Implementation
// ----------------------------------------------------------------------------
#ifdef PROFILE
rtems_task energy_probe_thread(rtems_task_argument unused){

    (void) unused;
    int status;
    total_energy = 0;
    tyBrd198Handle power_monitor_handler;
    tyAdcResultAllRails res;

    status = Brd198Init(&power_monitor_handler, lrt_i2c2Bus, NULL);

    iterator = 0;
    if(status != 0){
         printf("Board 198 init error\n");
    }

    while(!lrt_setting_semaphore){}
    lrt_sampling_semaphore = 1;
    DrvTimerSleepMs(1000);         
    while(iterator < SAMPLES){
        Brd198SampleAllRails(&power_monitor_handler, &res);
        total_energy += res.totalMilliWatts;
        iterator++;
    }
    lrt_sampling_semaphore = 0;
    while(!DrvLeonRTHasStopped()){}

    rtems_task_delete(RTEMS_SELF);
}

void Network_Manager::shave_profile(){

    rtems_status_code status;
    rtems_id task_id;
    rtems_name task_name;
    task_name = rtems_build_name('Y', 'M', 'E', 'D');
    u8 temp = 0;

    printf("Starting execution time profiling...\n");

    #if LINEAR
    for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){
        if ((*layer)->get_layer_type() == 1){
            for (u8 conv_function_it = 0; conv_function_it < 2; conv_function_it++){   
                for (u8 shaves_used = 1; shaves_used < 13; shaves_used++){
                    (*layer)->shaves_used = shaves_used;
                    (*layer)->cpu_cycles[shaves_used - 1][conv_function_it] = (*layer)->execute((*(layer - 1))->output_buffer, (*(layer - 1))->channels, (*(layer - 1))->input_height, (*(layer - 1))->input_width);
                }
                if (conv_function_it == 0){
                    temp = (*layer)->ddr_function;
                }
                (*layer)->ddr_function = 46;
            }
            (*layer)->ddr_function = temp;
        }    
        else{
            for (u8 shaves_used = 1; shaves_used < 13; shaves_used++){
                (*layer)->shaves_used = shaves_used;
                (*layer)->cpu_cycles[shaves_used - 1][0] = (*layer)->execute((*(layer - 1))->output_buffer, (*(layer - 1))->channels, (*(layer - 1))->input_height, (*(layer - 1))->input_width);
            }
        }
       printf("Layer %u complete\n", (layer - network_map.begin()));
    }   
    #else
    for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){
        if ((*layer)->get_layer_type() == 1){
            for (u8 conv_function_it = 0; conv_function_it < 2; conv_function_it++){   
                for (u8 shaves_used = 1; shaves_used < 13; shaves_used++){
                    (*layer)->shaves_used = shaves_used;
                    (*layer)->cpu_cycles[shaves_used - 1][conv_function_it] = (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                        (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                                (*(network_map.begin() + (*layer)->bottom_node))->input_width);
                }
                if (conv_function_it == 0){
                    temp = (*layer)->ddr_function;
                }
                (*layer)->ddr_function = 46;
            }
            (*layer)->ddr_function = temp;
        }
        else{
            for (u8 shaves_used = 1; shaves_used < 13; shaves_used++){
                (*layer)->shaves_used = shaves_used;
                (*layer)->cpu_cycles[shaves_used - 1][0] = (*layer)->execute((*(network_map.begin() + (*layer)->bottom_node))->output_buffer,
                                                    (*(network_map.begin() + (*layer)->bottom_node))->channels,
                                                        (*(network_map.begin() + (*layer)->bottom_node))->input_height,
                                                            (*(network_map.begin() + (*layer)->bottom_node))->input_width);     
            }
        }
        printf("Layer %u complete\n", (layer - network_map.begin()));
    }
    #endif

    printf("Time profiling complete\n");
    printf("Starting power consumption profiling...\n");

    lrt_setting_semaphore = 0;
    for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){

        // printf ("Direct measurement begins:\n");
        lrt_object_type = (*layer)->get_layer_type();

        lrt_output_buffer = (*layer)->output_buffer;
        lrt_channels = (*layer)->channels;
        lrt_input_height = (*layer)->input_height;
        lrt_input_width = (*layer)->input_width;

        lrt_bottom_output_buffer = network_map[(*layer)->bottom_node]->output_buffer;
        lrt_bottom_channels = network_map[(*layer)->bottom_node]->channels;
        lrt_bottom_input_height = network_map[(*layer)->bottom_node]->input_height;
        lrt_bottom_input_width = network_map[(*layer)->bottom_node]->input_width;

        switch(lrt_object_type){

            #if INPUT
            case 0:{
                printf("Warning: Input Layer should be first on the network_map!\n");
                break;
            }
            #endif

            #if CONVOLUTION
            case 1:{
                 for (u8 conv_function_it = 0; conv_function_it < 2; conv_function_it++){

                     lrt_weight_pointer = (*layer)->get_weight_pointer();
                     lrt_bias_pointer = (*layer)->get_bias_pointer();
                     lrt_kernel_size = (*layer)->get_kernel_size();
                     lrt_stride = (*layer)->get_stride();
                     lrt_pad = (*layer)->get_pad();
                     lrt_group = (*layer)->get_group();
                     lrt_ReLU_flag = (*layer)->get_ReLU_flag();
                     lrt_ddr_function = (*layer)->ddr_function;
                     
                     for (u8 shaves = 0; shaves < 12; shaves++){
    
                         lrt_shaves_used = shaves + 1;
                         // lrt_sampling_semaphore = 1;
                         DrvLeonRTStartup((u32)&lrt_start);
    
                         status = rtems_task_create(task_name, 1, RTEMS_MINIMUM_STACK_SIZE * 2, RTEMS_DEFAULT_MODES, RTEMS_LOCAL, &task_id);
                         if (status != RTEMS_SUCCESSFUL){
                             printf("rtems_task_create failed with %d\n", status);
                         }
                         DrvLeonRTWaitForBoot();
                         status = rtems_task_start(task_id, energy_probe_thread, 0);
                         if (status != RTEMS_SUCCESSFUL){
                             printf("rtems_task_start failed with %d\n", status);
                         }
                         (*layer)->power_consumption[shaves][conv_function_it] = total_energy / iterator;
                     }
                     (*layer)->ddr_function = 46;
                     // printf ("Im2Col measurement begins:\n");
                 }
                break;
            }
            #endif

            #if POOLING
            case 2:{
                lrt_kernel_size = (*layer)->get_kernel_size();
                lrt_stride = (*layer)->get_stride();
                lrt_pad = (*layer)->get_pad();
                lrt_ddr_function = (*layer)->ddr_function;
                lrt_pooling_method = (*layer)->get_pooling_method();

                for (u8 shaves = 0; shaves < 12; shaves++){

                    lrt_shaves_used = shaves + 1;
                    // lrt_sampling_semaphore = 1;
                    DrvLeonRTStartup((u32)&lrt_start);

                    status = rtems_task_create(task_name, 1, RTEMS_MINIMUM_STACK_SIZE * 2, RTEMS_DEFAULT_MODES, RTEMS_LOCAL, &task_id);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_create failed with %d\n", status);
                    }
                    DrvLeonRTWaitForBoot();
                    status = rtems_task_start(task_id, energy_probe_thread, 0);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_start failed with %d\n", status);
                    }
                    (*layer)->power_consumption[shaves][0] = total_energy / iterator;
                }
                break;
            }
            #endif

            #if INNERPRODUCT
            case 3:{
                lrt_weight_pointer = (*layer)->get_weight_pointer();
                lrt_bias_pointer = (*layer)->get_bias_pointer();
                lrt_ReLU_flag = (*layer)->get_ReLU_flag();

                for (u8 shaves = 0; shaves < 12; shaves++){

                    lrt_shaves_used = shaves + 1;
                    // lrt_sampling_semaphore = 1;
                    DrvLeonRTStartup((u32)&lrt_start);

                    status = rtems_task_create(task_name, 1, RTEMS_MINIMUM_STACK_SIZE * 2, RTEMS_DEFAULT_MODES, RTEMS_LOCAL, &task_id);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_create failed with %d\n", status);
                    }
                    DrvLeonRTWaitForBoot();
                    status = rtems_task_start(task_id, energy_probe_thread, 0);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_start failed with %d\n", status);
                    }
                    (*layer)->power_consumption[shaves][0] = total_energy / iterator;
                }
                break;
            }
            #endif

            #if LRN
            case 4:{

                lrt_ddr_function = (*layer)->ddr_function;

                if ((*layer)->ddr_function == 35){
                    lrt_local_size = ((*layer))->get_local_size();
                    lrt_alpha = ((*layer))->get_alpha();
                    lrt_beta = ((*layer))->get_beta();
                }
                else if ((*layer)->ddr_function != 36){
                    printf("Warning: LRN DDR function not found!\n");
                }
                for (u8 shaves = 0; shaves < 12; shaves++){

                    lrt_shaves_used = shaves + 1;
                    // lrt_sampling_semaphore = 1;
                    DrvLeonRTStartup((u32)&lrt_start);

                    status = rtems_task_create(task_name, 1, RTEMS_MINIMUM_STACK_SIZE * 2, RTEMS_DEFAULT_MODES, RTEMS_LOCAL, &task_id);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_create failed with %d\n", status);
                    }
                    DrvLeonRTWaitForBoot();
                    status = rtems_task_start(task_id, energy_probe_thread, 0);
                    if (status != RTEMS_SUCCESSFUL){
                        printf("rtems_task_start failed with %d\n", status);
                    }
                    (*layer)->power_consumption[shaves][0] = total_energy / iterator;
                }
                break;
            }
            #endif

            default:
                break;
        }
     printf("Layer %u complete\n", (layer - network_map.begin()));
    }
    printf("Power consumption profiling complete\n");
    return;
}

void Network_Manager::profile_output(){

    int length = 0;
    u8 concat_id = 0;

    length = sprintf(profile_output_buffer, "\t\tExecution time(ms)\t\t\t\t\t\t\t\t\t\t\t\t\tAverage power consumption(mW)\nlayer id - type \\ shaves used\t\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\n");

    for (std::vector<Layers *>::iterator layer = network_map.begin() + 1; layer != network_map.end(); layer++){

        switch(((*layer))->get_layer_type()){

            case 0:
                printf("Warning: Input Layer should be first on the network_map!\n");
                break;

            case 1:{
                    for (u8 conv_function_it = 0; conv_function_it < 2; conv_function_it++){
                        if (conv_function_it == 0){
                            length += sprintf(profile_output_buffer + length, "%d\tConv-Direct\t", (layer - network_map.begin() - concat_id));
                        }
                        else{
                            length += sprintf(profile_output_buffer + length, "%d\tConv_I2C\t", (layer - network_map.begin() - concat_id));
                        }

                        for (u8 shaves = 0; shaves < 12; shaves++){
                            length += sprintf(profile_output_buffer + length, "%f\t", DrvTimerTicksToMs((*layer)->cpu_cycles[shaves][conv_function_it]));
                        }
                        length += sprintf(profile_output_buffer + length, "\t");

                        for (u8 shaves = 0; shaves < 12; shaves++){
                            length += sprintf(profile_output_buffer + length, "%f\t", (*layer)->power_consumption[shaves][conv_function_it]);
                        }
                        length += sprintf(profile_output_buffer + length, "\n");
                    }
                break;
            }

            case 2:{

                length += sprintf(profile_output_buffer + length, "%d\tPooling\t", (layer - network_map.begin() - concat_id));

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", DrvTimerTicksToMs((*layer)->cpu_cycles[shaves][0]));
                }
                length += sprintf(profile_output_buffer + length, "\t");

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", (*layer)->power_consumption[shaves][0]);
                }
                length += sprintf(profile_output_buffer + length, "\n");
                break;
            }

            case 3:{

                length += sprintf(profile_output_buffer + length, "%d\tInnerProduct\t", (layer - network_map.begin() - concat_id));

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", DrvTimerTicksToMs((*layer)->cpu_cycles[shaves][0]));
                }
                length += sprintf(profile_output_buffer + length, "\t");

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", (*layer)->power_consumption[shaves][0]);
                }
                length += sprintf(profile_output_buffer + length, "\n");
                break;
            }

            case 4:{
                length += sprintf(profile_output_buffer + length, "%d\tLRN\t", (layer - network_map.begin() - concat_id));

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", DrvTimerTicksToMs((*layer)->cpu_cycles[shaves][0]));
                }
                length += sprintf(profile_output_buffer + length, "\t");

                for (u8 shaves = 0; shaves < 12; shaves++){
                    length += sprintf(profile_output_buffer + length, "%f\t", (*layer)->power_consumption[shaves][0]);
                }
                length += sprintf(profile_output_buffer + length, "\n");
                break;
            }

            #if !(LINEAR)
            case 5:
                concat_id++;
                continue;
            #endif

            default:{
                break;
            }
        }

        // for (u8 shaves = 0; shaves < 12; shaves++){
        //     length += sprintf(profile_output_buffer + length, "%f\t", DrvTimerTicksToMs(cpu_cycles[shaves][layer - network_map.begin() - 1]));
        // }
        // length += sprintf(profile_output_buffer + length, "\t");

        // for (u8 shaves = 0; shaves < 12; shaves++){
        //     length += sprintf(profile_output_buffer + length, "%f\t", power_consumption[shaves][layer - network_map.begin() - 1]);
        // }
        // length += sprintf(profile_output_buffer + length, "\n");
    }

    saveMemoryToFile((u32)&profile_output_buffer, length, "./network_profile.csv");
    printf("Network profile written to output\n");
    return;
}
#endif
