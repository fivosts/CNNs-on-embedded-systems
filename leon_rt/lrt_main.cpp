// 1: Includes
extern "C"{
#include "lrt_app_config.h"
// #include "DrvSvu.h"
}

#ifdef PROFILE
    #include <stdio.h>
    #include "MV0212.h"
	#include "brdGpioCfgs/brdMv0182R5GpioDefaults.h"
	#include "DrvCDCEL.h"
#elif defined(DUAL_CPU)
    #include "lrt_network.h"
    #include <vector>
    #include <iterator>
    #include "lrt_weight_data.h"
    #include "Fp16Convert.h"//TODO
#endif

#include <stdlib.h>
#include "DrvTimer.h"
#include <DrvLeon.h>
#include "lrt_caffe_layers.h"
#include <network_defines.h>
#include <ddr_functions_types.h>

#ifdef PROFILE
	#define NUM_I2C_DEVS 3

	I2CM_Device	*i2c2Bus;

	u8 __attribute__((section(".ddr_direct.data"))) object_type;

	u8 __attribute__((section(".ddr_direct.data"))) *output_buffer;
	u8 __attribute__((section(".ddr_direct.data"))) shaves_used;

	u16 __attribute__((section(".ddr_direct.data"))) input_height;
	u16 __attribute__((section(".ddr_direct.data"))) input_width;
	u16 __attribute__((section(".ddr_direct.data"))) channels;

	u8 __attribute__((section(".ddr_direct.data"))) ddr_function;
	u8 __attribute__((section(".ddr_direct.data"))) kernel_size;
	u8 __attribute__((section(".ddr_direct.data"))) stride;
	u8 __attribute__((section(".ddr_direct.data"))) pad;
	u8 __attribute__((section(".ddr_direct.data"))) group;
	u8 __attribute__((section(".ddr_direct.data"))) ReLU_flag;
	u8 __attribute__((section(".ddr_direct.data"))) *weight_pointer;
	u8 __attribute__((section(".ddr_direct.data"))) *bias_pointer;

	u8 __attribute__((section(".ddr_direct.data"))) local_size;
	fp16 __attribute__((section(".ddr_direct.data"))) alpha;
	fp16 __attribute__((section(".ddr_direct.data"))) beta;

	pooling_type __attribute__((section(".ddr_direct.data"))) pooling_method;

	u8 __attribute__((section(".ddr_direct.data"))) *bottom_output_buffer;
	u16 __attribute__((section(".ddr_direct.data"))) bottom_channels;
	u16 __attribute__((section(".ddr_direct.data"))) bottom_input_height;
	u16 __attribute__((section(".ddr_direct.data"))) bottom_input_width;

#elif defined(DUAL_CPU)
	u8 __attribute__((section(".ddr_direct.data"))) execution_semaphore_lrt;
	u8 __attribute__((section(".ddr_direct.data"))) execution_semaphore_los;
	u64 __attribute__((section(".ddr_direct.data"))) round_execution_cycles;

#endif

// 6: Functions Implementation
// ----------------------------------------------------------------------------
#ifdef PROFILE
int power_measurement(){

   	s32 boardStatus;
    int32_t rc;

    initClocksAndMemory();
    BoardI2CInfo info[NUM_I2C_DEVS];
    BoardConfigDesc config[] = {
        { BRDCONFIG_GPIO, (void *)brdMV0182R5GpioCfgDefault },
        { BRDCONFIG_END, NULL }
    };

    rc = BoardInit(config);
    if (rc!=BRDCONFIG_SUCCESS){
    	printf("Error: board initialization failed with %ld status\n", rc);
        return rc;
    }
    boardStatus = BoardInitExtPll(EXT_PLL_CFG_148_24_24MHZ);
    if (boardStatus != BRDCONFIG_SUCCESS){
    	printf("Error: board initialization failed with %ld status\n", boardStatus);
    	return -1;
    }
    rc = BoardGetI2CInfo(info, NUM_I2C_DEVS);
    if (rc!=BRDCONFIG_SUCCESS){
        printf("Error: board configuration read failed with %ld status\n", rc);
        return rc; 
    }

    i2c2Bus	= info[2].handler;
	DrvLeonRTSignalBootCompleted();

    u64 network_cycles = 0;

    switch(object_type){

        #if CONVOLUTION
        case 1:{

            lrt_Layers *convolution_energy_probe = new Convolution(output_buffer, shaves_used, 
                                                            weight_pointer, bias_pointer, channels,
                                                                input_height, input_width, bottom_output_buffer, bottom_channels, 
                                                                    bottom_input_height, bottom_input_width, kernel_size,
                                                                    stride, pad, group, ddr_function, ReLU_flag);
            network_cycles += convolution_energy_probe->execute();
            delete convolution_energy_probe;

            break;
        }
        #endif
        #if POOLING
        case 2:{
            lrt_Layers *pooling_energy_probe = new Pooling(output_buffer, shaves_used,
                                                        channels, input_height, input_width,
                                                            kernel_size, stride, pad, ddr_function, pooling_method,
                                                            	bottom_output_buffer, bottom_channels, bottom_input_height, bottom_input_width);
            
            network_cycles += pooling_energy_probe->execute();
            delete pooling_energy_probe;
            break;
        }
        #endif
        #if INNERPRODUCT
        case 3:{
            lrt_Layers *ip_energy_probe = new InnerProduct(output_buffer, shaves_used,
                                                                weight_pointer, bias_pointer, input_width, bottom_output_buffer, 
                                                                	bottom_channels, bottom_input_height, bottom_input_width, ReLU_flag);

            network_cycles += ip_energy_probe->execute();
            delete ip_energy_probe;
            break;
        }
        #endif
        #if LRN
        case 4:{
            if (ddr_function == 36){
                lrt_Layers *lrn_energy_probe = new Lrn(output_buffer, shaves_used, bottom_output_buffer,
                											bottom_channels, bottom_input_height, bottom_input_width);
	            network_cycles += lrn_energy_probe->execute();
	            delete lrn_energy_probe;
            }
            else{
                lrt_Layers *lrn_energy_probe = new Lrn(output_buffer, shaves_used,
                                                        local_size, alpha, beta, bottom_output_buffer,
                											bottom_channels, bottom_input_height, bottom_input_width);
	            network_cycles += lrn_energy_probe->execute();
	            delete lrn_energy_probe;
            }
            break;
        }
        #endif
        default:
            break;
    }
    (void)network_cycles;
	return 0;
}
#elif defined(DUAL_CPU)

int parallel_network_deployment(){

    initClocksAndMemory();
    std::vector<lrt_Layers *> lrt_subnet = lrt_create_network();
    DrvLeonRTSignalBootCompleted();

    u64 round_time = 0;
    while(execution_semaphore_los == 0){DrvTimerSleepMs(20);}
    for (std::vector<lrt_Layers *>::iterator layer = lrt_subnet.begin(); layer != lrt_subnet.end(); layer++){

        execution_semaphore_lrt = (*layer)->event_handler;
        switch((*layer)->event_handler){
                case 1:

                    round_time += (*layer)->execute();
                    round_execution_cycles = round_time;
                    round_time = 0;
                    execution_semaphore_lrt = 5;
                    while(execution_semaphore_los != 5){DrvTimerSleepMs(20);}
                    execution_semaphore_los = 0;
                    break;

                default:
                    round_time += (*layer)->execute();
                    break;
        }

    }
    execution_semaphore_lrt = 10;
    return 0;
}
#endif

int main(void){

	int rc = 0;

#ifdef PROFILE
	rc = power_measurement();
	if (rc){ exit(-1); }
#elif defined(DUAL_CPU)
	rc = parallel_network_deployment();
    if (rc){ exit (-1); }
#endif

	DrvLeonRTSignalStop();
    exit(0);
}
