// 1: Includes
// ----------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <mv_types.h>
#include "network_manager.h"
#include "network_defines.h"
#include <DrvTimer.h>
#include <DrvLeon.h>
extern "C"{
#include <app_config.h>
#include <rtems.h>
}

#include <chrono>
#include <iostream>
#include "weight_data.h"

// 6: Functions Implementation
// ----------------------------------------------------------------------------

extern "C" void *POSIX_Init (void *args)
{
    UNUSED(args);

    s32 sc;

    sc = InitClocksAndMemory();
    if (sc) exit(sc);

    sc = InitShaveL2C();
    if (sc) exit(sc);

    sc = ConfigShaveL2C();
    if (sc) exit(sc);

    sc = DrvTimerInit();
    if (sc) exit(sc);

    printf("System frequency: %d Mhz\n", DrvCprGetClockFreqKhz(SYS_CLK, NULL) / 1000);
    printf ("RTEMS POSIX Started\n");

    Network_Manager network_manager;
    #ifdef PROFILE
    network_manager.shave_profile();
    network_manager.profile_output();
    #else
    network_manager.execute();
    network_manager.network_output();
    #endif
    network_manager.~Network_Manager();

    printf("Program Terminated Successfully!\n");

    exit(0);
}
    // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //insert code to be measured here
    // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    // std::cout << 1000*100*time_span.count() << " miliseconds" << std::endl;