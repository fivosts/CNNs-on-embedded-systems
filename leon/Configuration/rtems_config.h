#ifndef LEON_RTEMS_CONFIG_H_
#define LEON_RTEMS_CONFIG_H_


#ifndef _RTEMS_CONFIG_H_
#define _RTEMS_CONFIG_H_

// 1: Includes
// ----------------------------------------------------------------------------
#include "app_config.h"

#if defined(__RTEMS__)

#if !defined (__CONFIG__)
#define __CONFIG__

// 2: Defines
// ----------------------------------------------------------------------------
#define CLOCKS_NONE 0

/* ask the system to generate a configuration table */
#define CONFIGURE_INIT


#ifndef RTEMS_POSIX_API
#define RTEMS_POSIX_API
#endif

#define CONFIGURE_MICROSECONDS_PER_TICK         1000    /* 1 millisecond */

#define CONFIGURE_TICKS_PER_TIMESLICE           10      /* 10 milliseconds */

#define CONFIGURE_APPLICATION_NEEDS_CONSOLE_DRIVER

#define CONFIGURE_APPLICATION_NEEDS_CLOCK_DRIVER

#define CONFIGURE_POSIX_INIT_THREAD_TABLE

#define CONFIGURE_MINIMUM_TASK_STACK_SIZE      (4*1024)

#define CONFIGURE_MAXIMUM_TASKS                 4

#define CONFIGURE_MAXIMUM_POSIX_THREADS         4

#define CONFIGURE_MAXIMUM_POSIX_MUTEXES         8

#define CONFIGURE_MAXIMUM_POSIX_KEYS            8

#define CONFIGURE_MAXIMUM_POSIX_SEMAPHORES      8
#define CONFIGURE_MAXIMUM_SEMAPHORES            4

#define CONFIGURE_MAXIMUM_POSIX_MESSAGE_QUEUES  8

#define CONFIGURE_MAXIMUM_POSIX_TIMERS          4
#define CONFIGURE_MAXIMUM_TIMERS                4

// Needed for ethernet
#define CONFIGURE_LIBIO_MAXIMUM_FILE_DESCRIPTORS 			30

#define CONFIGURE_MAXIMUM_USER_EXTENSIONS    1
#define CONFIGURE_INITIAL_EXTENSIONS         { .fatal = Fatal_extension }

// 3:  Exported Global Data (generally better to avoid)
// ----------------------------------------------------------------------------  
// 4:  Functions (non-inline)
// ----------------------------------------------------------------------------

static void Fatal_extension (
  Internal_errors_Source  the_source,
  bool                    is_internal,
  uint32_t                the_error
);
void POSIX_Init (void *args);

#include <rtems/confdefs.h>

#endif // __CONFIG__

#endif // __RTEMS__

BSP_SET_CLOCK(OSC_CLOCK_KHZ, 	// Reference oscillator used
			  APP_CLOCK_KHZ, 	// PLL0 Target Frequency
			  1, 				// Master Divider Numerator
			  1, 				// Master Divider Denominator
			  CSS_CLOCKS,  	//CSS Clocks
			  MSS_CLOCKS, 	// MSS Clocks
			  UPA_CLOCKS, 	// UPA Clocks
			  CLOCKS_NONE, 	// SIPP Clocks
			  CLOCKS_NONE 	// AUX Clocks
);

BSP_SET_L2C_CONFIG( 1, 					// Enable (1) / Disable (0)
					L2C_REPL_LRU, 		// Either L2C_REPL_LRU (default), 
										//        L2C_REPL_PSEUDO_RANDOM, 
										//        L2C_REPL_MASTER_INDEX_REP 
										//     or L2C_REPL_MASTER_INDEX_MOD
					0, 					// Cache ways
					L2C_MODE_COPY_BACK, // Either L2C_MODE_COPY_BACK 
										//     or L2C_MODE_WRITE_TROUGH
					0, 					// Number of MTRR registers to program
					NULL 				// Array of MTRR configuration
);


#endif // _RTEMS_CONFIG_H_

#endif // LEON_RTEMS_CONFIG_H_
