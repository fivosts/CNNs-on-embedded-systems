#include <rtems.h>
#include "rtems_config.h"

static void Fatal_extension(Internal_errors_Source the_source, bool is_internal, uint32_t the_error){

    switch (the_source){

	    case RTEMS_FATAL_SOURCE_EXIT:
	        if (the_error)
	            printk("Exited with error code %lu\n", the_error);
	        break;

	    case RTEMS_FATAL_SOURCE_ASSERT:
	        printk("%s : %d in %s \n", ((rtems_assert_context * )the_error)->file, ((rtems_assert_context * )the_error)->line, ((rtems_assert_context * )the_error)->function);
	        break;

	    case RTEMS_FATAL_SOURCE_EXCEPTION:
	        rtems_exception_frame_print((const rtems_exception_frame *) the_error);
	        break;

	    default:
	        printk("\nSource %d Internal %d Error %lu  0x%lX:\n", the_source, is_internal, the_error, the_error);
	        break;
    }
    return;
}