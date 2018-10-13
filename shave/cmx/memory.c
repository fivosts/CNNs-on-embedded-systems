#include <swcWhoAmI.h>
#include <mv_types.h>
#include <stddef.h>

#include "memory.h"
#include <ddr_functions.h>

static u8 __attribute__(( aligned (128))) mem[MEMORY_POOL];
static u32 nextAddress;
static PRINTF_PTR printf;
static int primaryShave;

void setAlignedMem(int shaveId, J_FUNCPTR_T jumpTable)
{
    nextAddress = (u32)mem;
    printf = (PRINTF_PTR) jumpTable(CM_printf);
    primaryShave = shaveId;
}

void *getAlignedMem(u32 alignment, u32 bytes)
{
    u32 residue = nextAddress % (u32)alignment;
    u32 ret = nextAddress + (residue ? ((u32)alignment - residue) : 0);
    nextAddress = ret + bytes;

    if ((nextAddress - (u32)mem) > sizeof(mem))
    	if (primaryShave == (swcWhoAmI() - PROCESS_SHAVE0))
    		printf("Error: Trying to allocate memory space that exceeds local"
    			   " memory pool capacity!\n");

    return (void *) ret;
}