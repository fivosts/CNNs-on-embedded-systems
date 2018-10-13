#ifndef __MEMORY_H__
#define __MEMORY_H__

// Includes
// ----------------------------------------------------------------------------
#include <ddr_functions.h>

// Defines
// ----------------------------------------------------------------------------
#define MEMORY_POOL 123*1024

void setAlignedMem(int shaveId, J_FUNCPTR_T jumpTable);
void *getAlignedMem(u32 alignment, u32 bytes);

#endif//__MEMORY_H__
