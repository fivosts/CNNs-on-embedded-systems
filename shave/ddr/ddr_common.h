#ifndef __DDR_COMMON_H__
#define __DDR_COMMON_H__

// 1: Includes
// ----------------------------------------------------------------------------
#include <swcCdma.h>

#include <stddef.h>
#include <mv_types.h>
#include <moviVectorUtils.h>

#include <ddr_functions.h>
#include "../cmx/memory.h"

// 2:  Source Specific #defines and types  (typedef, enum, struct)
// ----------------------------------------------------------------------------
typedef void (*SETALIGNEDMEM_PTR) (int shaveId, J_FUNCPTR_T jumpTable);
typedef void* (*GETALIGNEDMEM_PTR) (u32 alignment, u32 bytes);

typedef dmaRequesterId (*DMAINITREQUESTER_PTR) (int priority);

typedef dmaTransactionList* (*DMACREATETRANSACTION_PTR) (
    dmaRequesterId ReqId, 
    dmaTransactionList *NewTransaction, 
    u8* Src, 
    u8* Dst, 
    u32 ByteLength
);

typedef dmaTransactionList* (*DMACREATETRANSACTIONDSTSTRIDE_PTR) (
    dmaRequesterId ReqId, 
    dmaTransactionList *NewTransaction, 
    u8* Src, 
    u8* Dst, 
    u32 ByteLength, 
    u32 LineWidth, 
    s32 DstStride
);

typedef dmaTransactionList* (*DMACREATETRANSACTIONSRCSTRIDE_PTR) (
    dmaRequesterId ReqId, 
    dmaTransactionList *NewTransaction, 
    u8* Src, 
    u8* Dst, 
    u32 ByteLength, 
    u32 LineWidth, 
    s32 SrcStride
);

typedef dmaTransactionList* (*DMACREATETRANSACTIONFULLOPTIONS_PTR) (
    dmaRequesterId ReqId, 
    dmaTransactionList *NewTransaction, 
    u8* Src, 
    u8* Dst, 
    u32 ByteLength, 
    u32 SrcLineWidth, 
    u32 DstLineWidth, 
    s32 SrcStride, 
    s32 DstStride
);

typedef dmaTransactionList* (*DMACREATE3DTRANSACTION_PTR) (    
    dmaRequesterId ReqId, 
    dmaTransactionList *NewTransaction, 
    u8* Src, 
    u8* Dst, 
    u32 ByteLength, 
    u32 SrcLineWidth, 
    u32 DstLineWidth, 
    s32 SrcStride, 
    s32 DstStride,
    u32 NumPlanes,
    s32 SrcPlaneStride,
    s32 DstPlaneStride
);

typedef int (*DMASTARTLISTTASK_PTR) (dmaTransactionList* ListPtr);
typedef void (*DMAWAITTASK_PTR) (dmaTransactionList* ListPtr);

typedef struct {
    DMAINITREQUESTER_PTR dmaInitRequester;
    DMACREATETRANSACTION_PTR dmaCreateTransaction;
    DMACREATETRANSACTIONDSTSTRIDE_PTR dmaCreateTransactionDstStride;
    DMACREATETRANSACTIONSRCSTRIDE_PTR dmaCreateTransactionSrcStride;
    DMACREATETRANSACTIONFULLOPTIONS_PTR dmaCreateTransactionFullOptions;
    DMACREATE3DTRANSACTION_PTR dmaCreate3DTransaction;

    DMASTARTLISTTASK_PTR dmaStartListTask;
    DMAWAITTASK_PTR dmaWaitTask;
    dmaTransactionList_t *task;
    dmaTransactionList_t **ref;
} dma_context;

typedef struct {
    SETALIGNEDMEM_PTR setAlignedMem;
    GETALIGNEDMEM_PTR getAlignedMem;
} memory_context;

typedef struct {
    int shaveId;
    J_FUNCPTR_T jumpTable;
} common_context;

#endif//__DDR_COMMON_H__