#include <stddef.h>
#include <mv_types.h>
#include <moviVectorUtils.h>

#include "utils.h"

inline int 
round_up64(int x)
{
    return ((x + 63) / 64) * 64;
}

void 
memset64_h(half *ptr, int size, half val)
{
    half8 v = {val, val, val, val, val, val, val, val};
    half8 *addr_h8 = (half8 *)ptr;

    for (int j = 0; j < size/8; j += 8) {
        addr_h8[j]   = v;
        addr_h8[j+1] = v;
        addr_h8[j+2] = v;
        addr_h8[j+3] = v;
        addr_h8[j+4] = v;
        addr_h8[j+5] = v;
        addr_h8[j+6] = v;
        addr_h8[j+7] = v;
    }
}

void 
relu64_h(half *in, half *out, int size)
{
    // +Inf in fp16 format
    int p_inf_bin = 0x00007C00;
    half p_inf = *((half *) &p_inf_bin);

    half8 v = {p_inf, p_inf, p_inf, p_inf, p_inf, p_inf, p_inf, p_inf};
    half8 *in_h8 = (half8 *)in;
    half8 *out_h8 = (half8 *)out;

    for (int j = 0; j < size/8; j += 8) {
        out_h8[j]   = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j  ], v);
        out_h8[j+1] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+1], v);
        out_h8[j+2] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+2], v);
        out_h8[j+3] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+3], v);
        out_h8[j+4] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+4], v);
        out_h8[j+5] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+5], v);
        out_h8[j+6] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+6], v);
        out_h8[j+7] = __builtin_shave_cmu_clamp0_f16_rr_half8(in_h8[j+7], v);
    }
}

void 
relu_inplace64_h(half *ptr, int size)
{
    // +Inf in fp16 format
    int p_inf_bin = 0x00007C00;
    half p_inf = *((half *) &p_inf_bin);

    half8 v = {p_inf, p_inf, p_inf, p_inf, p_inf, p_inf, p_inf, p_inf};
    half8 *addr_h8 = (half8 *)ptr;

    for (int j = 0; j < size/8; j += 8) {
        addr_h8[j]   = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j  ], v);
        addr_h8[j+1] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+1], v);
        addr_h8[j+2] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+2], v);
        addr_h8[j+3] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+3], v);
        addr_h8[j+4] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+4], v);
        addr_h8[j+5] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+5], v);
        addr_h8[j+6] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+6], v);
        addr_h8[j+7] = __builtin_shave_cmu_clamp0_f16_rr_half8(addr_h8[j+7], v);
    }
}