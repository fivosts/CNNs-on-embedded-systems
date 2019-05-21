///
/// @file
/// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief
///

.version 00.87.03

.data .data.maxPool3x3s1hh
.align 4
.code .text.maxPool3x3s1hh

//void maxPool3x3s1hh(half** src, half** dst, u32 outputWidth)
//                          i18         i17        i16
mvcvMaxPool3x3s1hh_asm:

    lsu1.ldi.64 i0 i18   || lsu0.ldil i3 0x10
    lsu1.ld.32 i2 i18    || lsu0.ldil i5 ___MaxPool3x3s1hhLoop
    iau.shr.u32 i4 i16 3 || lsu0.ldih i5 ___MaxPool3x3s1hhLoop
    vau.xor v9 v9 v9     || iau.xor i6 i6 i6
    iau.sub i4 i4 1
    lsu1.ld.32 i17 i17   || iau.incs i6 7
    iau.and i7 i6 i16
    lsu1.ldxvi v0 i0 i3
    lsu0.ldxvi v1 i1 i3
    lsu1.ldxvi v2 i2 i3
    lsu0.ldxvi v10 i0 i3
    lsu1.ldxvi v11 i1 i3
    lsu0.ldxvi v12 i2 i3

    nop 5
    cmu.alignvec v3 v0 v10 2
    cmu.alignvec v4 v0 v10 4
    cmu.alignvec v5 v1 v11 2
    cmu.alignvec v6 v1 v11 4 || lsu1.ldxvi v10 i0 i3
    cmu.alignvec v7 v2 v12 2 || lsu0.ldxvi v11 i1 i3
    cmu.alignvec v8 v2 v12 4 || lsu1.ldxvi v12 i2 i3

    cmu.max.f16 v13 v0 v1    || vau.or v0 v9 v10
    cmu.max.f16 v14 v2 v3    || vau.or v1 v9 v11
    cmu.max.f16 v15 v4 v5    || vau.or v2 v9 v12
    cmu.max.f16 v16 v6 v7
    cmu.max.f16 v17 v13 v8
    cmu.max.f16 v18 v14 v15
    cmu.max.f16 v19 v16 v17
    nop
    cmu.max.f16 v20 v18 v19
    nop

    cmu.alignvec v3 v0 v10 2 || bru.rpl i5 i4
    cmu.alignvec v4 v0 v10 4
    cmu.alignvec v5 v1 v11 2
    cmu.alignvec v6 v1 v11 4 || lsu1.ldxvi v10 i0 i3
    cmu.alignvec v7 v2 v12 2 || lsu0.ldxvi v11 i1 i3
    cmu.alignvec v8 v2 v12 4 || lsu1.ldxvi v12 i2 i3
    cmu.max.f16 v13 v0 v1    || vau.or v0 v9 v10
    cmu.max.f16 v14 v2 v3    || vau.or v1 v9 v11
    cmu.max.f16 v15 v4 v5    || vau.or v2 v9 v12
    ___MaxPool3x3s1hhLoop:
    cmu.max.f16 v16 v6 v7
    cmu.max.f16 v17 v13 v8
    cmu.max.f16 v18 v14 v15
    cmu.max.f16 v19 v16 v17
    lsu1.sti.64.l v20 i17
    cmu.max.f16 v20 v18 v19 || lsu1.sti.64.h v20 i17
    nop


    //Compensate
    cmu.cmz.i32 i7
    peu.pc1c eq || bru.jmp i30
    cmu.alignvec v3 v0 v10 2 || lsu0.sti.64.l v20 i17
    cmu.alignvec v4 v0 v10 4 || lsu0.sti.64.h v20 i17
    cmu.alignvec v5 v1 v11 2
    cmu.alignvec v6 v1 v11 4
    cmu.alignvec v7 v2 v12 2
    cmu.alignvec v8 v2 v12 4
    cmu.max.f16 v13 v0 v1
    cmu.max.f16 v14 v2 v3
    cmu.max.f16 v15 v4 v5
    cmu.max.f16 v16 v6 v7    || bru.jmp i30
    cmu.max.f16 v17 v13 v8
    cmu.max.f16 v18 v14 v15
    cmu.max.f16 v19 v16 v17
    cmu.max.f16 v20 v18 v19
    nop
    lsu1.st.64.l v20 i17

.end
