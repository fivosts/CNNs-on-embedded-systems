// ///
// /// @file
// /// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
// ///            For License Warranty see: common/license.txt
// ///
// /// @brief
// ///

.version 00.87.03

.data .rodata.avgPool3x3s2hh
.align 16
avgPool3x3s2hh_asm_multiply:
// Επειδή το ave pooling χρειάζεται διόρθωση στις άκρες, αλλάζουμε πρόχειρα τη
// ρουτίνα ώστε να διαιρεί με το 1, αντί με το 9. Η διαίρεση με το 9 γίνεται στον
// κώδικα του shave.
//    .float16    0.11111111111111111
.float16    1.0

.code .text.avgPool3x3s2hh

// Πρέπει το outputWidth να είναι πολλαπλάσιο του 8

//void avgPool3x3s2hh(half** src, half** dst, u32 outputWidth)
//                          i18         i17        i16
mvcvAvgPool3x3s2hh_asm:

    lsu0.ldil i7 avgPool3x3s2hh_asm_multiply || lsu1.ldih i7 avgPool3x3s2hh_asm_multiply 
    lsu1.ldi.64 i0 i18   || lsu0.ldil i3 0x10
    lsu1.ld.32 i2 i18    || lsu0.ldil i5 ___avgPool3x3s2hhLoop
    iau.shr.u32 i4 i16 3 || lsu0.ldih i5 ___avgPool3x3s2hhLoop || lsu1.ld.16r v9 i7
    lsu0.ld.32 i17 i17   || lsu1.ldil i6 ___avgPool3x3s2hh_compensate
    lsu0.ldih i6 ___avgPool3x3s2hh_compensate
    nop 2

    lsu1.ldxvi v0 i0 i3
    lsu0.ldxvi v1 i1 i3
    lsu1.ldxvi v2 i2 i3
    lsu0.ldxvi v10 i0 i3
    lsu1.ldxvi v11 i1 i3
    lsu0.ldxvi v12 i2 i3
    lsu1.ldxvi v20 i0 i3
    lsu0.ldxvi v21 i1 i3
    lsu1.ldxvi v22 i2 i3

    nop 2

    cmu.vdilv.x16 v3 v4 v0 v10  || lsu0.ldxvi v10 i0 i3
    cmu.vdilv.x16 v5 v6 v1 v11  || lsu1.ldxvi v11 i1 i3
    cmu.vdilv.x16 v7 v8 v2 v12  || lsu0.ldxvi v12 i2 i3
    cmu.alignvec v14 v4 v20 2   || lsu1.ldxvi v20 i0 i3
    cmu.alignvec v16 v6 v21 2   || lsu0.ldxvi v21 i1 i3
    cmu.alignvec v18 v8 v22 2   || lsu1.ldxvi v22 i2 i3

    VAU.MACPZ.f16 v3  v9        || cmu.cpvv v0 v20
    VAU.MACP.f16 v4  v9         || cmu.cpvv v1 v21
    VAU.MACP.f16 v5  v9         || cmu.cpvv v2 v22
    VAU.MACP.f16 v6  v9         || iau.sub i4 i4 1
    VAU.MACP.f16 v7  v9
    VAU.MACP.f16 v8  v9
    VAU.MACP.f16 v14  v9
    VAU.MACP.f16 v16  v9
    VAU.MACPW.f16 v19 v18 v9    || cmu.cmz.i32 i4

    peu.pc1c eq || bru.bra ___avgPool3x3s2hh_compensate

    cmu.vdilv.x16 v3 v4 v0 v10  || lsu0.ldxvi v10 i0 i3 || bru.rpl i5 i4
    cmu.vdilv.x16 v5 v6 v1 v11  || lsu1.ldxvi v11 i1 i3
    cmu.vdilv.x16 v7 v8 v2 v12  || lsu0.ldxvi v12 i2 i3
    cmu.alignvec v14 v4 v20 2   || lsu1.ldxvi v20 i0 i3
    cmu.alignvec v16 v6 v21 2   || lsu0.ldxvi v21 i1 i3
    cmu.alignvec v18 v8 v22 2   || lsu1.ldxvi v22 i2 i3

    nop

    VAU.MACPZ.f16 v3  v9       
    VAU.MACP.f16 v4  v9         || cmu.cpvv v0 v20
    VAU.MACP.f16 v5  v9         || cmu.cpvv v1 v21 || lsu0.sti.64.l v19 i17
    VAU.MACP.f16 v6  v9         || cmu.cpvv v2 v22 || lsu0.sti.64.h v19 i17
  ___avgPool3x3s2hhLoop:
    VAU.MACP.f16 v7  v9        
    VAU.MACP.f16 v8  v9
    VAU.MACP.f16 v14  v9       
    VAU.MACP.f16 v16  v9
    VAU.MACPW.f16 v19 v18 v9
    nop
    sau.xor i7 i7 i1    // Αντί για nop (γιατί είναι το τέλος του βρόχου)

    nop 3

    //Compensate
___avgPool3x3s2hh_compensate:
    nop 3
    bru.jmp i30
    nop
    lsu0.sti.64.l v19 i17
    lsu0.sti.64.h v19 i17
    nop 3

.end