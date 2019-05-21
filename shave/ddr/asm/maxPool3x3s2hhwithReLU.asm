; ///
; /// @file
; /// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
; ///            For License Warranty see: common/license.txt
; ///
; /// @brief
; ///

.version 00.50.00

.data .data.maxPool3x3s2hhwithReLU
.code .text.maxPool3x3s2hhwithReLU

; Πρέπει το outputWidth να είναι πολλαπλάσιο του 8

;void maxPool3x3s2hh(half** src, half** dst, u32 outputWidth)
;                          i18         i17        i16
mvcvMaxPool3x3s2hhwithReLU_asm:

    lsu1.ldi.64 i0 i18   || lsu0.ldil i3 0x10
    lsu1.ld.32 i2 i18    || vau.xor v9 v9 v9  || lsu0.ldil i5 ___maxPool3x3s2hhwithReLULoop
    iau.shr.u32 i4 i16 3 || lsu0.ldih i5 ___maxPool3x3s2hhwithReLULoop
    lsu0.ld.32 i17 i17   || lsu1.ldil i6 ___maxPool3x3s2hhwithReLU_compensate
    lsu0.ldih i6 ___maxPool3x3s2hhwithReLU_compensate
    nop 2

; Για τα παραπάνω
; i0 = διεύθυνση πρώτης γραμμής του input
; i1 = διεύθυνση δεύτερης γραμμής του input
; i2 = διεύθυνση τρίτης γραμμής του input
; i3 = ο αριθμός 16
; i4 = outputWidth / 8 (ακέραια διαίρεση της C)
; i5 = διεύθυνση της label ___maxPool3x3s2hhLoop
; i6 = ο αριθμός 000...00111 (binary σε χώρο 32 bit)
; i7 = outputWidth % 8
; v9 = {0,0,...,0}

    lsu1.ldxvi v0 i0 i3
    lsu0.ldxvi v1 i1 i3
    lsu1.ldxvi v2 i2 i3
    lsu0.ldxvi v10 i0 i3
    lsu1.ldxvi v11 i1 i3
    lsu0.ldxvi v12 i2 i3
    lsu1.ldxvi v20 i0 i3
    lsu0.ldxvi v21 i1 i3
    lsu1.ldxvi v22 i2 i3

; Για τα παραπάνω
; Κάθε εντολή ldxvi διαρκεί 2 κύκλους (έχει latency 6) και φορτώνει
; 128 bit.
; v0 = οι πρώτοι 8 float της πρώτης γραμμής
; v1 = οι πρώτοι 8 float της δεύτερης γραμμής
; v2 = οι πρώτοι 8 float της τρίτης γραμμής
; v10 = οι δεύτεροι 8 float της πρώτης γραμμής
; v11 = οι δεύτεροι 8 float της δεύτερης γραμμής
; v12 = οι δεύτεροι 8 float της τρίτης γραμμής
; v20 = οι τρίτοι 8 float της πρώτης γραμμής
; v21 = οι τρίτοι 8 float της δεύτερης γραμμής
; v22 = οι τρίτοι 8 float της τρίτης γραμμής
; τα i0, i1, i2 δείχνουν στα επόμενα δεδομένα των αντίστοιχων γραμμών.
    nop 2

    cmu.vdilv.x16 v3 v4 v0 v10  || lsu0.ldxvi v10 i0 i3
    cmu.vdilv.x16 v5 v6 v1 v11  || lsu1.ldxvi v11 i1 i3
    cmu.vdilv.x16 v7 v8 v2 v12  || lsu0.ldxvi v12 i2 i3
    cmu.alignvec v14 v4 v20 2   || lsu1.ldxvi v20 i0 i3
    cmu.alignvec v16 v6 v21 2   || lsu0.ldxvi v21 i1 i3
    cmu.alignvec v18 v8 v22 2   || lsu1.ldxvi v22 i2 i3

    cmu.max.f16 v13 v3 v4       || vau.or v0 v9 v20
    cmu.max.f16 v15 v5 v6       || vau.or v1 v9 v21
    cmu.max.f16 v17 v7 v8       || vau.or v2 v9 v22

    iau.sub i4 i4 1
    cmu.max.f16 v17 v17 v9      ;; Αυτό κάνει το ReLU
    cmu.cmz.i32 i4
    peu.pc1c eq || bru.bra ___maxPool3x3s2hhwithReLU_compensate

    cmu.max.f16 v13 v13 v15
    cmu.max.f16 v15 v14 v16
    cmu.max.f16 v13 v13 v17
    cmu.max.f16 v15 v15 v18
    nop
    cmu.max.f16 v19 v13 v15
    nop 2

    cmu.vdilv.x16 v3 v4 v0 v10  || lsu0.ldxvi v10 i0 i3 || bru.rpl i5 i4
    cmu.vdilv.x16 v5 v6 v1 v11  || lsu1.ldxvi v11 i1 i3
    cmu.vdilv.x16 v7 v8 v2 v12  || lsu0.ldxvi v12 i2 i3
    cmu.alignvec v14 v4 v20 2   || lsu1.ldxvi v20 i0 i3
    cmu.alignvec v16 v6 v21 2   || lsu0.ldxvi v21 i1 i3
    cmu.alignvec v18 v8 v22 2   || lsu1.ldxvi v22 i2 i3

    cmu.max.f16 v13 v3 v4       || lsu0.sti.64.l v19 i17
    cmu.max.f16 v15 v5 v6       || lsu0.sti.64.h v19 i17
    cmu.max.f16 v17 v7 v8       || vau.or v0 v9 v20
    cmu.max.f16 v13 v13 v15     || vau.or v1 v9 v21

    ___maxPool3x3s2hhwithReLULoop:
    cmu.max.f16 v17 v17 v9      ;; Αυτό κάνει το ReLU
    cmu.max.f16 v15 v14 v16     || vau.or v2 v9 v22
    cmu.max.f16 v13 v13 v17
    cmu.max.f16 v15 v15 v18
    nop
    cmu.max.f16 v19 v13 v15
    nop

;    lsu0.sti.64.l v19 i17
;    lsu0.sti.64.h v19 i17
;    nop 6
;    bru.jmp i30
;    nop 6


    ;Compensate
___maxPool3x3s2hhwithReLU_compensate:
    bru.jmp i30
    nop
    lsu0.sti.64.l v19 i17
    lsu0.sti.64.h v19 i17
    nop 3
;    cmu.cmz.i16 i7
;    peu.pc1c eq || bru.jmp i30
;    cmu.vdilv.x16 v3 v4 v0 v10 || lsu0.sti.64.l v19 i17
;    cmu.vdilv.x16 v5 v6 v1 v11 || lsu0.sti.64.h v19 i17
;    cmu.vdilv.x16 v7 v8 v2 v12
;    cmu.alignvec v14 v4 v20 2
;    cmu.alignvec v16 v6 v21 2
;    cmu.alignvec v18 v8 v22 2
;    cmu.max.f16 v13 v3 v4
;    cmu.max.f16 v15 v5 v6
;    cmu.max.f16 v17 v7 v8
;    cmu.max.f16 v13 v13 v15
;    cmu.max.f16 v15 v14 v16 || bru.jmp i30
;    cmu.max.f16 v13 v13 v17
;    cmu.max.f16 v15 v15 v18
;    nop
;    cmu.max.f16 v19 v13 v15
;    nop
;    lsu0.sti.64.l v19 i17

.end
