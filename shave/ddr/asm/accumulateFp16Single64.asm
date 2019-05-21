; /// @file
; /// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
; ///            For License Warranty see: common/license.txt
; ///
; /// @brief
; ///

.version 00.51.04

.data .data.accumulateFp16Single64

.code .text.accumulateFp16Single64
;void mvcvAccumulateFp16Single64 (half* dst, half bias, u32 width);
;                                     i18       i17       i16
.lalign
mvcvAccumulateFp16Single64_asm:
;.nowarn 9,10,12
iau.shr.u32 i4 i16 6  || lsu0.ldil i0 accumulateFp16Single64_loop 
                      || lsu1.ldih i0 accumulateFp16Single64_loop

iau.shl i6 i4  6      || cmu.cpivr.x16 v0 i17
iau.sub i5 i16 i6
cmu.cpii i10 i18

lsu0.ldo.64.l v1 i10 0   || lsu1.ldo.64.h v1 i10 8
lsu0.ldo.64.l v2 i10 16  || lsu1.ldo.64.h v2 i10 24 || cmu.cmz.i32 i4
peu.pc1c eq || bru.bra accumulateFp16Single64_compensate
lsu0.ldo.64.l v3 i10 32  || lsu1.ldo.64.h v3 i10 40 
lsu0.ldo.64.l v4 i10 48  || lsu1.ldo.64.h v4 i10 56
lsu0.ldo.64.l v5 i10 64  || lsu1.ldo.64.h v5 i10 72 
lsu0.ldo.64.l v6 i10 80  || lsu1.ldo.64.h v6 i10 88 
lsu0.ldo.64.l v7 i10 96  || lsu1.ldo.64.h v7 i10 104
lsu0.ldo.64.l v8 i10 112 || lsu1.ldo.64.h v8 i10 120  || iau.add i10 i10 128

vau.add.f16 v9  v1 v0   || lsu0.ldo.64.l v1 i10 0   || lsu1.ldo.64.h v1 i10 8   || bru.rpl i0 i4
vau.add.f16 v10 v2 v0   || lsu0.ldo.64.l v2 i10 16  || lsu1.ldo.64.h v2 i10 24
vau.add.f16 v11 v3 v0   || lsu0.ldo.64.l v3 i10 32  || lsu1.ldo.64.h v3 i10 40
vau.add.f16 v12 v4 v0   || lsu0.ldo.64.l v4 i10 48  || lsu1.ldo.64.h v4 i10 56
vau.add.f16 v13 v5 v0   || lsu0.ldo.64.l v5 i10 64  || lsu1.ldo.64.h v5 i10 72
vau.add.f16 v14 v6 v0   || lsu0.ldo.64.l v6 i10 80  || lsu1.ldo.64.h v6 i10 88
vau.add.f16 v15 v7 v0   || lsu0.ldo.64.l v7 i10 96  || lsu1.ldo.64.h v7 i10 104
vau.add.f16 v16 v8 v0   || lsu0.ldo.64.l v8 i10 112 || lsu1.ldo.64.h v8 i10 120  || iau.add i10 i10 128


lsu0.sto.64.l v9  i18 0    || lsu1.sto.64.h v9  i18 8
accumulateFp16Single64_loop:
lsu0.sto.64.l v10 i18 16   || lsu1.sto.64.h v10 i18 24
lsu0.sto.64.l v11 i18 32   || lsu1.sto.64.h v11 i18 40
lsu0.sto.64.l v12 i18 48   || lsu1.sto.64.h v12 i18 56
lsu0.sto.64.l v13 i18 64   || lsu1.sto.64.h v13 i18 72
lsu0.sto.64.l v14 i18 80   || lsu1.sto.64.h v14 i18 88
lsu0.sto.64.l v15 i18 96   || lsu1.sto.64.h v15 i18 104
lsu0.sto.64.l v16 i18 112  || lsu1.sto.64.h v16 i18 120 || iau.add i18 i18 128

accumulateFp16Single64_compensate:

vau.add.f16 v9   v1 v0
vau.add.f16 v10  v2 v0
; Αν το width είναι το μηδέν, τότε δεν αλλάζουμε καθόλου το dst.
cmu.cmz.i32 i5
peu.pc1c eq || bru.bra accumulateFp16Single64_exit

vau.add.f16 v11  v3 v0
vau.add.f16 v12  v4 v0
vau.add.f16 v13  v5 v0
vau.add.f16 v14  v6 v0
vau.add.f16 v15  v7 v0
vau.add.f16 v16  v8 v0

; Κάνουμε στρογγυλοποίηση προς τα πάνω το width % 64 στο κοντινότερο πολλαπλάσιο
; του 8.
               lsu0.sto.64.l v9  i18 0   || lsu1.sto.64.h v9  i18 8   || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v10 i18 16  || lsu1.sto.64.h v10 i18 24  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v11 i18 32  || lsu1.sto.64.h v11 i18 40  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v12 i18 48  || lsu1.sto.64.h v12 i18 56  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v13 i18 64  || lsu1.sto.64.h v13 i18 72  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v14 i18 80  || lsu1.sto.64.h v14 i18 88  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v15 i18 96  || lsu1.sto.64.h v15 i18 104 || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v16 i18 112 || lsu1.sto.64.h v16 i18 120 || IAU.INCS i5 -8

accumulateFp16Single64_exit:

BRU.jmp i30
nop 6
; .nowarnend
.end