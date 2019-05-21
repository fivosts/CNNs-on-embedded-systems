; /// @file
; /// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
; ///            For License Warranty see: common/license.txt
; ///
; /// @brief
; ///

.version 00.51.04

.data .data.accumulateFp16Single32

.code .text.accumulateFp16Single32
;void mvcvAccumulateFp16Single32 (half* dst, half bias, u32 width);
;                                     i18       i17       i16
.lalign
mvcvAccumulateFp16Single32_asm:
;.nowarn 9,10,12
iau.shr.u32 i4 i16 5  || lsu0.ldil i0 accumulateFp16Single32_loop 
                      || lsu1.ldih i0 accumulateFp16Single32_loop

iau.shl i6 i4  5      || cmu.cpivr.x16 v0 i17
iau.sub i5 i16 i6
cmu.cpii i10 i18

lsu0.ldo.64.l v1 i10 0   || lsu1.ldo.64.h v1 i10 8  || cmu.cmz.i32 i4
peu.pc1c eq || bru.bra accumulateFp16Single32_compensate
lsu0.ldo.64.l v2 i10 16  || lsu1.ldo.64.h v2 i10 24
lsu0.ldo.64.l v3 i10 32  || lsu1.ldo.64.h v3 i10 40 
lsu0.ldo.64.l v4 i10 48  || lsu1.ldo.64.h v4 i10 56 || iau.add i10 i10 64
nop 4

vau.add.f16 v9  v1 v0   || lsu0.ldo.64.l v1 i10 0   || lsu1.ldo.64.h v1 i10 8   || bru.rpl i0 i4
accumulateFp16Single32_loop:
vau.add.f16 v10 v2 v0   || lsu0.ldo.64.l v2 i10 16  || lsu1.ldo.64.h v2 i10 24
vau.add.f16 v11 v3 v0   || lsu0.ldo.64.l v3 i10 32  || lsu1.ldo.64.h v3 i10 40
vau.add.f16 v12 v4 v0   || lsu0.ldo.64.l v4 i10 48  || lsu1.ldo.64.h v4 i10 56  || iau.add i10 i10 64
lsu0.sto.64.l v9  i18 0    || lsu1.sto.64.h v9  i18 8
lsu0.sto.64.l v10 i18 16   || lsu1.sto.64.h v10 i18 24
lsu0.sto.64.l v11 i18 32   || lsu1.sto.64.h v11 i18 40
lsu0.sto.64.l v12 i18 48   || lsu1.sto.64.h v12 i18 56 || iau.add i18 i18 64


accumulateFp16Single32_compensate:

; Αν το width είναι το μηδέν, τότε δεν αλλάζουμε καθόλου το dst.
cmu.cmz.i32 i5
peu.pc1c eq || bru.bra accumulateFp16Single32_exit

vau.add.f16 v9   v1 v0
vau.add.f16 v10  v2 v0
vau.add.f16 v11  v3 v0
vau.add.f16 v12  v4 v0
nop 2

; Κάνουμε στρογγυλοποίηση προς τα πάνω το width % 32 στο κοντινότερο πολλαπλάσιο
; του 8.
               lsu0.sto.64.l v9  i18 0   || lsu1.sto.64.h v9  i18 8   || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v10 i18 16  || lsu1.sto.64.h v10 i18 24  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v11 i18 32  || lsu1.sto.64.h v11 i18 40  || IAU.INCS i5 -8
peu.pc1i gt || lsu0.sto.64.l v12 i18 48  || lsu1.sto.64.h v12 i18 56  || IAU.INCS i5 -8

accumulateFp16Single32_exit:

BRU.jmp i30
nop 6
; .nowarnend
.end