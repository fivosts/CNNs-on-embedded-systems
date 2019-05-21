; /// @file
; /// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
; ///            For License Warranty see: common/license.txt
; ///
; /// @brief
; ///

.version 00.51.04

.data .data.vecVecDotProduct32

.code .text.vecVecDotProduct32
;float mvcvvecVecDotProduct32 (half* vec1, half* vec2, u32 width);
;                                     i18         i17        i16
.lalign
mvcvvecVecDotProduct32_asm:
;.nowarn 9,10,12
iau.shr.u32 i4 i16 5  || lsu0.ldil i0 vecVecDotProduct32_loop 
                      || lsu1.ldih i0 vecVecDotProduct32_loop
iau.shl i6 i4  5 	|| vau.xor v16 v16 v16
iau.sub i5 i16 i6 	|| vau.xor v17 v17 v17
cmu.cpii i10 i18
cmu.cmz.i32 i5 

vau.xor v0 v0 v0 || lsu0.ldi.128.f16.f32 v0  i10  || lsu1.ldi.128.f16.f32 v1  i17  || cmu.cmz.i32 i4
vau.xor v1 v1 v1 || lsu0.ldi.128.f16.f32 v2  i10  || lsu1.ldi.128.f16.f32 v3  i17
vau.macpz.f32 v0 v1
peu.pc1c eq || bru.bra vecVecDotProduct32_compensate

lsu0.ldi.128.f16.f32 v4  i10  || lsu1.ldi.128.f16.f32 v5  i17
lsu0.ldi.128.f16.f32 v6  i10  || lsu1.ldi.128.f16.f32 v7  i17
lsu0.ldi.128.f16.f32 v8  i10  || lsu1.ldi.128.f16.f32 v9  i17
lsu0.ldi.128.f16.f32 v10 i10  || lsu1.ldi.128.f16.f32 v11 i17
lsu0.ldi.128.f16.f32 v12 i10  || lsu1.ldi.128.f16.f32 v13 i17
lsu0.ldi.128.f16.f32 v14 i10  || lsu1.ldi.128.f16.f32 v15 i17

vau.macp.f32  v0  v1      || lsu0.ldi.128.f16.f32 v0  i10  || lsu1.ldi.128.f16.f32 v1  i17 || bru.rpl i0 i4
vecVecDotProduct32_loop:
vau.macp.f32  v2  v3      || lsu0.ldi.128.f16.f32 v2  i10  || lsu1.ldi.128.f16.f32 v3  i17
vau.macp.f32  v4  v5      || lsu0.ldi.128.f16.f32 v4  i10  || lsu1.ldi.128.f16.f32 v5  i17
vau.macp.f32  v6  v7      || lsu0.ldi.128.f16.f32 v6  i10  || lsu1.ldi.128.f16.f32 v7  i17
vau.macp.f32  v8  v9      || lsu0.ldi.128.f16.f32 v8  i10  || lsu1.ldi.128.f16.f32 v9  i17
vau.macp.f32  v10 v11     || lsu0.ldi.128.f16.f32 v10 i10  || lsu1.ldi.128.f16.f32 v11 i17
vau.macp.f32  v12 v13     || lsu0.ldi.128.f16.f32 v12 i10  || lsu1.ldi.128.f16.f32 v13 i17
vau.macp.f32  v14 v15     || lsu0.ldi.128.f16.f32 v14 i10  || lsu1.ldi.128.f16.f32 v15 i17

vecVecDotProduct32_compensate:

peu.pc1i gt || vau.macp.f32  v0   v1   || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v2   v3   || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v4   v5   || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v6   v7   || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v8   v9   || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v10  v11  || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v12  v13  || IAU.INCS i5 -4
peu.pc1i gt || vau.macp.f32  v14  v15  || IAU.INCS i5 -4

vau.macpw.f32 v18 v16 v17

nop 8
BRU.jmp i30
nop
sau.sumx.f32 i18 v18
nop 4
; .nowarnend
.end