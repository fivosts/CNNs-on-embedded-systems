///
/// @file
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief
///

.version 00.87.03


.code .text.convolution1x1Fp16ToFp16
.lalign
.nowarnend

//void Convolution1x1Fp16ToFp16_asm(half** in(i18), half** out(i17), half conv[1](i16), u32 inWidth(i15))
//internal computation are made on fp16, output result is fp16
mvcvConvolution1x1Fp16ToFp16_asm:

	lsu0.ld.32 i18 i18 || lsu1.ld.32 i17 i17
	lsu0.ldil i0 0x10	

	lsu0.ldil i12 the_end || lsu1.ldih i12 the_end

	iau.shr.u32 i6 i15 3
	lsu0.ld.64.l v10 i16	||	iau.sub i6 i6 0x01

	nop 2
	lsu0.ldi.64.l v4 i18 i0 || lsu1.ldo.64.h v4 i18 0x08
	nop 5

loopa:

	bru.rpl i12 i6 the_end	||	lsu0.ldi.64.l v4 i18 i0 || lsu1.ldo.64.h v4 i18 0x08
	cmu.cpvv v3 v4
	vau.mul.f16 v6 v10 v3 || lsu0.swzv8 [00000000]
	nop 3
	lsu0.sti.64.l v6 i17 i0 	|| lsu1.sto.64.h v6 i17 0x08

the_end:
vau.mul.f16 v6 v10 v4 || lsu0.swzv8 [00000000]
nop 2
lsu0.st.64.l v6 i17  		|| lsu1.sto.64.h v6 i17 0x08

nop 7
BRU.jmp i30
nop
nop 4
// .nowarnend
.end
