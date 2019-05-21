.version 00.87.03

.data .data.LRN_AcrossChannels_LR5_A0_0001_B_0_75
.code .text.LRN_AcrossChannels_LR5_A0_0001_B_0_75

//					i18				i17			i16				i15
//new arguments: fp16** input, fp16** output, u16 channels, u16 pixel_batch)
//

.lalign
.nowarnend
LRN_AcrossChannels_LR5_A0_0001_B_0_75:	//einai simantiko na iparxei ayto to tag edo. alliws undefined refernce

lsu0.ld.32 i18 i18		||	lsu1.ld.32 i17 i17		//load original input and output address
iau.xor i3 i3 i3
lsu0.ldil i0 0xc5ac 	||		lsu1.ldih i0 0x37a7			//local_ratio set to fp32 5.
iau.add i15 i15 i15		||		cmu.cpivr.X32 v14 i0 		//pixel_batch * 2 gia sosto offset sta fp16 //local_ratio passed to 4 element vector

lsu0.ldil i0 0x00		||		iau.xor i2 i2 i2			//pixel_batch_offset counter = 2*4 (0 stin arxi ayksanei kata 2*4 se kathe kiklo) stoixeia (tosa xorane ston VRF) (deksio slide sto 2D)
cmu.cpii i14 i15		||		iau.add i3 i3 0x04			//i14 = 46
iau.add i2 i2 0x06

//-------------------//
//legend:: v0 = sum of squares for each given output pixel
//			v1-v5 = value of each pixel of the kernel 		//warning! 4 of them might be useless, unless they help reduce RAW hazards in first-last pixels label
//			v6-v10 = value of each pixel of the kernel squared
//			v11 = last pixel to be removed
//			v12 = current pixel
//			v13 = output
//			v14 = paronomastis

//			i0 = pixel_batch_offset counter: incremented by 4 elements for each batch
//			i1 = channel_counter
//			i2 = store size comparator (0x06)
//			i3 = store size comparator (0x02)
//			i4 = (input address + k*pixel_batch) + f(channel number)
//			i5 = (output address + k*pixel_batch) + f(channel number)
//			i6 = LRN nominator
//			i7 = LRN nominator
//			i8 = LRN nominator
//			i9 = LRN nominator
//			i10 = LRN denominator and output result
//			i11 = LRN denominator and output result
//			i12 = LRN denominator and output result
//			i13 = LRN denominator and output result
// 			i14 = unused

loop_on_whole_picture:

nop 6	//TODO
lsu0.ldil i1 0x04 	||		cmu.cpii i4 i18 			//load input address to i4 for channel sliding
cmu.cpii i5 i17			//load output address to i5 for channel sliding


//--------------------1st pixel batch---------
	cmu.cpzv v0 3	||	lsu0.ldi.128.F16.F32 v1 i4 i15
	nop 5
	lsu1.ldi.128.F16.F32 v2 i4 i15
	nop 5 	//prosoxi se aytes tis nop! sigoura diavazei se kalo offset ??
	lsu0.ldi.128.F16.F32 v3 i4 i15

	vau.mul.F32 v6 v1 v1
	nop 2
	vau.mul.F32 v7 v2 v2
	nop 2
	vau.mul.F32 v8 v3 v3
	nop 2
	vau.add.F32 v0 v0 v6 			// first pixel^2 -> s0
	nop 2
	vau.add.F32 v0 v0 v7			//second pixel^2 -> s0
	nop 2
//--------------------1st pixel computation---------
	vau.add.F32 v0 v0 v8
	nop 2

	//place if v1 = 0 here

	vau.mul.F32 v13 v0 v14		||	cmu.cpvid i6 v1.0 //cmu.cpvid i6 v12.0 		
	nop 2
	vau.add.F32 v13 v13 +1.0	||	cmu.cpvid i8 v1.2 //cmu.cpvid i8 v12.2
	cmu.cpii.F32.F16 i6 i6
	cmu.cpii.F32.F16 i7 i7
	vau.mul.F32 v15 v13 v13			//(1 + sum(xi^2) * l_r/alpha)^2
	cmu.cpii.F32.F16 i8 i8
	cmu.cpii.F32.F16 i9 i9
	vau.mul.F32 v13 v13 v15 		//(1 + sum(xi^2) * l_r/alpha)^3
	nop 2

	cmu.cpvid i10 v13.0				//copy (1 + sum(xi^2) * l_r/alpha)^3 to i10->i13
	cmu.cpvid i12 v13.2

	lsu0.ldi.128.F16.F32 v4 i4 i15 

	cmu.cpii.F32.F16 i10 i10
	cmu.cpii.F32.F16 i11 i11	||	sau.sqt i10 i10.0	//conversion to fp16 for square rooting
	cmu.cpii.F32.F16 i12 i12	||	sau.sqt i11 i11.0
	cmu.cpii.F32.F16 i13 i13	||	sau.sqt i12 i12.0

	sau.sqt i13 i13.0
	nop								
	sau.sqt i10 i10.0	||	cmu.cmii.I32 i14 i2
	sau.sqt i11 i11.0
	sau.sqt i12 i12.0
	sau.sqt i13 i13.0

	sau.div.F16 i10 i6 i10
	sau.div.F16 i11 i7 i11
	sau.div.F16 i12 i8 i12
	sau.div.F16 i13 i9 i13

	vau.mul.F32 v9 v4 v4

	peu.pc1c GT		||	bru.bra first_store_4
	peu.pc1c EQ		|| 	bru.bra first_store_3
	cmu.cmii.I32 i14 i3
	peu.pc1c EQ		|| 	bru.bra first_store_2
	peu.pc1c LT 	|| 	bru.bra first_store_1
	nop 6

	first_store_3:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.add i5 i5 0x02
		lsu0.st.16 i12 i5	||	iau.sub i5 i5 0x04
		iau.add i5 i5 i15	||	vau.add.F32 v0 v0 v9
		bru.bra first_next
		nop 6

	first_store_2:
		lsu0.st.16 i10 i5 	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.sub i5 i5 0x02
		iau.add i5 i5 i15	||	vau.add.F32 v0 v0 v9
		bru.bra first_next
		nop 6

	first_store_1:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 i15
		bru.bra first_next	||	vau.add.F32 v0 v0 v9
		nop 6

	first_store_4:
		cmu.cpiv.X16 v13.0 i10.L
		cmu.cpiv.X16 v13.1 i11.L	||	vau.add.F32 v0 v0 v9
		cmu.cpiv.X16 v13.2 i12.L
		cmu.cpiv.X16 v13.3 i13.L
		lsu0.sti.64.L v13 i5 i15

	first_next:

//--------------------2nd pixel computation---------
	vau.mul.F32 v13 v0 v14		||	cmu.cpvid i6 v2.0//cmu.cpvid i6 v12.0
	nop 2
	vau.add.F32 v13 v13 +1.0	||	cmu.cpvid i8 v2.2//cmu.cpvid i8 v12.2
	cmu.cpii.F32.F16 i6 i6 		//TODO na ta valo pano mpas kai figoun ta nop 2
	cmu.cpii.F32.F16 i7 i7
	vau.mul.F32 v15 v13 v13	
	cmu.cpii.F32.F16 i8 i8 	//curent pixels edo!!!!
	cmu.cpii.F32.F16 i9 i9
	vau.mul.F32 v13 v13 v15
	nop 2
	cmu.cpvid i10 v13.0					
	cmu.cpvid i12 v13.2

	nop

	cmu.cpii.F32.F16 i10 i10
	cmu.cpii.F32.F16 i11 i11	||	sau.sqt i10 i10.0
	cmu.cpii.F32.F16 i12 i12	||	sau.sqt i11 i11.0
	cmu.cpii.F32.F16 i13 i13	||	sau.sqt i12 i12.0

	sau.sqt i13 i13.0	
	nop
	sau.sqt i10 i10.0	||	cmu.cmii.I32 i14 i2
	sau.sqt i11 i11.0
	sau.sqt i12 i12.0
	sau.sqt i13 i13.0

	nop
	sau.div.F16 i10 i6 i10
	sau.div.F16 i11 i7 i11				
	sau.div.F16 i12 i8 i12
	sau.div.F16 i13 i9 i13

	peu.pc1c GT		||	bru.bra second_store_4
	peu.pc1c EQ		|| 	bru.bra second_store_3
	cmu.cmii.I32 i14 i3
	peu.pc1c EQ		|| 	bru.bra second_store_2
	peu.pc1c LT 	|| 	bru.bra second_store_1
	nop 6

	second_store_3:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.add i5 i5 0x02
		lsu0.st.16 i12 i5	||	iau.sub i5 i5 0x04
		iau.add i5 i5 i15
		bru.bra main_loop
		nop 6

	second_store_2:
		lsu0.st.16 i10 i5 	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.sub i5 i5 0x02
		iau.add i5 i5 i15
		bru.bra main_loop
		nop 6

	second_store_1:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 i15
		bru.bra main_loop
		nop 6
	
	second_store_4:
		cmu.cpiv.X16 v13.0 i10.L
		cmu.cpiv.X16 v13.1 i11.L
		cmu.cpiv.X16 v13.2 i12.L
		cmu.cpiv.X16 v13.3 i13.L
		lsu0.sti.64.L v13 i5 i15

main_loop:
//--------------------5th pixel---------
	lsu0.ldi.128.F16.F32 v5 i4 i15	||	cmu.cpvid i6 v3.0
	iau.add i1 i1 0x01	||	cmu.cpvid i8 v3.2
	nop
	cmu.cpii.F32.F16 i6 i6
	cmu.cpii.F32.F16 i7 i7
	cmu.cpii.F32.F16 i8 i8
	cmu.cpii.F32.F16 i9 i9
	vau.mul.F32 v10 v5 v5
	nop 2
	vau.add.F32 v0 v0 v10
	nop 2

//--------------------3rd pixel computation---------	
	vau.mul.F32 v13 v0 v14	 //cmu.cpvid i6 v12.0 
	nop 2
	vau.add.F32 v13 v13 +1.0	//cmu.cpvid i8 v12.2			
	nop 2
	vau.mul.F32 v15 v13 v13		
	nop 2
	vau.mul.F32 v13 v13 v15 		
	nop 2
	cmu.cpvid i10 v13.0					
	cmu.cpvid i12 v13.2

	nop

	cmu.cpii.F32.F16 i10 i10	||	vau.sub.F32 v0 v0 v6
	cmu.cpii.F32.F16 i11 i11	||	sau.sqt i10 i10.0
	cmu.cpii.F32.F16 i12 i12	||	sau.sqt i11 i11.0
	cmu.cpii.F32.F16 i13 i13	||	sau.sqt i12 i12.0

	sau.sqt i13 i13.0	||	cmu.cmii.I32 i14 i2	
	cmu.cpvv v3 v4
	sau.sqt i10 i10.0	||	cmu.cpvv v6 v7
	sau.sqt i11 i11.0	||	cmu.cpvv v7 v8
	sau.sqt i12 i12.0	||	cmu.cpvv v8 v9
	sau.sqt i13 i13.0	||	cmu.cpvv v9 v10

	cmu.cpvv v4 v5
	sau.div.F16 i10 i6 i10
	sau.div.F16 i11 i7 i11
	sau.div.F16 i12 i8 i12
	sau.div.F16 i13 i9 i13
	
	peu.pc1c GT		||	bru.bra loop_store_4
	peu.pc1c EQ		|| 	bru.bra loop_store_3
	cmu.cmii.I32 i14 i3
	peu.pc1c EQ		|| 	bru.bra loop_store_2
	peu.pc1c LT 	|| 	bru.bra loop_store_1
	nop 6

	loop_store_3:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.add i5 i5 0x02
		lsu0.st.16 i12 i5	||	iau.sub i5 i5 0x04
		iau.add i5 i5 i15	||	cmu.cmii.U16 i1 i16
		bru.bra loop_next
		nop 6

	loop_store_2:
		lsu0.st.16 i10 i5 	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.sub i5 i5 0x02
		iau.add i5 i5 i15	||	cmu.cmii.U16 i1 i16
		bru.bra loop_next
		nop 6

	loop_store_1:
		cmu.cmii.U16 i1 i16	||	lsu0.st.16 i10 i5	||	iau.add i5 i5 i15
		bru.bra loop_next
		nop 6
	
	loop_store_4:
		cmu.cpiv.X16 v13.0 i10.L
		cmu.cpiv.X16 v13.1 i11.L
		cmu.cpiv.X16 v13.2 i12.L
		cmu.cpiv.X16 v13.3 i13.L
		lsu0.sti.64.L v13 i5 i15	||	cmu.cmii.U16 i1 i16

	loop_next:
		
	peu.pc1c NEQ	||	bru.bra main_loop

final_pixels:

//--------------------semi final pixel computation---------
	vau.mul.F32 v13 v0 v14	||	cmu.cpvid i6 v3.0//cmu.cpvid i6 v12.0
	nop 2
	vau.add.F32 v13 v13 +1.0	||	cmu.cpvid i8 v3.2//cmu.cpvid i8 v12.2
	cmu.cpii.F32.F16 i6 i6
	cmu.cpii.F32.F16 i7 i7
	vau.mul.F32 v15 v13 v13	
	cmu.cpii.F32.F16 i8 i8
	cmu.cpii.F32.F16 i9 i9
	vau.mul.F32 v13 v13 v15
	nop 2
	cmu.cpvid i10 v13.0	
	cmu.cpvid i12 v13.2

	nop

	cmu.cpii.F32.F16 i10 i10
	cmu.cpii.F32.F16 i11 i11	||	sau.sqt i10 i10.0
	cmu.cpii.F32.F16 i12 i12	||	sau.sqt i11 i11.0
	cmu.cpii.F32.F16 i13 i13	||	sau.sqt i12 i12.0

	sau.sqt i13 i13.0	
	nop
	sau.sqt i10 i10.0	||	cmu.cmii.I32 i14 i2
	sau.sqt i11 i11.0
	sau.sqt i12 i12.0	||	vau.sub.F32 v0 v0 v7
	sau.sqt i13 i13.0

	nop
	sau.div.F16 i10 i6 i10
	sau.div.F16 i11 i7 i11
	sau.div.F16 i12 i8 i12
	sau.div.F16 i13 i9 i13

	peu.pc1c GT		||	bru.bra semi_store_4
	peu.pc1c EQ		|| 	bru.bra semi_store_3
	cmu.cmii.I32 i14 i3
	peu.pc1c EQ		|| 	bru.bra semi_store_2
	peu.pc1c LT 	|| 	bru.bra semi_store_1
	nop 6

	semi_store_3:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.add i5 i5 0x02
		lsu0.st.16 i12 i5	||	iau.sub i5 i5 0x04
		iau.add i5 i5 i15
		bru.bra semi_next
		nop 6

	semi_store_2:
		lsu0.st.16 i10 i5 	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.sub i5 i5 0x02
		iau.add i5 i5 i15
		bru.bra semi_next
		nop 6

	semi_store_1:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 i15
		bru.bra semi_next
		nop 6
	
	semi_store_4:
		cmu.cpiv.X16 v13.0 i10.L
		cmu.cpiv.X16 v13.1 i11.L
		cmu.cpiv.X16 v13.2 i12.L
		cmu.cpiv.X16 v13.3 i13.L
		lsu0.sti.64.L v13 i5 i15

	semi_next:

//--------------------final pixel computation---------
	
	vau.mul.F32 v13 v0 v14	||	cmu.cpvid i6 v4.0//cmu.cpvid i6 v12.0
	nop 2
	vau.add.F32 v13 v13 +1.0	||	cmu.cpvid i8 v4.2//cmu.cpvid i8 v12.2
	nop 2
	vau.mul.F32 v15 v13 v13	
	cmu.cpii.F32.F16 i6 i6
	cmu.cpii.F32.F16 i7 i7
	vau.mul.F32 v13 v13 v15 
	cmu.cpii.F32.F16 i8 i8
	cmu.cpii.F32.F16 i9 i9
	cmu.cpvid i10 v13.0	
	cmu.cpvid i12 v13.2

	nop

	cmu.cpii.F32.F16 i10 i10	||	iau.add i18 i18 0x08
	cmu.cpii.F32.F16 i11 i11	||	sau.sqt i10 i10.0
	cmu.cpii.F32.F16 i12 i12	||	sau.sqt i11 i11.0
	cmu.cpii.F32.F16 i13 i13	||	sau.sqt i12 i12.0

	sau.sqt i13 i13.0	||	iau.add i17 i17 0x08
	nop
	sau.sqt i10 i10.0	||	cmu.cmii.I32 i14 i2
	sau.sqt i11 i11.0	||	iau.add i0 i0 0x08
	sau.sqt i12 i12.0
	sau.sqt i13 i13.0

	nop
	sau.div.F16 i10 i6 i10
	sau.div.F16 i11 i7 i11
	sau.div.F16 i12 i8 i12
	sau.div.F16 i13 i9 i13

	peu.pc1c GT		||	bru.bra final_store_4
	peu.pc1c EQ		|| 	bru.bra final_store_3
	cmu.cmii.I32 i14 i3
	peu.pc1c EQ		|| 	bru.bra final_store_2
	peu.pc1c LT 	|| 	bru.bra final_store_1
	nop 6

	final_store_3:
		lsu0.st.16 i10 i5	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.add i5 i5 0x02
		lsu0.st.16 i12 i5	||	iau.sub i5 i5 0x04
		iau.add i5 i5 i15	||	cmu.cmii.U16 i0 i15
		bru.bra final_next
		nop 6

	final_store_2:
		lsu0.st.16 i10 i5 	||	iau.add i5 i5 0x02
		lsu1.st.16 i11 i5	||	iau.sub i5 i5 0x02
		iau.add i5 i5 i15	||	cmu.cmii.U16 i0 i15
		bru.bra final_next
		nop 6

	final_store_1:
		cmu.cmii.U16 i0 i15	||	lsu0.st.16 i10 i5	||	iau.add i5 i5 i15
		bru.bra final_next
		nop 6
	
	final_store_4:
		cmu.cpiv.X16 v13.0 i10.L
		cmu.cpiv.X16 v13.1 i11.L
		cmu.cpiv.X16 v13.2 i12.L
		cmu.cpiv.X16 v13.3 i13.L
		lsu0.sti.64.L v13 i5 i15	||	cmu.cmii.U16 i0 i15

	final_next:

iau.sub i14 i14 0x08
peu.pc1c LT		||	bru.bra loop_on_whole_picture

Lrn_end:

nop 7
BRU.jmp i30
nop
nop 4
// .nowarnend
.end
