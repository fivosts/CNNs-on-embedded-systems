///
/// @file
/// @copyright All code copyright Movidius Ltd 2013, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief   
///

.version 00.51.00
.data .rodata.mvcvAvgPool7x7xk_asm 
.align 16
 mvcvAvgPool7x7xk_asm_multiply:
		.float16    0.02040816326530612
//-------------------------------------------------------------------------------	
.code .text.AvgPool7x7xk
//	void AvgPool7x7xk_asm(	fp16** src1, fp16** dst, u32 width)
//			   				i18				i17			i16
.lalign
mvcvAvgPool7x7xk_asm:
lsu0.ldil i0 mvcvAvgPool7x7xk_asm_multiply || lsu1.ldih i0 mvcvAvgPool7x7xk_asm_multiply 
lsu0.ldil i5 mvcvAvgPool7x7xk_asm_loop || lsu1.ldih i5 mvcvAvgPool7x7xk_asm_loop 
lsu0.ld.32 i17 i17 || lsu1.ld.16r v18 i0 		//mipos thelei ld o i18?
nop 4
lsu0.ld.32 i18 i18
nop 10
lsu0.ldo.32   i0  i18 192 || IAU.SHR.i32 i16 i16 3
lsu0.ldo.64.l  v0 i18   0 || lsu1.ldo.64.h  v0 i18 8 || vau.xor v12 v12 v12
lsu0.ldo.64.l  v1 i18  16 || lsu1.ldo.64.h  v1 i18 24 
lsu0.ldo.64.l  v2 i18  32 || lsu1.ldo.64.h  v2 i18 40
lsu0.ldo.64.l  v3 i18  48 || lsu1.ldo.64.h  v3 i18 56 
lsu0.ldo.64.l  v4 i18  64 || lsu1.ldo.64.h  v4 i18 72
lsu0.ldo.64.l  v5 i18  80 || lsu1.ldo.64.h  v5 i18 88 
lsu0.ldo.64.l  v6 i18  96 || lsu1.ldo.64.h  v6 i18 104 || cmu.cpiv.x32 v12.0 i0
lsu0.ldo.64.l  v7 i18 112 || lsu1.ldo.64.h  v7 i18 120 || cmu.cpvi.x32 i1 v0.0
lsu0.ldo.64.l  v8 i18 128 || lsu1.ldo.64.h  v8 i18 136 
lsu0.ldo.64.l  v9 i18 144 || lsu1.ldo.64.h  v9 i18 152 
lsu0.ldo.64.l v10 i18 160 || lsu1.ldo.64.h v10 i18 168 
lsu0.ldo.64.l v11 i18 176 || lsu1.ldo.64.h v11 i18 184 
cmu.cpvi.x32 i2 v0.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8
cmu.cpvi.x32 i3 v0.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8
cmu.cpvi.x32 i4 v0.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8
cmu.cpvi.x32 i1 v1.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8                                             
cmu.cpvi.x32 i2 v1.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8                                                      
cmu.cpvi.x32 i3 v1.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8                                                      
cmu.cpvi.x32 i4 v1.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || vau.add.i32  v0  v0 16 || bru.rpl i5 i16		 
cmu.cpvi.x32 i1 v2.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || vau.add.i32  v1  v1 16	                                                  
cmu.cpvi.x32 i2 v2.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACPZ.f16 v13  v18                                                         
cmu.cpvi.x32 i3 v2.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                           
cmu.cpvi.x32 i4 v2.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v3.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18                                                 
cmu.cpvi.x32 i2 v3.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18                                                      
cmu.cpvi.x32 i3 v3.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                      
cmu.cpvi.x32 i4 v3.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v4.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18                                          
cmu.cpvi.x32 i2 v4.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18                                                     
cmu.cpvi.x32 i3 v4.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                     
cmu.cpvi.x32 i4 v4.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v5.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18                                 
cmu.cpvi.x32 i2 v5.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18                                                      
cmu.cpvi.x32 i3 v5.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                      
cmu.cpvi.x32 i4 v5.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v6.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18                                 
cmu.cpvi.x32 i2 v6.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18                                                     
cmu.cpvi.x32 i3 v6.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                     
cmu.cpvi.x32 i4 v6.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v7.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18                                 
cmu.cpvi.x32 i2 v7.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18                                                     
cmu.cpvi.x32 i3 v7.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                     
cmu.cpvi.x32 i4 v7.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v8.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18	                                  
cmu.cpvi.x32 i2 v8.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18 	                                                 
cmu.cpvi.x32 i3 v8.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                     
cmu.cpvi.x32 i4 v8.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v9.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18	                                  
cmu.cpvi.x32 i2 v9.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18 	                                                  
cmu.cpvi.x32 i3 v9.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                      
cmu.cpvi.x32 i4 v9.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v10.0  || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18	                            
cmu.cpvi.x32 i2 v10.1  || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18 	                                                  
cmu.cpvi.x32 i3 v10.2  || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                      
cmu.cpvi.x32 i4 v10.3  || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v11.0  || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18	                                        
cmu.cpvi.x32 i2 v11.1  || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18 	                                                 
cmu.cpvi.x32 i3 v11.2  || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || VAU.MACP.f16  v14  v18                                                     
cmu.cpvi.x32 i4 v11.3  || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || VAU.MACP.f16  v15  v18    
cmu.cpvi.x32 i1 v12.0  || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || VAU.MACP.f16  v16  v18	 
                          lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || VAU.MACP.f16  v13  v18 	
					                                                              VAU.MACP.f16  v14  v18             
																			      VAU.MACP.f16  v15  v18    			 
																			      VAU.MACP.f16  v16  v18				 
																				  VAU.MACP.f16  v13  v18				 
																				  VAU.MACP.f16  v14  v18				 
																				  VAU.MACP.f16  v15  v18				 
																				  VAU.MACP.f16  v16  v18				 
																				  VAU.MACPW.f16 v19 v13 v18	 
																			     nop 8
                                                                                  vau.add.i32  v2  v2 16 ||	lsu0.cp i1 v0.0   	 
lsu0.sto.64.l  v19 i17   0 || lsu1.sto.64.h  v19 i17 8 || iau.add i17 i17 16   || vau.add.i32  v3  v3 16		 
																				  vau.add.i32  v4  v4 16		 
																				  vau.add.i32  v5  v5 16	
mvcvAvgPool7x7xk_asm_loop:																	  
vau.add.i32  v6  v6 16		                               
cmu.cpvi.x32 i2 v0.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || vau.add.i32  v7  v7 16		 
cmu.cpvi.x32 i3 v0.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || vau.add.i32  v8  v8 16		 
cmu.cpvi.x32 i4 v0.3   || lsu0.ldo.64.l  v15 i3   0 || lsu1.ldo.64.h  v15 i3 8 || vau.add.i32  v9  v9 16		 
cmu.cpvi.x32 i1 v1.0   || lsu0.ldo.64.l  v16 i4   0 || lsu1.ldo.64.h  v16 i4 8 || vau.add.i32 v10 v10 16		 
cmu.cpvi.x32 i2 v1.1   || lsu0.ldo.64.l  v13 i1   0 || lsu1.ldo.64.h  v13 i1 8 || vau.add.i32 v11 v11 16		 
cmu.cpvi.x32 i3 v1.2   || lsu0.ldo.64.l  v14 i2   0 || lsu1.ldo.64.h  v14 i2 8 || vau.add.i32 v12 v12 16																									 

BRU.JMP i30
nop 6

.end
