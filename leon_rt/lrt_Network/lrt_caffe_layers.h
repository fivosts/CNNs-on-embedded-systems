#ifndef LRT_CAFFE_LAYERS_H
#define LRT_CAFFE_LAYERS_H
#include <ddr_functions_types.h>
#include <network_defines.h>

class lrt_Layers{
public:
	inline lrt_Layers(){}
	virtual ~lrt_Layers(){}
	virtual u64 execute(){ return 0; }
	#ifdef DUAL_CPU
		u8 event_handler;
	#endif

protected:
	u8 *output_buffer;
	u8 ddr_function;
	u8 shaves_used;
	u16 input_height;
	u16 input_width;
	u16 channels;
	u8 *bottom_output_buffer;
	u16 bottom_channels;
	u16 bottom_input_height;
	u16 bottom_input_width;
};

#if INPUT
class Input : public lrt_Layers{
public:
	#ifdef PROFILE
		inline Input(u8 *output, u16 channel, u16 height, u16 width, u8 *bottom_buffer,
						u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
			(void)bottom_buffer;
			(void)bottom_ch;
			(void)bottom_height;
			(void)bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline Input(u8 *output, u16 channel, u16 height, u16 width, u8 *bottom_buffer,
						u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 event){
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
			event_handler = event;
			(void)bottom_buffer;
			(void)bottom_ch;
			(void)bottom_height;
			(void)bottom_width;
		}
	#endif
	~Input(){}
	u64 execute() override{
		return 0;
	}

private:
};
#endif

#if CONVOLUTION
class Convolution : public lrt_Layers{
public:
	#ifdef PROFILE
		inline Convolution(u8 *output, u8 shaves, u8 *weight, u8 *bias, 
							u16 channel, u16 height, u16 width, u8 *bottom_buffer,
							u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 k_size, u8 strid,
							u8 padding, u8 grp, u8 function, u8 ReLU){
			
			output_buffer = output;
			shaves_used = shaves;
			channels = channel;
			input_height = height;
			input_width = width;
			kernel_size = k_size;
			stride = strid;
			pad = padding;
			group = grp;
			ddr_function = function;
			ReLU_flag = ReLU;
			weight_pointer = weight;
			bias_pointer = bias;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline Convolution(u8 *output, u8 shaves, u8 *weight, u8 *bias, 
								u16 channel, u16 height, u16 width, u8 *bottom_buffer,
								u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 k_size, u8 strid,
								u8 padding, u8 grp, u8 function, u8 event, u8 ReLU){
				
				output_buffer = output;
				shaves_used = shaves;
				channels = channel;
				input_height = height;
				input_width = width;
				kernel_size = k_size;
				stride = strid;
				pad = padding;
				group = grp;
				ddr_function = function;
				event_handler = event;
				ReLU_flag = ReLU;
				weight_pointer = weight;
				bias_pointer = bias;
				bottom_output_buffer = bottom_buffer;
				bottom_channels = bottom_ch;
				bottom_input_height = bottom_height;
				bottom_input_width = bottom_width;
			}
	#endif
	~Convolution(){}
	u64 execute() override;

private:
	u8 kernel_size;
	u8 stride;
	u8 pad;
	u8 group;
	u8 ReLU_flag;
	u8 *weight_pointer;
	u8 *bias_pointer;
};
#endif

#if POOLING
class Pooling : public lrt_Layers{
public:
	#ifdef PROFILE
		inline Pooling(u8 *output, u8 shaves, u16 channel, u16 height, u16 width, 
							u8 k_size, u8 strid, u8 padding, u8 function,
								pooling_type pool_m, u8 *bottom_buffer,
									u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			shaves_used = shaves;
			channels = channel;
			input_height = height;
			input_width = width;
			kernel_size = k_size;
			stride = strid;
			pad = padding;
			ddr_function = function;
			pooling_method = pool_m;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline Pooling(u8 *output, u8 shaves, u16 channel, u16 height, u16 width, 
							u8 k_size, u8 strid, u8 padding, u8 function, pooling_type pool_m, u8 *bottom_buffer, 
								u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 event){
			
			output_buffer = output;
			shaves_used = shaves;
			channels = channel;
			input_height = height;
			input_width = width;
			kernel_size = k_size;
			stride = strid;
			pad = padding;
			ddr_function = function;
			pooling_method = pool_m;
			event_handler = event;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#endif
	~Pooling(){}
	u64 execute() override;

private:
	u8 kernel_size;	
	u8 stride;
	u8 pad;
	pooling_type pooling_method;
};
#endif

#if INNERPRODUCT
class InnerProduct : public lrt_Layers{
public:
	#ifdef PROFILE
		inline InnerProduct(u8 *output, u8 shaves, u8 *weight, u8 *bias, 
								u16 width, u8 *bottom_buffer, u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 ReLU){
			
			output_buffer = output;
			shaves_used = shaves;
			input_height = 1;
			input_width = width;
			channels = 1;
			weight_pointer = weight;
			bias_pointer = bias;
			ReLU_flag = ReLU;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline InnerProduct(u8 *output, u8 shaves, u8 *weight, u8 *bias, 
								u16 width, u8 *bottom_buffer, u16 bottom_ch, u16 bottom_height, u16 bottom_width, u8 event, u8 ReLU){
			
			output_buffer = output;
			shaves_used = shaves;
			input_height = 1;
			input_width = width;
			channels = 1;
			weight_pointer = weight;
			bias_pointer = bias;
			ReLU_flag = ReLU;
			event_handler = event;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#endif
	~InnerProduct(){}
	u64 execute() override;

private:
	u8 *weight_pointer;
	u8 *bias_pointer;
	u8 ReLU_flag;
};
#endif

#if LRN
class Lrn : public lrt_Layers{
public:
	#ifdef PROFILE
		inline Lrn(u8 *output, u8 shaves, u8 l_size, fp16 a, fp16 b, u8 *bottom_buffer,
							u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			shaves_used = shaves;
			ddr_function = 35;
			local_size = l_size;
			alpha = a;
			beta = b;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
		inline Lrn(u8 *output, u8 shaves, u8 *bottom_buffer,
									u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			ddr_function = 36;
			shaves_used = shaves;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline Lrn(u8 *output, u8 shaves, u8 l_size, fp16 a, fp16 b, u8 event, u8 *bottom_buffer,
							u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			shaves_used = shaves;
			ddr_function = 35;
			local_size = l_size;
			alpha = a;
			beta = b;
			event_handler = event;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
		inline Lrn(u8 *output, u8 shaves, u8 event, u8 *bottom_buffer,
									u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			ddr_function = 36;
			shaves_used = shaves;
			event_handler = event;
			bottom_output_buffer = bottom_buffer;
			bottom_channels = bottom_ch;
			bottom_input_height = bottom_height;
			bottom_input_width = bottom_width;
		}
	#endif
	~Lrn(){}
	u64 execute() override;

private:
	u8 local_size;
	fp16 alpha;
	fp16 beta;
};
#endif

#if !(LINEAR)
class Concat : public lrt_Layers{
public:
	#ifdef PROFILE
		inline Concat(u8 *output, u16 channel, u16 height, u16 width, u8 *bottom_buffer,
								u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
			(void)bottom_buffer;
			(void)bottom_ch;
			(void)bottom_height;
			(void)bottom_width;
		}
	#elif defined(DUAL_CPU)
		inline Concat(u8 *output, u16 channel, u16 height, u16 width, u8 event, u8 *bottom_buffer,
								u16 bottom_ch, u16 bottom_height, u16 bottom_width){
			
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
			event_handler = event;
			(void)bottom_buffer;
			(void)bottom_ch;
			(void)bottom_height;
			(void)bottom_width;
		}
	#endif
	~Concat(){}
	u64 execute() override{	return 0;}

private:
};
#endif
#endif