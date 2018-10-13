#ifndef CAFFE_LAYERS_H
#define CAFFE_LAYERS_H
#include "ddr_functions_types.h"
#include "network_defines.h"

class Layers{
public:
	inline Layers(){}
	virtual ~Layers(){}
	virtual u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width){
		(void)bottom_output_buffer;
		(void)bottom_channels;
		(void)bottom_input_height;
		(void)bottom_input_width;
		return 0;
	}

	#ifdef PROFILE
		virtual u8 get_layer_type(){return -1;}
		#if CONVOLUTION
		virtual u8* get_weight_pointer(){return NULL;}
		virtual u8* get_bias_pointer(){return NULL;}
		virtual u8 get_kernel_size(){return -1;}
		virtual u8 get_stride(){return -1;}
		virtual u8 get_pad(){return -1;}
		virtual u8 get_group(){return -1;}
		virtual u8 get_ReLU_flag(){return -1;}
		#endif
		#if POOLING
		virtual pooling_type get_pooling_method(){ return pooling_INVALID; }
		#endif
		#if LRN
		virtual u8 get_local_size(){return -1;}
		virtual fp16 get_alpha(){return -1;}
		virtual fp16 get_beta(){return -1;}
		#endif
	#endif
	
	friend class Network_Manager;

protected:
	u8 *output_buffer;
	u8 ddr_function;
	u8 shaves_used;
	u16 bottom_node;
	u16 input_height;
	u16 input_width;
	u16 channels;
	#ifdef DUAL_CPU
		u8 event_handler;
	#elif defined(PROFILE)
		double *power_consumption[12];
		u64 *cpu_cycles[12];
	#endif
};

#if INPUT
class Input : public Layers{
public:
	#ifndef DUAL_CPU
		inline Input(u8 *output, u16 channel, u16 height, u16 width){
			
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
		}
	#else
		inline Input(u8 *output, u16 channel, u16 height, u16 width, u8 event){
			
			output_buffer = output;
			channels = channel;
			input_height = height;
			input_width = width;
			event_handler = event;
		}
	#endif
	~Input(){}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override{
		(void)bottom_output_buffer;
		(void)bottom_channels;
		(void)bottom_input_height;
		(void)bottom_input_width;
		return 0;
	}

	#ifdef PROFILE
	u8 get_layer_type() override{
		return layer_type;
	}
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 0;
	#endif
};
#endif

#if CONVOLUTION
class Convolution : public Layers{
public:
	#ifndef DUAL_CPU
		inline Convolution(u8 *output, u16 bottom, u8 shaves, u8 *weight, u8 *bias, 
							u16 channel, u16 height, u16 width, u8 k_size, u8 strid, 
									u8 padding, u8 grp, u8 function, u8 ReLU){
			
			output_buffer = output;
			bottom_node = bottom;
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
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[2];
					power_consumption[i] = new double[2];
				}
			#endif
		}
	#else
		inline Convolution(u8 *output, u16 bottom, u8 shaves, u8 *weight, u8 *bias, 
							u16 channel, u16 height, u16 width, u8 k_size, u8 strid, 
									u8 padding, u8 grp, u8 function, u8 event, u8 ReLU){
			
			output_buffer = output;
			bottom_node = bottom;
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
			event_handler = event;
		}
	#endif
	~Convolution(){
		#ifdef PROFILE
		for (int i = 0; i < 12; i++){
			delete [] cpu_cycles[i];
			delete [] power_consumption[i];
		}
		#endif
	}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override;

	#ifdef PROFILE
	u8 get_layer_type() override{ return layer_type; }
	u8 get_kernel_size() override{ return kernel_size; }
	u8 get_stride() override{ return stride; }
	u8 get_pad() override{ return pad; }
	u8 get_group() override{ return group; }
	u8 get_ReLU_flag() override{ return ReLU_flag; }
	u8* get_weight_pointer() override{ return weight_pointer; }
	u8* get_bias_pointer() override{ return bias_pointer; }
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 1;
	#endif
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
class Pooling : public Layers{
public:
	#ifndef DUAL_CPU
		inline Pooling(u8 *output, u16 bottom, u8 shaves, u16 channel, u16 height, u16 width, 
							u8 k_size, u8 strid, u8 padding, u8 function,
								pooling_type pool_m){
			
			output_buffer = output;
			bottom_node = bottom;
			shaves_used = shaves;
			channels = channel;
			input_height = height;
			input_width = width;
			kernel_size = k_size;
			stride = strid;
			pad = padding;
			ddr_function = function;
			pooling_method = pool_m;
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[1];
					power_consumption[i] = new double[1];
				}
			#endif
		}
	#else
		inline Pooling(u8 *output, u16 bottom, u8 shaves, u16 channel, u16 height, u16 width, 
							u8 k_size, u8 strid, u8 padding, u8 function,
								pooling_type pool_m, u8 event){
			
			output_buffer = output;
			bottom_node = bottom;
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
		}
	#endif
	~Pooling(){
		#ifdef PROFILE
		for (int i = 0; i < 12; i++){
			delete [] cpu_cycles[i];
			delete [] power_consumption[i];
		}
		#endif
	}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override;
	
	#ifdef PROFILE
	u8 get_layer_type() override{ return layer_type; }
	u8 get_kernel_size() override{ return kernel_size; }
	u8 get_stride() override{ return stride; }
	u8 get_pad() override{ return pad; }
	pooling_type get_pooling_method() override{ return pooling_method;}
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 2;
	#endif
	u8 kernel_size;	
	u8 stride;
	u8 pad;
	pooling_type pooling_method;
};
#endif

#if INNERPRODUCT
class InnerProduct : public Layers{
public:
	#ifndef DUAL_CPU
		inline InnerProduct(u8 *output, u16 bottom, u8 shaves, u8 *weight, u8 *bias, 
								u16 width, u8 ReLU){
			
			output_buffer = output;
			bottom_node = bottom;
			shaves_used = shaves;
			input_height = 1;
			input_width = width;
			channels = 1;
			weight_pointer = weight;
			bias_pointer = bias;
			ReLU_flag = ReLU;
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[1];
					power_consumption[i] = new double[1];
				}
			#endif
		}
	#else
		inline InnerProduct(u8 *output, u16 bottom, u8 shaves, u8 *weight, u8 *bias, 
								u16 width, u8 event, u8 ReLU){
			
			output_buffer = output;
			bottom_node = bottom;
			shaves_used = shaves;
			input_height = 1;
			input_width = width;
			channels = 1;
			weight_pointer = weight;
			bias_pointer = bias;
			event_handler = event;
			ReLU_flag = ReLU;
		}
	#endif
	~InnerProduct(){
		#ifdef PROFILE
		for (int i = 0; i < 12; i++){
			delete [] cpu_cycles[i];
			delete [] power_consumption[i];
		}
		#endif
	}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override;

	#ifdef PROFILE
	u8 get_layer_type() override{ return layer_type; }
	u8 get_ReLU_flag() override{ return ReLU_flag; }
	u8* get_weight_pointer() override{ return weight_pointer; }
	u8* get_bias_pointer() override{ return bias_pointer; }
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 3;
	#endif
	u8 *weight_pointer;
	u8 *bias_pointer;
	u8 ReLU_flag;
};
#endif

#if LRN
class Lrn : public Layers{
public:
	#ifndef DUAL_CPU
		inline Lrn(u8 *output, u16 bottom, u8 shaves, u8 l_size, fp16 a, fp16 b){

			output_buffer = output;
			bottom_node = bottom;
			shaves_used = shaves;
			ddr_function = 35;
			local_size = l_size;
			alpha = a;
			beta = b;
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[1];
					power_consumption[i] = new double[1];
				}
			#endif
		}
		inline Lrn(u8 *output, u16 bottom, u8 shaves){

			output_buffer = output;
			bottom_node = bottom;
			ddr_function = 36;
			shaves_used = shaves;
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[1];
					power_consumption[i] = new double[1];
				}
			#endif		
		}
	#else
		inline Lrn(u8 *output, u16 bottom, u8 shaves, u8 l_size, fp16 a, fp16 b, u8 event){

			output_buffer = output;
			bottom_node = bottom;
			shaves_used = shaves;
			ddr_function = 35;
			local_size = l_size;
			alpha = a;
			beta = b;
			event_handler = event;
		}
		inline Lrn(u8 *output, u16 bottom, u8 shaves, u8 event){

			output_buffer = output;
			bottom_node = bottom;
			ddr_function = 36;
			shaves_used = shaves;
			event_handler = event;
		}
	#endif
	~Lrn(){
		#ifdef PROFILE
		for (int i = 0; i < 12; i++){
			delete [] cpu_cycles[i];
			delete [] power_consumption[i];
		}
		#endif
	}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override;

	#ifdef PROFILE
	u8 get_layer_type() override{ return layer_type; }
	u8 get_local_size() override{ return local_size; }
	fp16 get_alpha() override{ return alpha; }
	fp16 get_beta() override{ return beta; }
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 4;
	#endif
	u8 local_size;
	fp16 alpha;
	fp16 beta;
};
#endif

#if !(LINEAR)
class Concat : public Layers{
public:
	#ifndef DUAL_CPU
		inline Concat(u8 *output, u16 channel, u16 height, u16 width){
			
			output_buffer = output;
			bottom_node = 0;
			channels = channel;
			input_height = height;
			input_width = width;
			#ifdef PROFILE
				for (int i = 0; i < 12; i++){
					cpu_cycles[i] = new u64[1];
					power_consumption[i] = new double[1];
				}
			#endif
		}
	#else
		inline Concat(u8 *output, u16 channel, u16 height, u16 width){
			
			output_buffer = output;
			bottom_node = 0;
			channels = channel;
			input_height = height;
			input_width = width;
		}
	#endif
	~Concat(){
		#ifdef PROFILE
		for (int i = 0; i < 12; i++){
			delete [] cpu_cycles[i];
			delete [] power_consumption[i];
		}
		#endif
	}
	u64 execute(u8 *bottom_output_buffer, u16 &bottom_channels, 
							u16 &bottom_input_height, u16 &bottom_input_width) override{
		(void)bottom_output_buffer;
		(void)bottom_channels;
		(void)bottom_input_height;
		(void)bottom_input_width;
		return 0;
	}
	
	#ifdef PROFILE
	u8 get_layer_type() override{ return layer_type; }
	#endif

private:
	#ifdef PROFILE
		const u8 layer_type = 5;
	#endif
};
#endif
#endif