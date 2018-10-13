#include "mv_types.h"
#include "network.h"
#include "weight_data.h"
#include "ddr_functions_types.h"

std::vector<Layers *> create_network(){

    std::vector<Layers *> network_vector;

    network_vector.emplace_back(new Input((u8*)(&data_input), 3, 224, 224));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0), 0, 11, (u8*)(&conv1_7x7_s2_weights), (u8*)(&conv1_7x7_s2_biases), 64, 112, 112, 7, 2, 3, 1, 12, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_0_1), 1, 9, 64, 56, 56, 3, 2, 0, 33, pooling_MAX));

    network_vector.emplace_back(new Lrn((u8*)(&branch_output_buffer_0_0), 2, 12));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1), 3, 12, (u8*)(&conv2_3x3_reduce_weights), (u8*)(&conv2_3x3_reduce_biases), 64, 56, 56, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0), 4, 12, (u8*)(&conv2_3x3_weights), (u8*)(&conv2_3x3_biases), 192, 56, 56, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Lrn((u8*)(&branch_output_buffer_0_1), 5, 12));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_0_0), 6, 11, 192, 28, 28, 3, 2, 0, 33, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[0]), 7, 12, (u8*)(&inception_3a_1x1_weights), (u8*)(&inception_3a_1x1_biases), 64, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 7, 12, (u8*)(&inception_3a_3x3_reduce_weights), (u8*)(&inception_3a_3x3_reduce_biases), 96, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[100352]), 9, 12, (u8*)(&inception_3a_3x3_weights), (u8*)(&inception_3a_3x3_biases), 128, 28, 28, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 7, 8, (u8*)(&inception_3a_5x5_reduce_weights), (u8*)(&inception_3a_5x5_reduce_biases), 16, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[301056]), 11, 12, (u8*)(&inception_3a_5x5_weights), (u8*)(&inception_3a_5x5_biases), 32, 28, 28, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 7, 12, 192, 28, 28, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[351232]), 13, 12, (u8*)(&inception_3a_pool_proj_weights), (u8*)(&inception_3a_pool_proj_biases), 32, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_1), 256, 28, 28));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[0]), 15, 12, (u8*)(&inception_3b_1x1_weights), (u8*)(&inception_3b_1x1_biases), 128, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 15, 12, (u8*)(&inception_3b_3x3_reduce_weights), (u8*)(&inception_3b_3x3_reduce_biases), 128, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[200704]), 17, 12, (u8*)(&inception_3b_3x3_weights), (u8*)(&inception_3b_3x3_biases), 192, 28, 28, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 15, 11, (u8*)(&inception_3b_5x5_reduce_weights), (u8*)(&inception_3b_5x5_reduce_biases), 32, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[501760]), 19, 12, (u8*)(&inception_3b_5x5_weights), (u8*)(&inception_3b_5x5_biases), 96, 28, 28, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 15, 9, 256, 28, 28, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[652288]), 21, 12, (u8*)(&inception_3b_pool_proj_weights), (u8*)(&inception_3b_pool_proj_biases), 64, 28, 28, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_0), 480, 28, 28));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_0_1), 23, 12, 480, 14, 14, 3, 2, 0, 33, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[0]), 24, 12, (u8*)(&inception_4a_1x1_weights), (u8*)(&inception_4a_1x1_biases), 192, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 24, 12, (u8*)(&inception_4a_3x3_reduce_weights), (u8*)(&inception_4a_3x3_reduce_biases), 96, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[75264]), 26, 12, (u8*)(&inception_4a_3x3_weights), (u8*)(&inception_4a_3x3_biases), 208, 14, 14, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 24, 9, (u8*)(&inception_4a_5x5_reduce_weights), (u8*)(&inception_4a_5x5_reduce_biases), 16, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[156800]), 28, 12, (u8*)(&inception_4a_5x5_weights), (u8*)(&inception_4a_5x5_biases), 48, 14, 14, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 24, 12, 480, 14, 14, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[175616]), 30, 11, (u8*)(&inception_4a_pool_proj_weights), (u8*)(&inception_4a_pool_proj_biases), 64, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_0), 512, 14, 14));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[0]), 32, 12, (u8*)(&inception_4b_1x1_weights), (u8*)(&inception_4b_1x1_biases), 160, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 32, 12, (u8*)(&inception_4b_3x3_reduce_weights), (u8*)(&inception_4b_3x3_reduce_biases), 112, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[62720]), 34, 12, (u8*)(&inception_4b_3x3_weights), (u8*)(&inception_4b_3x3_biases), 224, 14, 14, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 32, 8, (u8*)(&inception_4b_5x5_reduce_weights), (u8*)(&inception_4b_5x5_reduce_biases), 24, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[150528]), 36, 12, (u8*)(&inception_4b_5x5_weights), (u8*)(&inception_4b_5x5_biases), 64, 14, 14, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 32, 12, 512, 14, 14, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[175616]), 38, 12, (u8*)(&inception_4b_pool_proj_weights), (u8*)(&inception_4b_pool_proj_biases), 64, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_1), 512, 14, 14));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[0]), 40, 12, (u8*)(&inception_4c_1x1_weights), (u8*)(&inception_4c_1x1_biases), 128, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 40, 12, (u8*)(&inception_4c_3x3_reduce_weights), (u8*)(&inception_4c_3x3_reduce_biases), 128, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[50176]), 42, 12, (u8*)(&inception_4c_3x3_weights), (u8*)(&inception_4c_3x3_biases), 256, 14, 14, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 40, 12, (u8*)(&inception_4c_5x5_reduce_weights), (u8*)(&inception_4c_5x5_reduce_biases), 24, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[150528]), 44, 12, (u8*)(&inception_4c_5x5_weights), (u8*)(&inception_4c_5x5_biases), 64, 14, 14, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 40, 11, 512, 14, 14, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[175616]), 46, 12, (u8*)(&inception_4c_pool_proj_weights), (u8*)(&inception_4c_pool_proj_biases), 64, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_0), 512, 14, 14));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[0]), 48, 12, (u8*)(&inception_4d_1x1_weights), (u8*)(&inception_4d_1x1_biases), 112, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 48, 12, (u8*)(&inception_4d_3x3_reduce_weights), (u8*)(&inception_4d_3x3_reduce_biases), 144, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[43904]), 50, 12, (u8*)(&inception_4d_3x3_weights), (u8*)(&inception_4d_3x3_biases), 288, 14, 14, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 48, 10, (u8*)(&inception_4d_5x5_reduce_weights), (u8*)(&inception_4d_5x5_reduce_biases), 32, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[156800]), 52, 12, (u8*)(&inception_4d_5x5_weights), (u8*)(&inception_4d_5x5_biases), 64, 14, 14, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 48, 12, 512, 14, 14, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[181888]), 54, 11, (u8*)(&inception_4d_pool_proj_weights), (u8*)(&inception_4d_pool_proj_biases), 64, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_1), 528, 14, 14));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[0]), 56, 12, (u8*)(&inception_4e_1x1_weights), (u8*)(&inception_4e_1x1_biases), 256, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 56, 12, (u8*)(&inception_4e_3x3_reduce_weights), (u8*)(&inception_4e_3x3_reduce_biases), 160, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[100352]), 58, 12, (u8*)(&inception_4e_3x3_weights), (u8*)(&inception_4e_3x3_biases), 320, 14, 14, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 56, 12, (u8*)(&inception_4e_5x5_reduce_weights), (u8*)(&inception_4e_5x5_reduce_biases), 32, 14, 14, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[225792]), 60, 12, (u8*)(&inception_4e_5x5_weights), (u8*)(&inception_4e_5x5_biases), 128, 14, 14, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 56, 12, 528, 14, 14, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[275968]), 62, 12, (u8*)(&inception_4e_pool_proj_weights), (u8*)(&inception_4e_pool_proj_biases), 128, 14, 14, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_0), 832, 14, 14));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_0_1), 64, 12, 832, 7, 7, 3, 2, 0, 33, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[0]), 65, 12, (u8*)(&inception_5a_1x1_weights), (u8*)(&inception_5a_1x1_biases), 256, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 65, 12, (u8*)(&inception_5a_3x3_reduce_weights), (u8*)(&inception_5a_3x3_reduce_biases), 160, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[25088]), 67, 12, (u8*)(&inception_5a_3x3_weights), (u8*)(&inception_5a_3x3_biases), 320, 7, 7, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 65, 12, (u8*)(&inception_5a_5x5_reduce_weights), (u8*)(&inception_5a_5x5_reduce_biases), 32, 7, 7, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[56448]), 69, 12, (u8*)(&inception_5a_5x5_weights), (u8*)(&inception_5a_5x5_biases), 128, 7, 7, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 65, 11, 832, 7, 7, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_0[68992]), 71, 12, (u8*)(&inception_5a_pool_proj_weights), (u8*)(&inception_5a_pool_proj_biases), 128, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_0), 832, 7, 7));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[0]), 73, 12, (u8*)(&inception_5b_1x1_weights), (u8*)(&inception_5b_1x1_biases), 384, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_2_0), 73, 12, (u8*)(&inception_5b_3x3_reduce_weights), (u8*)(&inception_5b_3x3_reduce_biases), 192, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[37632]), 75, 12, (u8*)(&inception_5b_3x3_weights), (u8*)(&inception_5b_3x3_biases), 384, 7, 7, 3, 1, 1, 1, 1, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_3_0), 73, 11, (u8*)(&inception_5b_5x5_reduce_weights), (u8*)(&inception_5b_5x5_reduce_biases), 48, 7, 7, 1, 1, 0, 1, 46, 1));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[75264]), 77, 12, (u8*)(&inception_5b_5x5_weights), (u8*)(&inception_5b_5x5_biases), 128, 7, 7, 5, 1, 2, 1, 6, 1));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_4_0), 73, 9, 832, 7, 7, 3, 1, 1, 32, pooling_MAX));

    network_vector.emplace_back(new Convolution((u8*)(&branch_output_buffer_0_1[87808]), 79, 12, (u8*)(&inception_5b_pool_proj_weights), (u8*)(&inception_5b_pool_proj_biases), 128, 7, 7, 1, 1, 0, 1, 0, 1));

    network_vector.emplace_back(new Concat((u8*)(&branch_output_buffer_0_1), 1024, 7, 7));

    network_vector.emplace_back(new Pooling((u8*)(&branch_output_buffer_0_0), 81, 12, 1024, 1, 1, 7, 1, 0, 29, pooling_AVE));

    network_vector.emplace_back(new InnerProduct((u8*)(&branch_output_buffer_0_1), 82, 6, (u8*)(&loss3_classifier_weights), (u8*)(&loss3_classifier_biases), 1000, 0));

    return std::move(network_vector);
}