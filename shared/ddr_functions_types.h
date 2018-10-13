#ifndef __DDR_FUNCTIONS_TYPES_H__
#define __DDR_FUNCTIONS_TYPES_H__

enum data4_datatype
{
    __data4_datatype__ = -1,
    xhhx = 0x01020201,
    hhhh = 0x02020202,
    hhhx = 0x02020201,
    xhhh = 0x01020202,
    xxhh = 0x01010202,
    hhff = 0x02020303
};

enum data2_datatype
{
    __data2_datatype__ = -1,
    xx = 0x00000101,
    xh = 0x00000102,
    hx = 0x00000201,
    hh = 0x00000202
};

enum kernel_size
{
    __kernel_size__ = -1,
    k1x1 = 1,
    k2x2 = 2,
    k3x3 = 3,
    k5x5 = 5,
    k7x7 = 7,
    k9x9 = 9,
    k11x11 = 11,
    k14x14 = 14
};

enum kernel_stride
{
    __kernel_stride__ = -1,
    stride1 = 1,
    stride2 = 2,
    stride3 = 3,
    stride4 = 4,
    stride8 = 8
};

enum pooling_type
{
    pooling_AVE,
    pooling_MAX,
    pooling_STOCHASTIC,
    pooling_INVALID
};

#endif//__DDR_FUNCTIONS_TYPES_H__
