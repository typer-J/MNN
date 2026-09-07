//
//  MNNRvvC4Functions.hpp
//  MNN
//
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef MNN_RVV_C4_FUNCTIONS_HPP
#define MNN_RVV_C4_FUNCTIONS_HPP

#include <stddef.h>

// Keep distinct C++ symbols so the original C functions remain available as fallbacks.
// areaOffset contains source and destination area strides, in elements per channel.
void MNNPackC4_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNUnpackC4_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNScaleAndAddBias_RVV(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                            size_t biasNumber);
void MNNReluWithSlopeChannel_RVV(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

#endif // MNN_RVV_C4_FUNCTIONS_HPP
