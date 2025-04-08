const static uint NORMALISE_WORKGROUP_SIZE_X = 128;

import span;

[shader("compute")]
[numthreads(NORMALISE_WORKGROUP_SIZE_X, 1, 1)]
void normalise(
	uniform span<uint32_t> input,
	uniform span<uint16_t> output,
    uniform uint32_t input_min,
    uniform uint32_t input_max,
    uniform uint16_t min,
	uniform uint16_t max,
    uint3 global_thread_id: SV_DispatchThreadID,
) {
    uint global_thread_idx = global_thread_id.x;
    uint32_t x = input[global_thread_idx];

    float scale = __arithmetic_cast<float, uint32_t>(x - input_min) / __arithmetic_cast<float, uint32_t>(input_max - input_min);
    uint32_t normalised_value = scale * __arithmetic_cast<uint32_t, uint16_t>(max - min) + __arithmetic_cast<uint32_t, uint16_t>(min);
    output[global_thread_idx] = __arithmetic_cast<uint32_t, uint16_t>(normalised_value);
}