extern const static uint32_t HISTOGRAM_WORKGROUP_SIZE_X;

import span;

[shader("compute")]
[numthreads(HISTOGRAM_WORKGROUP_SIZE_X, 1, 1)]
void caculate_histogram(
    uniform span<uint32_t> histogram,
    uniform span<uint16_t> input,
    uint3 global_thread_id: SV_DispatchThreadID,
) {
    uint global_thread_idx = global_thread_id.x;

    uint16_t x = input[global_thread_idx];
    __atomic_add(histogram[__arithmetic_cast<uint32_t, uint16_t>(x)], 1);
}