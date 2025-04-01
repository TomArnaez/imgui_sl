import span;

extern const static uint WORKGROUP_SIZE_X;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, 1, 1)]
void caculate_histogram<T : __BuiltinArithmeticType>(
    uniform span<uint> histogram,
    uniform span<T> input,
    uint3 global_thread_id: SV_DispatchThreadID,
) {
    uint global_thread_idx = global_thread_id.x;

    T x = input[global_thread_idx];
    __atomic_add(histogram[__arithmetic_cast<uint, T>(x)], 1);
}