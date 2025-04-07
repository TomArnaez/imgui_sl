extern const static uint WORKGROUP_SIZE_X;

struct span<T> {
    private T *data_;
    private uint size_;

    __subscript(uint x)->T {
        get { return data_[x]; }
        set { data_[x] = newValue; }
    }

    property uint size {
        get { return size_; }
    }
};

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