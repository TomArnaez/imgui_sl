const static uint NORMALISE_WORKGROUP_SIZE_X = 128;

import span;

//[shader("compute")]
//[numthreads(NORMALISE_WORKGROUP_SIZE_X, 1, 1)]
//void normalise<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType>(
//	uniform span<T> input,
//	uniform span<U> output,
//	uniform T input_min,
//    uniform T input_max,
//    uniform U min,
//	uniform U max,
//    uint3 global_thread_id: SV_DispatchThreadID,
//) {
//    uint global_thread_idx = global_thread_id.x;
//    T x = input[global_thread_idx];
//    T normalised_value = (x - input_min) / (input_max - input_min) * __arithmetic_cast<T, U>(max - min) + __arithmetic_cast<T, U>(min);
//    output[global_thread_idx] = __arithmetic_cast<U, T>(normalised_value);
//}

[shader("compute")]
[numthreads(NORMALISE_WORKGROUP_SIZE_X, 1, 1)]
void normalise(
	uniform span<uint> input,
	uniform span<uint16_t> output,
    uniform uint input_min,
    uniform uint input_max,
    uniform uint16_t min,
	uniform uint16_t max,
    uint3 global_thread_id: SV_DispatchThreadID,
) {
    //uint global_thread_idx = global_thread_id.x;
    //uint32_t x = input[global_thread_idx];
    //uint32_t normalised_value = (x - input_min) / (input_max - input_min) * __arithmetic_cast<uint32_t, uint16_t>(max - min) + __arithmetic_cast<uint32_t, uint16_t>(min);
    //output[global_thread_idx] = __arithmetic_cast<uint32_t, uint32_t>(normalised_value);
}