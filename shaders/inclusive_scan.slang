extern const static uint SUBGROUP_SIZE;
extern const static uint INCLUSIVE_SCAN_WORKGROUP_SIZE;

import span;

groupshared uint shared_data[INCLUSIVE_SCAN_WORKGROUP_SIZE / SUBGROUP_SIZE];

[shader("compute")]
[numthreads(INCLUSIVE_SCAN_WORKGROUP_SIZE, 1, 1)]
void workgroup_inclusive_scan(
    uniform span<uint> input_span,
    uniform span<uint> output,
    uniform span<uint> group_sums,
    uint3 group_id: SV_GroupID,
    uint3 global_thread_id: SV_DispatchThreadID,
    uint group_thread_id: SV_GroupThreadID
) {
    uint global_thread_idx = global_thread_id.x;

    uint x = input_span[global_thread_idx];

    uint subgroup_prefix_sum = WavePrefixSum(x) + x;

    uint lane_index = WaveGetLaneIndex();
    uint wave_id = group_thread_id.x / WaveGetLaneCount();
    if (lane_index == WaveGetLaneCount() - 1)
        shared_data[wave_id] = subgroup_prefix_sum;

    GroupMemoryBarrierWithGroupSync();

    uint reduced = (lane_index < wave_id) ? shared_data[lane_index] : 0;

    uint prefix = WaveActiveSum(reduced);

    x = prefix + subgroup_prefix_sum;

    output[global_thread_idx] = x;

    if (group_thread_id.x == INCLUSIVE_SCAN_WORKGROUP_SIZE - 1)
        group_sums[group_id.x] = x;
}

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void subgroup_exclusive_scan(
    uniform span<uint> group_sums,
    uint3 group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID
) {
    uint idx = group_thread_id.x;

    if (idx >= group_sums.size) return;

    group_sums[idx] = WavePrefixSum(group_sums[idx]);
}

[shader("compute")]
[numthreads(INCLUSIVE_SCAN_WORKGROUP_SIZE, 1, 1)]
void propogate_group_sums(
    uniform span<uint> input_span,
    uniform span<uint> output,
    uniform span<uint> group_sums,
    uint3 group_id: SV_GroupID,
    uint3 global_thread_id: SV_DispatchThreadID,
) {
    output[global_thread_id.x] += group_sums[group_id.x];
}