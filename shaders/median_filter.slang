extern static const uint MEDIAN_FILTER_WORKGROUP_X;
extern static const uint MEDIAN_FILTER_WORKGROUP_Y;
extern static const uint HALO_RADIUS;

import span;

groupshared uint16_t shared_tile[MEDIAN_FILTER_WORKGROUP_X + HALO_RADIUS][MEDIAN_FILTER_WORKGROUP_X + HALO_RADIUS];

[shader("compute")]
[numthreads(MEDIAN_FILTER_WORKGROUP_X, MEDIAN_FILTER_WORKGROUP_Y, 1)]
void median_filter(
	uniform mdspan<uint16_t, 2> input,
	uniform mdspan<uint16_t, 2> output,
	uint3 group_id: SV_GroupID,
    uint3 global_id: SV_DispatchThreadID
) {

}