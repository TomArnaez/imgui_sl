extern static const uint MEDIAN_FILTER_WORKGROUP_X;
extern static const uint MEDIAN_FILTER_WORKGROUP_Y;

import span;

[shader("compute")]
[numthreads(MEDIAN_FILTER_WORKGROUP_X, MEDIAN_FILTER_WORKGROUP_Y, 1)]
void median_filter(
	uniform mdspan<uint16_t, 2> input,
	uniform mdspan<uint16_t, 2> output
) {

}