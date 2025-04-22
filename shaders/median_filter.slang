extern static const uint MEDIAN_FILTER_WORKGROUP_SIZE_X;
extern static const uint MEDIAN_FILTER_WORKGROUP_SIZE_Y;
static const uint KERNEL_SIZE = 3;
static const int32_t RADIUS = (KERNEL_SIZE - 1) / 2;

import span;

groupshared uint16_t shared_tile[MEDIAN_FILTER_WORKGROUP_SIZE_Y + KERNEL_SIZE - 1][MEDIAN_FILTER_WORKGROUP_SIZE_X + KERNEL_SIZE - 1];

uint16_t sort9(uint32_t thread_tile_index_y, uint32_t thread_tile_index_x) {
    uint16_t window[9];

    uint32_t index = 0;
    for (int32_t dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int32_t dx = -RADIUS; dx <= RADIUS; ++dx) {
            window[index++] = shared_tile[thread_tile_index_y + dy][thread_tile_index_x + dx];
        }
    }

    #define SWAP(i, j)                  \
        if (window[i] > window[j]) {    \
            uint16_t tmp = window[i];   \
            window[i] = window[j];      \
            window[j] = tmp;            \
        }

    SWAP(1, 0);

    SWAP(2, 1); SWAP(1, 0);

    SWAP(4, 3); SWAP(3, 2); SWAP(2, 1); SWAP(1, 0);

    SWAP(5, 4); SWAP(4, 3); SWAP(3, 2); SWAP(2, 1); SWAP(1, 0);

    SWAP(6, 5); SWAP(5, 4); SWAP(4, 3); SWAP(3, 2); SWAP(2, 1); SWAP(1, 0);

    SWAP(7, 6); SWAP(6, 5); SWAP(5, 4); SWAP(4, 3); SWAP(3, 2); SWAP(2, 1); SWAP(1, 0);

    SWAP(8, 7); SWAP(7, 6); SWAP(6, 5); SWAP(5, 4); SWAP(4, 3); SWAP(3, 2); SWAP(2, 1); SWAP(1, 0);

    return window[4];
}

[shader("compute")]
[numthreads(MEDIAN_FILTER_WORKGROUP_SIZE_X, MEDIAN_FILTER_WORKGROUP_SIZE_Y, 1)]
void median_filter(
	uniform mdspan<uint16_t, 2> input,
	uniform mdspan<uint16_t, 2> output,
	uint3 group_id: SV_GroupID,
    uint3 global_id: SV_DispatchThreadID,
    uint3 group_thread_id: SV_GroupThreadID
) {
    if (global_id.x >= input.extents[1] || global_id.y >= input.extents[0])
        return;

    // Compute where this tile starts in global coords *including* the halo
    // So we subtract the radius from the group-based offset
    int32_t base_x = group_id.x * MEDIAN_FILTER_WORKGROUP_SIZE_X - RADIUS;
    int32_t base_y = group_id.y * MEDIAN_FILTER_WORKGROUP_SIZE_Y - RADIUS;

    // We need to fill shared_tile
    // We can do that by letting each thread load multiple pixels in steps of the workgroup size
    for (uint load_x = group_thread_id.x; load_x < MEDIAN_FILTER_WORKGROUP_SIZE_X + (KERNEL_SIZE - 1); load_x += MEDIAN_FILTER_WORKGROUP_SIZE_X) {
        for (uint load_y = group_thread_id.y; load_y < MEDIAN_FILTER_WORKGROUP_SIZE_Y + (KERNEL_SIZE - 1); load_y += MEDIAN_FILTER_WORKGROUP_SIZE_Y) {
            int32_t global_x = base_x + load_x;
            int32_t global_y = base_y + load_y;

            global_y = clamp(global_y, 0, int32_t(input.extents[0] - 1));
            global_x = clamp(global_x, 0, int32_t(input.extents[1] - 1));

            shared_tile[load_y][load_x] = input[ { global_y, global_x }];
		}
	}

    GroupMemoryBarrierWithGroupSync();

    uint32_t thread_tile_index_x = group_thread_id.x + RADIUS;
    uint32_t thread_tile_index_y = group_thread_id.y + RADIUS;

    output[ { global_id.y, global_id.x }] = sort9(thread_tile_index_y, thread_tile_index_x);
}
