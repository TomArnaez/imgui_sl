extern static const uint MEDIAN_FILTER_WORKGROUP_SIZE_X;
extern static const uint MEDIAN_FILTER_WORKGROUP_SIZE_Y;
static const uint KERNEL_SIZE = 3;

import span;

groupshared uint16_t shared_tile[MEDIAN_FILTER_WORKGROUP_SIZE_X + KERNEL_SIZE - 1][MEDIAN_FILTER_WORKGROUP_SIZE_Y + KERNEL_SIZE - 1];

void compare_and_swap<T: IComparable>(uint32_t i, uint32_t j, T* data) {
    if (data[i] > data[j]) {
        T tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

void sort9<T : IComparable>(T *data) {
    compare_and_swap(1, 0, data);

    compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);

    compare_and_swap(4, 3, data); compare_and_swap(3, 2, data); compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);

    compare_and_swap(5, 4, data); compare_and_swap(4, 3, data); compare_and_swap(3, 2, data); compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);

    compare_and_swap(6, 5, data); compare_and_swap(5, 4, data); compare_and_swap(4, 3, data); compare_and_swap(3, 2, data); compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);

    compare_and_swap(7, 6, data); compare_and_swap(6, 5, data); compare_and_swap(5, 4, data); compare_and_swap(4, 3, data); compare_and_swap(3, 2, data); compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);

    compare_and_swap(8, 7, data); compare_and_swap(7, 6, data); compare_and_swap(6, 5, data); compare_and_swap(5, 4, data); compare_and_swap(4, 3, data); compare_and_swap(3, 2, data); compare_and_swap(2, 1, data); compare_and_swap(1, 0, data);
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

    //output[ { global_id.y, global_id.x }] = 1;

 //   static const uint RADIUS = (KERNEL_SIZE - 1) / 2;

 //   // Compute where this tile starts in global coords *including* the halo
 //   // So we subtract the radius from the group-based offset
 //   uint base_x = group_id.x * MEDIAN_FILTER_WORKGROUP_SIZE_X - RADIUS;
 //   uint base_y = group_id.y * MEDIAN_FILTER_WORKGROUP_SIZE_Y - RADIUS;

 //   // We need to fill shared_tile
 //   // We can do that by letting each thread load multiple pixels in steps of the workgroup size
 //   for (uint load_x = group_thread_id.x; load_x < MEDIAN_FILTER_WORKGROUP_SIZE_X + (KERNEL_SIZE - 1); load_x += MEDIAN_FILTER_WORKGROUP_SIZE_X) {
 //       for (uint load_y = group_thread_id.y; load_y < MEDIAN_FILTER_WORKGROUP_SIZE_Y + (KERNEL_SIZE - 1); load_y += MEDIAN_FILTER_WORKGROUP_SIZE_Y) {
 //           uint global_x = base_x + load_x;
 //           uint global_y = base_y + load_y;

 //           global_y = clamp(global_y, 0, input.extents[0] - 1);
 //           global_x = clamp(global_x, 0, input.extents[1] - 1);

 //           shared_tile[load_y][load_x] = input[ { global_y, global_x }];
	//	}
	//}

 //   GroupMemoryBarrierWithGroupSync();

 //   uint32_t thread_tile_index_x = group_thread_id.x + RADIUS;
 //   uint32_t thread_tile_index_y = group_thread_id.y + RADIUS;

 //   uint16_t window[KERNEL_SIZE*KERNEL_SIZE];

 //   uint32_t index = 0;
 //   for (uint32_t dy = -RADIUS; dy <= RADIUS; ++dy)
 //       for (uint32_t dx = -RADIUS; dx <= RADIUS; ++dx)
 //           window[index++] = shared_tile[thread_tile_index_y + dy][thread_tile_index_x + dx];

 //   sort9<uint16_t>(&window[0]);
 //   output[ { global_id.y, global_id.x }] = window[4];
}