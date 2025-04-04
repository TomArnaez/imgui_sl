#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <allocator.hpp>
#include <algorithms/inclusive_scan.hpp>
#include <algorithms/histogram.hpp>
#include <queue_family.hpp>
#include <typed_buffer.hpp>
#include <ranges>

int main() {
    spdlog::set_level(spdlog::level::debug);
    
    std::vector<const char*> device_extensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    };

    vkengine::vulkan_core core({}, device_extensions);
	vkengine::allocator allocator(core);
    vkengine::shader_manager shader_manager(core);

    auto& props = core.physical_device_properties();

    auto inclusive_scan_shader = shader_manager.load_shader(
        std::string(VKENGINE_SHADER_DIR) + "\\inclusive_scan.slang",
        "workgroup_inclusive_scan",
        {128, 1, 1}
    );

    VmaAllocationCreateInfo host_alloc_info = {};
    host_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    host_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    uint32_t element_count = 1024;

    vkengine::typed_buffer<uint32_t> buffer(
        allocator,
        core,
        100,
        host_alloc_info
    );

    vkengine::typed_buffer<uint32_t> group_sums(
        allocator,
        core,
        element_count,
        host_alloc_info
    );

    vkengine::typed_buffer<uint32_t> output(
        allocator,
        core,
        element_count,
        host_alloc_info
    );

    auto src = std::views::iota(0u, element_count);
	buffer.copy_from_host(src);

    buffer.destroy();
    group_sums.destroy();
	output.destroy();
}