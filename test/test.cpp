#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <allocator.hpp>
#include <algorithms/inclusive_scan.hpp>
#include <algorithms/histogram.hpp>
#include <typed_buffer.hpp>

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

    uint32_t element_count = 100;
    vkengine::typed_buffer<float> buffer(
        allocator,
        core,
        100,
        host_alloc_info
    );

    std::vector<float> source_data(element_count);
    std::ranges::generate(source_data, [n = 0]() mutable { return n++ * 1.5f; });
    buffer.copy_from_host(source_data);
    float* mapped = buffer.mapping();
    for (uint32_t i = 0; i < element_count; i++) {
        assert(mapped[i] == source_data[i] && "Data mismatch");
    }

    buffer.destroy();
}