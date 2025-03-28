#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <allocator.hpp>
#include <algorithms/inclusive_scan.hpp>
#include <typed_buffer.hpp>
#include <vulkan_core.hpp>
#include <iostream>

int main() {
    spdlog::set_level(spdlog::level::debug);
    
    std::vector<const char*> device_extensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    };

    vkengine::vulkan_core core({}, device_extensions);
	vkengine::allocator allocator(core);

    vkengine::shader_manager shader_manager(core);

    vkengine::typed_buffer<float> buffer(allocator, core, 1024);

	std::cout << buffer.device_address() << std::endl;

    buffer.destroy_buffer();

    auto& props = core.physical_device_properties();

    auto inclusive_scan_shader = shader_manager.load_shader(
        "C:\\dev\\repos\\imgui_test\\shaders\\inclusive_scan.slang",
        "workgroup_inclusive_scan",
        {128, 1, 1}
    );


}