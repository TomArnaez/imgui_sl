#pragma once

#include <vulkan_core.hpp>
#include <type_traits>

#include <slang-com-ptr.h>
#include <slang.h>

namespace vkengine {

struct shader_object {
    vk::PipelineLayout pipeline_layout;
    vk::PushConstantRange push_constant_range;
    vk::ShaderEXT shader_ext;
    vk::ShaderStageFlagBits stage;
};

class shader_manager {
public:
    shader_manager(std::reference_wrapper<vulkan_core> vulkan);

    shader_object load_shader(
        const std::string& module_name,
        const std::string& entry_point_name,
        const std::array<uint32_t, 3>& workgroup_sizes,
		const std::vector<slang::SpecializationArg> specialisation_args = {}
    );
private:
    void throw_exception_with_slang_diagnostics(const std::string& message, const std::source_location& location = std::source_location::current());

    void setup_slang();
    void create_subgroup_module();
    bool check_valid_workgroup_sizes(const std::array<uint32_t, 3>& workgroup_sizes);
    Slang::ComPtr<slang::IModule> create_workgroup_module(const std::array<uint32_t, 3>& workgroup_sizes);

    void log_scope(slang::VariableLayoutReflection* scope_variable_layout);
	void log_variable_layout(slang::VariableLayoutReflection* variable_layout);

    std::reference_wrapper<vulkan_core>     vulkan;

    Slang::ComPtr<slang::IBlob>             diagnostics;
    Slang::ComPtr<slang::IGlobalSession>    global_session;
    Slang::ComPtr<slang::ISession>          session;
    slang::SessionDesc                      session_desc;
    slang::TargetDesc                       target_desc;
    Slang::ComPtr<slang::IModule>           subgroup_module;
};

}