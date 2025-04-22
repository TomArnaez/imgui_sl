#pragma once

#include <shader_layout.hpp>
#include <vulkan_core.hpp>
#include <type_traits>

#include <slang-com-ptr.h>
#include <slang.h>
#include <source_location>

namespace vkengine {

struct shader_entry_point {
    vk::PipelineLayout      pipeline_layout;
    vk::PushConstantRange   push_constant_range;
    vk::ShaderEXT           shader_ext;
    vk::ShaderStageFlagBits stage;
};

struct shader_program {
    root_shader_object_layout       root_layout;
    std::vector<shader_entry_point> entry_points;
};

class shader_manager {
public:
    struct entry_point_compile_info {
		std::string                 name;
        std::vector<std::string>    specialisation_type_names;
    };

    shader_manager(std::reference_wrapper<vulkan_core> vulkan);

    Slang::ComPtr<slang::IModule> create_shader_module_from_source_string(
        const std::string& source_string,
        const std::string& module_name
    );
    shader_program load_shader(
        const std::string& module_name,
        const std::vector<entry_point_compile_info>& entry_point_infos,
        const std::vector<Slang::ComPtr<slang::IModule>> modules
    );
private:
    void throw_exception_with_slang_diagnostics(const std::string& message, const std::source_location& location = std::source_location::current());

    void setup_slang_session();
    void create_subgroup_module();

    std::reference_wrapper<vulkan_core>     vulkan;

    Slang::ComPtr<slang::IBlob>             diagnostics;
    Slang::ComPtr<slang::IGlobalSession>    global_session;
    Slang::ComPtr<slang::ISession>          session;
    slang::SessionDesc                      session_desc;
    slang::TargetDesc                       target_desc;
    Slang::ComPtr<slang::IModule>           subgroup_module;
};

}