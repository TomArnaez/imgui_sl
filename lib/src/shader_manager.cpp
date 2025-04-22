#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <shader_layout.hpp>
#include <spdlog/spdlog.h>
#include <detailed_exception.hpp>
#include <spdlog/fmt/ranges.h>
#include <ranges>
#include <fstream>


namespace vkengine {

shader_manager::shader_manager(std::reference_wrapper<vulkan_core> vulkan_core)
    : vulkan(vulkan_core) {
    setup_slang_session();
    create_subgroup_module();
}

void shader_manager::throw_exception_with_slang_diagnostics(
    const std::string& base_message,
    const std::source_location& location
) {
    std::string full_message = base_message;

    if (diagnostics != nullptr)
        if (const char* diagnostic_msg = static_cast<const char*>(diagnostics->getBufferPointer()))
            if (strlen(diagnostic_msg) > 0)
                full_message += fmt::format("\nSlang diagnostics:\n{}", diagnostic_msg);

    throw detailed_exception(location, full_message);
}

Slang::ComPtr<slang::IModule> shader_manager::create_shader_module_from_source_string(
    const std::string& source_string,
    const std::string& module_name
) {
	spdlog::info("Creating shader module from source string: {}", module_name);
	spdlog::debug("Source string: {}", source_string);

    std::string module_path = fmt::format("{}.slang", module_name);

	Slang::ComPtr<slang::IModule> module(
        session->loadModuleFromSourceString(
		module_name.c_str(),
        module_path.c_str(),
		source_string.c_str(),
		diagnostics.writeRef())
    );

	if (!module)
		throw detailed_exception("Failed to create shader module from source string");

	return module;
}

shader_program shader_manager::load_shader(
    const std::string& module_name,
    const std::vector<entry_point_compile_info>& entry_point_infos,
    const std::vector<Slang::ComPtr<slang::IModule>> modules
) {
    spdlog::info("Loading shader: {}", module_name);

    Slang::ComPtr<slang::IModule> module(session->loadModule(module_name.c_str(), diagnostics.writeRef()));

    if (!module)
        throw_exception_with_slang_diagnostics("Failed to create module");

	auto module_layout = module->getLayout();

    auto entry_points = entry_point_infos
        | std::views::transform(
            [&](const entry_point_compile_info& entry_point_info) {
                Slang::ComPtr<slang::IComponentType> entry_point;
                module->findEntryPointByName(entry_point_info.name.c_str(), reinterpret_cast<slang::IEntryPoint**>(entry_point.writeRef()));

                if (!entry_point)
                    throw_exception_with_slang_diagnostics("Failed to find entry point: " + entry_point_info.name);

                if (!entry_point_info.specialisation_type_names.empty()) {
                    Slang::ComPtr<slang::IComponentType> specialised_entry_point;

                    spdlog::debug("Specialising args for entry point {}", entry_point_info.name);

                    auto slang_specialisation_args = std::views::transform(
                            entry_point_info.specialisation_type_names,
                            [&](const std::string& type_name) {
                                return slang::SpecializationArg::fromType(module_layout->findTypeByName(type_name.c_str()));
                            }) | std::ranges::to<std::vector>();

					entry_point->specialize(
						slang_specialisation_args.data(),
						slang_specialisation_args.size(),
						specialised_entry_point.writeRef(),
						diagnostics.writeRef()
					);

					if (!specialised_entry_point)
						throw_exception_with_slang_diagnostics("Failed to specialise entry point: " + entry_point_info.name);

					entry_point = specialised_entry_point.get();
                }
                return entry_point;
            })
        | std::ranges::to<std::vector>();

    std::vector<slang::IComponentType*> components = { module, subgroup_module };
	components.insert(components.end(), entry_points.begin(), entry_points.end());
	components.insert(components.end(), modules.begin(), modules.end());

    Slang::ComPtr<slang::IComponentType> program = nullptr;
    session->createCompositeComponentType(components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    if (!program)
        throw_exception_with_slang_diagnostics("Failed to create slang program");

    Slang::ComPtr<slang::IComponentType> linked_program;
	program->link(linked_program.writeRef(), diagnostics.writeRef());

	if (!linked_program)
		throw_exception_with_slang_diagnostics("Failed to link program");

    slang::ProgramLayout* layout = linked_program->getLayout();

    vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eCompute;

    Slang::ComPtr<slang::IBlob> spirv_code;
    linked_program->getTargetCode(0, spirv_code.writeRef(), diagnostics.writeRef());

    if (!spirv_code)
        throw_exception_with_slang_diagnostics("Failed to create spirv code");

	slang::ProgramLayout* program_layout = linked_program->getLayout();

    auto entry_point_count = program_layout->getEntryPointCount();
    root_shader_layout_builder builder;

    builder.add_global_params(program_layout->getGlobalParamsVarLayout(), vulkan);
    for (uint32_t idx : std::views::iota(0u) | std::views::take(entry_point_count))
        builder.add_entry_point(program_layout->getEntryPointByIndex(idx), vulkan);

    root_shader_object_layout root_layout = builder.build();

    auto shader_objects = std::views::iota(0u, entry_point_count) | std::views::transform([&](uint32_t idx) {
        Slang::ComPtr<slang::IBlob> entry_point_code;
        linked_program->getEntryPointCode(
            idx, 0,
            entry_point_code.writeRef(),
            diagnostics.writeRef());

        if (!entry_point_code)
            throw_exception_with_slang_diagnostics("Failed to create entry point code");

        auto set_layouts = root_layout.entry_point_descriptor_sets(idx) | std::ranges::to<std::vector<vk::DescriptorSetLayout>>();
        auto& push_constants = root_layout.entry_push_constants(idx);

        auto pipeline_layout = vulkan.get().device()
            .createPipelineLayout(
                vk::PipelineLayoutCreateInfo{}
                .setSetLayouts(set_layouts)
                .setPushConstantRanges(push_constants)
            );

        auto shader_obj = vulkan.get().device()
            .createShaderEXT(
                vk::ShaderCreateInfoEXT{}
                .setStage(stage)
                .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
                .setCodeSize(spirv_code->getBufferSize())
                .setPName(root_layout.entry_points[idx].name.c_str())
                .setPCode(static_cast<const uint32_t*>(spirv_code->getBufferPointer()))
                .setPushConstantRanges(push_constants)
                .setSetLayouts(set_layouts)
            );

        if (shader_obj.result != vk::Result::eSuccess)
            throw detailed_exception("Failed to create shader object");

        return shader_entry_point {
            .pipeline_layout = pipeline_layout,
            .push_constant_range = push_constants[0],
            .shader_ext = shader_obj.value,
            .stage = stage
        };
    });

    return shader_program {
        .root_layout = root_layout,
        .entry_points = shader_objects | std::ranges::to<std::vector<shader_entry_point>>()
    };
}

void shader_manager::setup_slang_session() {
    slang::createGlobalSession(global_session.writeRef());

    target_desc = {
        .format = SLANG_SPIRV,
        .profile = global_session->findProfile("spirv_1_6"),
        .forceGLSLScalarBufferLayout = false
    };

    std::vector<slang::CompilerOptionEntry> compiler_option_entries = {
        {
            .name = slang::CompilerOptionName::VulkanUseEntryPointName,
            .value = {.kind = slang::CompilerOptionValueKind::Int, .intValue0 = 1}
        },
        {
            .name = slang::CompilerOptionName::GLSLForceScalarLayout,
            .value = {.kind = slang::CompilerOptionValueKind::Int, .intValue0 = 0}
        }
    };

    std::string shader_dir(VKENGINE_SHADER_DIR);
	std::array<const char*, 1> search_paths = { shader_dir.c_str() };

    session_desc = {
        .targets = &target_desc,
        .targetCount = 1,
		.searchPaths = search_paths.data(),
        .searchPathCount = search_paths.size(),
        .compilerOptionEntries = compiler_option_entries.data(),
        .compilerOptionEntryCount = static_cast<uint32_t>(compiler_option_entries.size()),
    };

    global_session->createSession(session_desc, session.writeRef());
}

void shader_manager::create_subgroup_module() {
    std::string subgroup_module_src = fmt::format("export static const uint SUBGROUP_SIZE = {};", vulkan.get().gpu().subgroup_properties.subgroupSize);
    subgroup_module = session->loadModuleFromSourceString("subgroup_size", "subgroup_size.slang", subgroup_module_src.c_str());

    if (!subgroup_module)
        throw std::runtime_error("Failed to create subgroup module");
}

}