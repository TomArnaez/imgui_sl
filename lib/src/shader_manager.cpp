#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <shader_layout.hpp>
#include <spdlog/spdlog.h>
#include <detailed_exception.hpp>
#include <spdlog/fmt/ranges.h>
#include <ranges>

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

std::vector<shader_object> shader_manager::load_shader(
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
    slang::EntryPointReflection* entry_point_reflection = layout->getEntryPointByIndex(0);

    vk::ShaderStageFlagBits stage;

    switch (entry_point_reflection->getStage()) {
    case SLANG_STAGE_COMPUTE:
        stage = vk::ShaderStageFlagBits::eCompute;
        break;
    default:
        throw detailed_exception("Unsupported shader stage");
    }

    Slang::ComPtr<slang::IBlob> spirv_code;
    linked_program->getTargetCode(0, spirv_code.writeRef(), diagnostics.writeRef());

    if (!spirv_code)
        throw_exception_with_slang_diagnostics("Failed to create spirv code");

	slang::ProgramLayout* program_layout = linked_program->getLayout();

    shader_layout shader_layout = create_pipeline_layout(program_layout, vulkan.get());

    for (auto* reflection : std::views::iota(0u, program_layout->getEntryPointCount())
        | std::views::transform([=, this](auto i) {return program_layout->getEntryPointByIndex(i); })) {
		log_scope(reflection->getVarLayout());
    }

    slang::TypeLayoutReflection* type_layout = entry_point_reflection->getTypeLayout();

    uint32_t param_count = entry_point_reflection->getParameterCount();

    for (uint32_t param_idx = 0; param_idx < param_count; ++param_idx) {
        slang::VariableLayoutReflection* param = entry_point_reflection->getParameterByIndex(param_idx);

        slang::TypeLayoutReflection*    type_layout = param->getTypeLayout();
        slang::ParameterCategory        category = param->getCategory();
        uint32_t                        category_count = param->getCategoryCount();

        spdlog::debug(
            "Param at index: {}, name: {}, category_count: {}, category: {}, offset: {}, alignment: {}, size: {}",
            param_idx,
            param->getName(),
            category_count,
            static_cast<uint32_t>(category),
            param->getOffset(),
            type_layout->getAlignment(),
            type_layout->getSize()
        );
    }

    std::vector<shader_object> shader_objects;

	for (auto [index, entry_point_layout] : std::views::enumerate(shader_layout.entry_point_layouts)) {
        Slang::ComPtr<slang::IBlob> entry_point_code;
        linked_program->getEntryPointCode(
            index,
            0,
            entry_point_code.writeRef(),
            diagnostics.writeRef()
        );

		if (!entry_point_code)
			throw_exception_with_slang_diagnostics("Failed to create entry point code");

        vk::ResultValue<vk::ShaderEXT> shader_obj = vulkan.get().device().createShaderEXT(
            vk::ShaderCreateInfoEXT()
            .setStage(stage)
            .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
            .setCodeSize(spirv_code->getBufferSize())
            .setPName(entry_point_layout.name.c_str())
            .setPCode(static_cast<const uint32_t*>(spirv_code->getBufferPointer()))
            .setPushConstantRanges(entry_point_layout.push_constant_ranges)
        );

        if (shader_obj.result != vk::Result::eSuccess)
            throw detailed_exception("Failed to create shader object");

        shader_objects.emplace_back(shader_object{
            .pipeline_layout = entry_point_layout.pipeline_layout,
            .push_constant_range = entry_point_layout.push_constant_ranges[0],
            .shader_ext = shader_obj.value,
            .stage = stage
            });
	}

    return shader_objects;
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

void shader_manager::log_scope(slang::VariableLayoutReflection* scope_variable_layout) {
    auto type_layout = scope_variable_layout->getTypeLayout();

    switch (type_layout->getKind()) {
    case slang::TypeReflection::Kind::Struct:
    {
        spdlog::debug("Scope type is struct");
        for (auto* param : std::views::iota(0u, type_layout->getFieldCount())| std::views::transform([&](auto i) {return type_layout->getFieldByIndex(i); })) {
            log_variable_layout(param);
        }
        break;
    }
    case slang::TypeReflection::Kind::ConstantBuffer:
        spdlog::debug("Scope type is constant buffer");
        log_scope(type_layout->getElementVarLayout());
        break;
    case slang::TypeReflection::Kind::ParameterBlock:
        spdlog::debug("Scope type is parameter block");
        break;
    default:
        spdlog::debug("Scope type is unknown");
    }
}

void shader_manager::log_variable_layout(slang::VariableLayoutReflection* variable_layout) {
	auto type_layout = variable_layout->getTypeLayout();
    
	spdlog::debug("Variable name: {}", variable_layout->getName());
	spdlog::debug("Type name: {}", type_layout->getName());

    if (type_layout->getSize() > 0) {
        spdlog::debug("Size in bytes: {}", type_layout->getSize());
		spdlog::debug("Alignment in bytes: {}", type_layout->getAlignment());
    }
}

}