#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <shader_layout.hpp>
#include <spdlog/fmt/ranges.h>
#include <ranges>

namespace vkengine {

shader_manager::shader_manager(std::reference_wrapper<vulkan_core> vulkan_core)
    : vulkan(vulkan_core) {
    setup_slang();
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

shader_object shader_manager::load_shader(
    const std::string& module_name,
    const std::string& entry_point_name,
    const std::array<uint32_t, 3>& workgroup_sizes,
    const std::vector<slang::SpecializationArg> specialisation_args
) {
    spdlog::info("Loading shader: {}, entry point: {}, workgroups: {}", module_name, entry_point_name, workgroup_sizes);

    Slang::ComPtr<slang::IModule> module(session->loadModule(module_name.c_str(), diagnostics.writeRef()));

    if (!module)
        throw_exception_with_slang_diagnostics("Failed to create module");

    Slang::ComPtr<slang::IEntryPoint> entry_point = nullptr;
    module->findEntryPointByName(entry_point_name.c_str(), entry_point.writeRef());

    if (!entry_point)
        throw detailed_exception("Failed to load entry point");

	if (specialisation_args.size() > 0) {
        Slang::ComPtr<slang::IComponentType> specialized_entry_point;

		entry_point->specialize(specialisation_args.data(), specialisation_args.size(), specialized_entry_point.writeRef(), diagnostics.writeRef());
	}

    Slang::ComPtr<slang::IModule> workgroup_module = create_workgroup_module(workgroup_sizes);

    slang::IComponentType* program = nullptr;
    std::array<slang::IComponentType*, 4> components = { module, entry_point, subgroup_module, workgroup_module };
    session->createCompositeComponentType(components.data(), components.size(), &program, diagnostics.writeRef());

    if (!program)
        throw_exception_with_slang_diagnostics("Failed to create slang program");

    slang::ProgramLayout* layout = program->getLayout();
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
    program->getTargetCode(0, spirv_code.writeRef(), diagnostics.writeRef());

    if (!spirv_code)
        throw_exception_with_slang_diagnostics("Failed to create spirv code");

	slang::ProgramLayout* program_layout = program->getLayout();

    shader_layout shader_layout = create_pipeline_layout(program_layout, vulkan.get());

    spdlog::debug("Push constant range: {}", shader_layout.push_constant_ranges[0].size);

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

    vk::ResultValue<vk::ShaderEXT> shader_ext = vulkan.get().device().createShaderEXT(
        vk::ShaderCreateInfoEXT()
        .setStage(stage)
        .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
        .setCodeSize(spirv_code->getBufferSize())
        .setPName(entry_point_name.data())
        .setPCode(static_cast<const uint32_t*>(spirv_code->getBufferPointer()))
        .setPushConstantRanges(shader_layout.push_constant_ranges)
    );

    if (shader_ext.result != vk::Result::eSuccess)
        throw detailed_exception("Failed to create shader object");

    shader_object shader_obj = {
        .pipeline_layout = shader_layout.pipeline_layout,
        .push_constant_range = shader_layout.push_constant_ranges[0],
        .shader_ext = shader_ext.value,
        .stage = stage
    };

    return shader_obj;
}

void shader_manager::setup_slang() {
    slang::createGlobalSession(global_session.writeRef());

    target_desc = {
        .format = SLANG_SPIRV,
        .profile = global_session->findProfile("spirv_1_5"),
        .forceGLSLScalarBufferLayout = true
    };

    std::vector<slang::CompilerOptionEntry> compiler_option_entries = {
        {
            .name = slang::CompilerOptionName::VulkanUseEntryPointName,
            .value = {.kind = slang::CompilerOptionValueKind::Int, .intValue0 = 1}
        }
    };

    session_desc = {
        .targets = &target_desc,
        .targetCount = 1,
        .compilerOptionEntries = compiler_option_entries.data(),
        .compilerOptionEntryCount = static_cast<uint32_t>(compiler_option_entries.size())
    };

    global_session->createSession(session_desc, session.writeRef());
}

void shader_manager::create_subgroup_module() {
    std::string subgroup_module_src = fmt::format("export static const uint SUBGROUP_SIZE = {};", vulkan.get().physical_device_properties().subgroup_properties.subgroupSize);
    subgroup_module = session->loadModuleFromSourceString("subgroup_size", "subgroup_size.slang", subgroup_module_src.c_str());

    if (!subgroup_module)
        throw std::runtime_error("Failed to create subgroup module");
}

bool shader_manager::check_valid_workgroup_sizes(const std::array<uint32_t, 3>& workgroup_sizes) {
    auto& max_workgroup_sizes = vulkan.get().physical_device_properties().properties.limits.maxComputeWorkGroupSize;
    if (workgroup_sizes[0] > max_workgroup_sizes[0] || workgroup_sizes[1] > max_workgroup_sizes[1] || workgroup_sizes[2] > max_workgroup_sizes[2])
        return false;
    return true;
}

Slang::ComPtr<slang::IModule> shader_manager::create_workgroup_module(const std::array<uint32_t, 3>& workgroup_sizes) {
    std::string workgroup_module_src = std::format(
        "export static const uint WORKGROUP_SIZE_X = {};"
        "export static const uint WORKGROUP_SIZE_Y = {};"
        "export static const uint WORKGROUP_SIZE_Z = {};",
        workgroup_sizes[0],
        workgroup_sizes[1],
        workgroup_sizes[2]
    );

    Slang::ComPtr<slang::IModule> workgroup_module(session->loadModuleFromSourceString("workgroup_sizes", "workgroup_sizes.slang", workgroup_module_src.c_str()));

    if (!workgroup_module)
        throw detailed_exception("Failed to load workgroup module");

    return workgroup_module;
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