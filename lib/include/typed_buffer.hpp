#pragma once

#include <vulkan_core.hpp>
#include <allocator.hpp>

namespace vkengine {

enum class access_policy {
    device,
    host_visible
};

template<access_policy Policy>
struct allocation_policy_traits;

template<>
struct allocation_policy_traits<access_policy::device> {
    static VmaAllocationCreateInfo get_allocation_create_info() {
        return VmaAllocationCreateInfo{
            .flags = 0,  // Device-local by default
            .usage = VMA_MEMORY_USAGE_AUTO
        };
    }
};

template<>
struct allocation_policy_traits<access_policy::host_visible> {
    static VmaAllocationCreateInfo get_allocation_create_info() {
        return VmaAllocationCreateInfo{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO
        };
    }
};

struct device_span {
    vk::DeviceAddress span;
    uint32_t size;
};

template<uint32_t dims>
struct device_mdspan {
    vk::DeviceAddress span;
    std::array<uint32_t, dims> dims;
};

template<typename T, access_policy Policy = access_policy::device>
class typed_buffer {
public:
    typed_buffer(
        std::reference_wrapper<allocator> allocator,
        const vulkan_core& core,
        uint32_t element_count
    ) : allocator(allocator), element_count(element_count) {
        auto allocation_info = allocation_policy_traits<Policy>::get_allocation_create_info();

        buffer = allocator.get().create_buffer(
            vk::BufferCreateInfo()
            .setSize(sizeof(T) * element_count)
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eShaderDeviceAddress),
            allocation_info
        );

        buffer_address = core.device().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer.handle));
    }

    void destroy() {
        allocator.get().destroy_buffer(buffer);
    }

    operator device_span() {
        return device_span();
    }

    device_span device_span() {
        return {
            .span = buffer_address,
            .size = element_count
        };
    }

    vk::DeviceAddress device_address() {
        return buffer_address;
    }

    vk::Buffer vk_handle() {
        return buffer.handle;
    }

    uint32_t size() const {
        return element_count;
    }

    uint32_t size_bytes() const {
        return buffer.size;
    }

    T* mapping() const requires (Policy == access_policy::host_visible) {
        T* mapped_data = static_cast<T*>(buffer.allocation_info.pMappedData);
        if (!mapped_data)
            throw detailed_exception("Buffer not mapped to host memory");
        return mapped_data;
    }

    auto data() const requires (Policy == access_policy::host_visible) {
        return std::ranges::subrange(
            mapping(),
            mapping() + element_count
        );
    }

private:
    std::reference_wrapper<allocator> allocator;
    vk::DeviceAddress buffer_address;
    buffer buffer;
    uint32_t element_count;
};

template<typename T>
using host_visible_buffer = typed_buffer<T, access_policy::host_visible>;

template<typename T>
using device_buffer = typed_buffer<T, access_policy::device>;

} // namespace vkengine