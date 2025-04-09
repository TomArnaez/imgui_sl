#pragma once

#include <vulkan_core.hpp>
#include <allocator.hpp>

namespace vkengine {

enum class access_policy {
    device,
    host_visible
};

template<access_policy policy>
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

template<typename T, uint32_t dims = 1, access_policy policy = access_policy::device>
class typed_buffer {
public:
    typed_buffer(
        std::reference_wrapper<allocator> allocator,
        const vulkan_core& core,
        const std::array<uint32_t, dims>& shape
    ) : allocator(allocator), shape_(shape) {
        uint64_t count = 1;
        for (auto d : shape)
            count *= d;
        element_count = static_cast<uint32_t>(count);

        auto allocation_info = allocation_policy_traits<policy>::get_allocation_create_info();

        buffer = allocator.get().create_buffer(
            vk::BufferCreateInfo()
                .setSize(sizeof(T) * element_count)
                .setUsage(
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress
                ),
            allocation_info
        );

        buffer_address = core.device().getBufferAddress(
            vk::BufferDeviceAddressInfo().setBuffer(buffer.handle));
    }

    template <uint32_t D = dims, typename = std::enable_if_t<D == 1>>
    typed_buffer(std::reference_wrapper<allocator> allocator,
        const vulkan_core& core,
        uint32_t element_count)
        : typed_buffer(
            allocator,
            core,
            std::array<uint32_t, 1>{ element_count }
        ) {}

    void destroy() {
        allocator.get().destroy_buffer(buffer);
    }

    operator device_mdspan<dims>() const {
        return device_mdspan<dims> {
            .span = buffer_address,
            .dims = shape_
        };
    }

    operator device_span() const {
        return device_span {
            .span = buffer_address,
            .size = element_count
        };
    }

    device_mdspan<dims> as_mdspan() const {
        return device_mdspan<dims>{
            .span = buffer_address,
            .dims = shape_
        };
    }

    vkengine::device_span as_span() const {
        return vkengine::device_span{
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

    const std::array<uint32_t, dims>& shape() const requires (dims >= 2) {
        return shape_;
    }

    uint32_t size() const {
        return element_count;
    }

    uint32_t size_bytes() const {
        return buffer.size;
    }

    T* mapping() const requires (policy == access_policy::host_visible) {
        T* mapped_data = static_cast<T*>(buffer.allocation_info.pMappedData);
        if (!mapped_data)
            throw detailed_exception("Buffer not mapped to host memory");
        return mapped_data;
    }

    auto data() const requires (policy == access_policy::host_visible) {
        return std::ranges::subrange(
            mapping(),
            mapping() + element_count
        );
    }

private:
    std::reference_wrapper<allocator> allocator;
    vk::DeviceAddress buffer_address;
    buffer buffer;

    std::array<uint32_t, dims> shape_;
    uint32_t element_count;
};

template<typename T, uint32_t dims>
using device_buffer_nd = typed_buffer<T, dims, access_policy::device>;

template<typename T, uint32_t dims>
using host_visible_buffer_nd = typed_buffer<T, dims, access_policy::host_visible>;

template<typename T>
using device_buffer = device_buffer_nd<T, 1>;

template<typename T>
using host_visible_buffer = host_visible_buffer_nd<T, 1>;

} // namespace vkengine