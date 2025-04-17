#pragma once
#include <vulkan_core.hpp>
#include <allocator.hpp>

namespace vkengine {

enum buffer_kind { storage, uniform, vertex };

template<buffer_kind Kind>
struct buffer_kind_traits;

template<>
struct buffer_kind_traits<buffer_kind::storage> {
    static constexpr vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;
};

template<>
struct buffer_kind_traits<buffer_kind::uniform> {
    static constexpr vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eUniformBuffer;
};

template<buffer_kind Kind>
concept device_addressable = (Kind != buffer_kind::uniform);

template<bool Active>
struct device_address_holder;

template<>
struct device_address_holder<true> {
    vk::DeviceAddress addr_{};

    void  set(vk::DeviceAddress a) { addr_ = a; }
    auto  get() const { return addr_; }
};

template<>
struct device_address_holder<false>{};

enum class access_policy { device, host_visible };
template<access_policy P> struct allocation_policy_traits;

template<>
struct allocation_policy_traits<access_policy::device> {
    static VmaAllocationCreateInfo get_allocation_create_info() {
        return { .flags = 0, .usage = VMA_MEMORY_USAGE_AUTO };
    }
};

template<>
struct allocation_policy_traits<access_policy::host_visible> {
    static VmaAllocationCreateInfo get_allocation_create_info() {
        return { .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO };
    }
};

struct device_span {
    vk::DeviceAddress span;
    uint32_t          size;
};

template<uint32_t dims>
struct device_mdspan {
    vk::DeviceAddress           span;
    std::array<uint32_t, dims>  dims;
};

template<
    typename        T,
    uint32_t        dims = 1,
    access_policy   policy = access_policy::device,
    buffer_kind     kind = buffer_kind::storage>
class typed_buffer : private device_address_holder<device_addressable<kind>> {
public:
    typed_buffer(std::reference_wrapper<allocator> alloc,
        const vulkan_core& core,
        const std::array<uint32_t, dims>& shape)
        : allocator_{ alloc }, shape_{ shape } {
        uint64_t cnt = 1;
        for (auto d : shape) cnt *= d;
        element_count_ = static_cast<uint32_t>(cnt);

        auto alloc_info = allocation_policy_traits<policy>::get_allocation_create_info();
        buffer_ = allocator_.get().create_buffer(
            vk::BufferCreateInfo{}
            .setSize(sizeof(T) * element_count_)
            .setUsage(buffer_kind_traits<kind>::usage),
            alloc_info);

        if constexpr (device_addressable<kind>)
            address_.set(core.device().getBufferAddress(
                vk::BufferDeviceAddressInfo{}.setBuffer(buffer_.handle)));
    }

    template<uint32_t D = dims> requires (D == 1)
        typed_buffer(std::reference_wrapper<allocator> alloc,
            const vulkan_core& core,
            uint32_t                          elements)
        : typed_buffer(alloc, core, std::array<uint32_t, 1>{elements}) {
    }

    void destroy() { allocator_.get().destroy_buffer(buffer_); }

    operator device_mdspan<dims>() const
        requires device_addressable<kind> {
        return { address_.get(), shape_};
    }

    operator device_span() const
        requires device_addressable<kind> {
        return { address_.get(), element_count_ };
    }

    device_mdspan<dims> as_mdspan() const
        requires device_addressable<kind> {
        return { address_.get(), shape_ };
    }

    vkengine::device_span as_span() const
        requires device_addressable<kind> {
        return { address_.get(), element_count_ };
    }

    vk::DeviceAddress device_address() const
        requires device_addressable<kind> {
        return address_.get();
    }

    vk::Buffer vk_handle() const { return buffer_.handle; }

    const std::array<uint32_t, dims>& shape() const
        requires (dims >= 2) {
        return shape_;
    }

    uint32_t size() const { return element_count_; }
    uint32_t size_bytes() const { return buffer_.size; }

    T* mapping() const 
        requires (policy == access_policy::host_visible) {
        if (auto* p = static_cast<T*>(buffer_.allocation_info.pMappedData); p)
            return p;
        throw detailed_exception("Buffer not mapped to host memory");
    }

    auto data() const 
        requires (policy == access_policy::host_visible) {
        return std::ranges::subrange(mapping(), mapping() + element_count_);
    }
private:
    using addr_base = device_address_holder<device_addressable<kind>>;

    std::reference_wrapper<allocator>   allocator_;
    [[no_unique_address]] addr_base     address_;
    buffer                              buffer_;
    std::array<uint32_t, dims>          shape_;
    uint32_t                            element_count_{};
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
