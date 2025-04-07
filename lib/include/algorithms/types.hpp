#pragma once

namespace vkengine {

// Primary template (undefined to prevent implicit instantiation)
template<typename T>
struct type_name;

template<>
struct type_name<uint16_t> {
    static constexpr const char* value = "uint16_t";
};

template<>
struct type_name<uint32_t> {
    static constexpr const char* value = "uint32_t";
};

}