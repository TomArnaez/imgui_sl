#pragma once

#include <cstdio>
#include <stdlib.h>

#include <vulkan/vulkan.h>
#include <detailed_exception.hpp>

#define VK_CHECK(expr)                                                     \
    do {                                                                   \
        VkResult vk_result = (expr);                                       \
        if (vk_result != VK_SUCCESS) {                                     \
            throw detailed_exception("Vulkan error: " +                    \
                                     std::to_string(vk_result));           \
        }                                                                  \
    } while (0)

inline void check_vk_result(VkResult err) {
    if (err != VK_SUCCESS)
        throw detailed_exception("Error: VkResult");
}