#pragma once

#include <detailed_exception.hpp>
#include <slang.h>
#include <source_location>

namespace vkengine {

void throw_on_slang_error(SlangResult result, std::source_location& source) {
    if (result < 0)
        throw detailed_exception(source, "Got slang error code: {}", result);
}

}