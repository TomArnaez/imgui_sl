#pragma once

#include <stdexcept>
#include <sstream>
#include <string>
#include <source_location>
#include <spdlog/spdlog.h>

namespace vkengine {

class detailed_exception : public std::runtime_error {
public:
    template <typename... Args>
    detailed_exception(fmt::format_string<Args...> fmt_str, Args&&... args)
        : detailed_exception(std::source_location::current(), fmt_str, std::forward<Args>(args)...) {
    }

    template <typename... Args>
    detailed_exception(const std::source_location& location, fmt::format_string<Args...> fmt_str, Args&&... args)
        : std::runtime_error(format_message(fmt::format(fmt_str, std::forward<Args>(args)...), location)) {
        spdlog::error("Exception thrown: {}", what());
    }

    detailed_exception(const std::source_location& location, const std::string& message)
        : std::runtime_error(format_message(message, location)) {
        spdlog::error("Exception thrown: {}", what());
    }


private:
    static std::string format_message(const std::string& message,
        const std::source_location& location) {
        return fmt::format(
            "{} [Function: {}, File: {}, Line: {}]",
            message,
            location.function_name(),
            location.file_name(),
            location.line()
        );
    }
};

} // namespace vkengine