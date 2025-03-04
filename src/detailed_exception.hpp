#pragma once

#include <stdexcept>
#include <sstream>
#include <string>
#include <source_location>
#include <spdlog/spdlog.h>

class detailed_exception : public std::runtime_error {
    public:
        detailed_exception(const std::string& message,
                          const std::source_location& location = std::source_location::current())
            : std::runtime_error(format_message(message, location)) {
            spdlog::error("{}", what());
        }
    
    private:
        static std::string format_message(const std::string& message,
                                         const std::source_location& location)
        {
            std::ostringstream oss;
            oss << message << " [Function: " << location.function_name()
                << ", File: " << location.file_name()
                << ", Line: " << location.line() << "]";
            return oss.str();
        }
    };