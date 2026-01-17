#ifndef BASE_WINDOW_CONFIG_HPP
#define BASE_WINDOW_CONFIG_HPP

#include <cstdint>
#include <string>

namespace Base {
struct WindowConfig final {
    std::string Title;
    uint32_t Width;
    uint32_t Height;
    bool CursorDisabled;
    bool Fullscreen;
    bool Resizable;
};
} // namespace Base

#endif // BASE_WINDOW_CONFIG_HPP
