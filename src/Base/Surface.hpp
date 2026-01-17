#ifndef BASE_SURFACE_HPP
#define BASE_SURFACE_HPP

#include "Instance.hpp"
#include "Window.hpp"

namespace Base {
class Instance;
class Window;

class Surface final {
  public:
    VULKAN_NON_COPIABLE(Surface)

    explicit Surface(const Instance& instance);
    ~Surface();

    const class Instance& Instance() const {
        return instance_;
    }

  private:
    const class Instance& instance_;

    VULKAN_HANDLE(VkSurfaceKHR, surface_)
};

} // namespace Base

#endif // BASE_SURFACE_HPP
