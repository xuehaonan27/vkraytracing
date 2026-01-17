#ifndef UTILS_MACROS_HPP
#define UTILS_MACROS_HPP

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#ifndef GLFW_INCLUDE_NONE
    #define GLFW_INCLUDE_NONE
#endif

#ifndef GLFW_INCLUDE_VULKAN
    #define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>
#undef APIENTRY

#include <stdexcept>

#define VULKAN_NON_COPIABLE(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(const ClassName&) = delete; \
    ClassName& operator=(ClassName&&) = delete;

#define VULKAN_HANDLE(VulkanHandleType, name) \
  public: \
    VulkanHandleType Handle() const { \
        return name; \
    } \
\
  private: \
    VulkanHandleType name {};

void Check(VkResult result, const char* operation);
const char* ToString(VkResult result);

#endif // UTILS_MACROS_HPP
