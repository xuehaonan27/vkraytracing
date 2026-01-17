#include "DebugUtils.hpp"

namespace Base {

DebugUtils::DebugUtils(VkInstance instance) :
    vkSetDebugUtilsObjectNameEXT_(
        reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT")
        )
    ) {
#ifndef NDEBUG
    if (vkSetDebugUtilsObjectNameEXT_ == nullptr) {
        throw std::runtime_error("failed to get address of 'vkSetDebugUtilsObjectNameEXT'");
    }
#endif
}

} // namespace Base