#ifndef BASE_VERSION_HPP
#define BASE_VERSION_HPP

#include <ostream>
#include <vulkan/vulkan.hpp>

namespace Base {

class Version final {
  public:
    explicit Version(const uint32_t version) :
        Major(VK_VERSION_MAJOR(version)),
        Minor(VK_VERSION_MINOR(version)),
        Patch(VK_VERSION_PATCH(version)) {}

    Version(const uint32_t version, const uint32_t vendorId) :
        Major(VK_VERSION_MAJOR(version)),
        Minor(VK_VERSION_MINOR(version) >> (vendorId == 0x10DE ? 2 : 0)),
        Patch(VK_VERSION_PATCH(version) >> (vendorId == 0x10DE ? 4 : 0)) {}

    const unsigned Major;
    const unsigned Minor;
    const unsigned Patch;

    friend std::ostream& operator<<(std::ostream& out, const Version& version) {
        return out << version.Major << "." << version.Minor << "." << version.Patch;
    }
};

} // namespace Base

#endif // BASE_VERSION_HPP
