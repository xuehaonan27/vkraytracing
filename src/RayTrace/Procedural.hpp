#ifndef RAYTRACE_PROCEDURAL_HPP
#define RAYTRACE_PROCEDURAL_HPP

#include <utility>

#include "Utils/Glm.hpp"

namespace RayTrace {

class Procedural {
  public:
    Procedural(const Procedural&) = delete;
    Procedural(Procedural&&) = delete;
    Procedural& operator=(const Procedural&) = delete;
    Procedural& operator=(Procedural&&) = delete;

    Procedural() = default;
    virtual ~Procedural() = default;
    ;
    virtual std::pair<glm::vec3, glm::vec3> BoundingBox() const = 0;
};
} // namespace RayTrace

#endif // RAYTRACE_PROCEDURAL_HPP
