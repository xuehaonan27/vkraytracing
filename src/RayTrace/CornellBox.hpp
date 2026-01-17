#ifndef RAYTRACE_CORNELLBOX_HPP
#define RAYTRACE_CORNELLBOX_HPP

#include <vector>

#include "Material.hpp"
#include "Vertex.hpp"

namespace RayTrace {

class CornellBox final {
  public:
    static void Create(
        float scale,
        std::vector<Vertex>& vertices,
        std::vector<uint32_t>& indices,
        std::vector<Material>& materials
    );
};

} // namespace RayTrace

#endif // RAYTRACE_CORNELLBOX_HPP
