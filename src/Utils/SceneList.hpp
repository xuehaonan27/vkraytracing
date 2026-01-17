#ifndef UTILS_SCENE_LIST_HPP
#define UTILS_SCENE_LIST_HPP

#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Glm.hpp"

namespace RayTrace {
class Model;
class Texture;
}; // namespace RayTrace

typedef std::tuple<std::vector<RayTrace::Model>, std::vector<RayTrace::Texture>> SceneAssets;

class SceneList final {
  public:
    struct CameraInitialSate {
        glm::mat4 ModelView;
        float FieldOfView;
        float Aperture;
        float FocusDistance;
        float ControlSpeed;
        bool GammaCorrection;
        bool HasSky;
    };

    static SceneAssets CubeAndSpheres(CameraInitialSate& camera);
    static SceneAssets RayTracingInOneWeekend(CameraInitialSate& camera);
    static SceneAssets PlanetsInOneWeekend(CameraInitialSate& camera);
    static SceneAssets LucyInOneWeekend(CameraInitialSate& camera);
    static SceneAssets CornellBox(CameraInitialSate& camera);
    static SceneAssets CornellBoxLucy(CameraInitialSate& camera);

    static const std::vector<std::pair<std::string, std::function<SceneAssets(CameraInitialSate&)>>>
        AllScenes;
};

#endif // UTILS_SCENE_LIST_HPP
