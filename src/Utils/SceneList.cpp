#include "SceneList.hpp"

#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "RayTrace/Material.hpp"
#include "RayTrace/Model.hpp"
#include "RayTrace/Texture.hpp"

using namespace glm;
using RayTrace::Material;
using RayTrace::Model;
using RayTrace::Texture;

void AddRayTracingInOneWeekendCommonScene(
    std::vector<RayTrace::Model>& models,
    const bool& isProc,
    std::function<float()>& random
) {
    // Common models from the final scene from Ray Tracing In One Weekend book. Only the three central spheres are missing.
    // Calls to random() are always explicit and non-inlined to avoid C++ undefined evaluation order of function arguments,
    // this guarantees consistent and reproducible behaviour across different platforms and compilers.

    models.push_back(
        Model::CreateSphere(
            vec3(0, -1000, 0),
            1000,
            Material::Lambertian(vec3(0.5f, 0.5f, 0.5f)),
            isProc
        )
    );

    for (int i = -11; i < 11; ++i) {
        for (int j = -11; j < 11; ++j) {
            const float chooseMat = random();
            const float center_y = static_cast<float>(j) + 0.9f * random();
            const float center_x = static_cast<float>(i) + 0.9f * random();
            const vec3 center(center_x, 0.2f, center_y);

            if (length(center - vec3(4, 0.2f, 0)) > 0.9f) {
                if (chooseMat < 0.8f) // Diffuse
                {
                    const float b = random() * random();
                    const float g = random() * random();
                    const float r = random() * random();

                    models.push_back(
                        Model::CreateSphere(
                            center,
                            0.2f,
                            Material::Lambertian(vec3(r, g, b)),
                            isProc
                        )
                    );
                } else if (chooseMat < 0.95f) // Metal
                {
                    const float fuzziness = 0.5f * random();
                    const float b = 0.5f * (1 + random());
                    const float g = 0.5f * (1 + random());
                    const float r = 0.5f * (1 + random());

                    models.push_back(
                        Model::CreateSphere(
                            center,
                            0.2f,
                            Material::Metallic(vec3(r, g, b), fuzziness),
                            isProc
                        )
                    );
                } else // Glass
                {
                    models.push_back(
                        Model::CreateSphere(center, 0.2f, Material::Dielectric(1.5f), isProc)
                    );
                }
            }
        }
    }
}

const std::vector<std::pair<std::string, std::function<SceneAssets(SceneList::CameraInitialSate&)>>>
    SceneList::AllScenes = {
        // std::make_pair("Cube And Spheres", CubeAndSpheres),
        std::make_pair("Ray Tracing In One Weekend", RayTracingInOneWeekend),
        std::make_pair("Planets In One Weekend", PlanetsInOneWeekend),
        std::make_pair("Lucy In One Weekend", LucyInOneWeekend),
        std::make_pair("Cornell Box", CornellBox),
        std::make_pair("Cornell Box & Lucy", CornellBoxLucy),
        std::make_pair("Giant Humanoid Plane Monster", GiantHumanoidPlaneMonster),
};

SceneAssets SceneList::CubeAndSpheres(CameraInitialSate& camera) {
    // Basic test scene.

    camera.ModelView = translate(mat4(1), vec3(0, 0, -2));
    camera.FieldOfView = 90;
    camera.Aperture = 0.05f;
    camera.FocusDistance = 2.0f;
    camera.ControlSpeed = 2.0f;
    camera.GammaCorrection = false;
    camera.HasSky = true;

    std::vector<Model> models;
    std::vector<Texture> textures;

    models.push_back(Model::LoadModel("../assets/models/cube_multi.obj"));
    models.push_back(
        Model::CreateSphere(
            vec3(1, 0, 0),
            0.5,
            Material::Metallic(vec3(0.7f, 0.5f, 0.8f), 0.2f),
            true
        )
    );
    models.push_back(Model::CreateSphere(vec3(-1, 0, 0), 0.5, Material::Dielectric(1.5f), true));
    models.push_back(
        Model::CreateSphere(vec3(0, 1, 0), 0.5, Material::Lambertian(vec3(1.0f), 0), true)
    );

    textures.push_back(
        Texture::LoadTexture(
            "../assets/textures/land_ocean_ice_cloud_2048.png",
            Base::SamplerConfig()
        )
    );

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::RayTracingInOneWeekend(CameraInitialSate& camera) {
    // Final scene from Ray Tracing In One Weekend book.

    camera.ModelView = lookAt(vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 20;
    camera.Aperture = 0.1f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    AddRayTracingInOneWeekendCommonScene(models, isProc, random);

    models.push_back(Model::CreateSphere(vec3(0, 1, 0), 1.0f, Material::Dielectric(1.5f), isProc));
    models.push_back(
        Model::CreateSphere(
            vec3(-4, 1, 0),
            1.0f,
            Material::Lambertian(vec3(0.4f, 0.2f, 0.1f)),
            isProc
        )
    );
    models.push_back(
        Model::CreateSphere(
            vec3(4, 1, 0),
            1.0f,
            Material::Metallic(vec3(0.7f, 0.6f, 0.5f), 0.0f),
            isProc
        )
    );

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::PlanetsInOneWeekend(CameraInitialSate& camera) {
    // Same as RayTracingInOneWeekend but using textures.

    camera.ModelView = lookAt(vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 20;
    camera.Aperture = 0.1f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;
    std::vector<Texture> textures;

    AddRayTracingInOneWeekendCommonScene(models, isProc, random);

    models.push_back(
        Model::CreateSphere(vec3(0, 1, 0), 1.0f, Material::Metallic(vec3(1.0f), 0.1f, 2), isProc)
    );
    models.push_back(
        Model::CreateSphere(vec3(-4, 1, 0), 1.0f, Material::Lambertian(vec3(1.0f), 0), isProc)
    );
    models.push_back(
        Model::CreateSphere(vec3(4, 1, 0), 1.0f, Material::Metallic(vec3(1.0f), 0.0f, 1), isProc)
    );

    textures.push_back(
        Texture::LoadTexture("../assets/textures/2k_mars.jpg", Base::SamplerConfig())
    );
    textures.push_back(
        Texture::LoadTexture("../assets/textures/2k_moon.jpg", Base::SamplerConfig())
    );
    textures.push_back(
        Texture::LoadTexture(
            "../assets/textures/land_ocean_ice_cloud_2048.png",
            Base::SamplerConfig()
        )
    );

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::LucyInOneWeekend(CameraInitialSate& camera) {
    // Same as RayTracingInOneWeekend but using the Lucy 3D model.

    camera.ModelView = lookAt(vec3(13, 2, 3), vec3(0, 1.0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 20;
    camera.Aperture = 0.05f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    AddRayTracingInOneWeekendCommonScene(models, isProc, random);

    auto lucy0 = Model::LoadModel("../assets/models/lucy.obj");
    auto lucy1 = lucy0;
    auto lucy2 = lucy0;

    const auto i = mat4(1);
    const float scaleFactor = 0.0035f;

    lucy0.Transform(rotate(
        scale(translate(i, vec3(0, -0.08f, 0)), vec3(scaleFactor)),
        radians(90.0f),
        vec3(0, 1, 0)
    ));

    lucy1.Transform(rotate(
        scale(translate(i, vec3(-4, -0.08f, 0)), vec3(scaleFactor)),
        radians(90.0f),
        vec3(0, 1, 0)
    ));

    lucy2.Transform(rotate(
        scale(translate(i, vec3(4, -0.08f, 0)), vec3(scaleFactor)),
        radians(90.0f),
        vec3(0, 1, 0)
    ));

    lucy0.SetMaterial(Material::Dielectric(1.5f));
    lucy1.SetMaterial(Material::Lambertian(vec3(0.4f, 0.2f, 0.1f)));
    lucy2.SetMaterial(Material::Metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f));

    models.push_back(std::move(lucy0));
    models.push_back(std::move(lucy1));
    models.push_back(std::move(lucy2));

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::CornellBox(CameraInitialSate& camera) {
    camera.ModelView = lookAt(vec3(278, 278, 800), vec3(278, 278, 0), vec3(0, 1, 0));
    camera.FieldOfView = 40;
    camera.Aperture = 0.0f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 500.0f;
    camera.GammaCorrection = true;
    camera.HasSky = false;

    const auto i = mat4(1);
    const auto white = Material::Lambertian(vec3(0.73f, 0.73f, 0.73f));

    auto box0 = Model::CreateBox(vec3(0, 0, -165), vec3(165, 165, 0), white);
    auto box1 = Model::CreateBox(vec3(0, 0, -165), vec3(165, 330, 0), white);

    box0.Transform(
        rotate(translate(i, vec3(555 - 130 - 165, 0, -65)), radians(-18.0f), vec3(0, 1, 0))
    );
    box1.Transform(
        rotate(translate(i, vec3(555 - 265 - 165, 0, -295)), radians(15.0f), vec3(0, 1, 0))
    );

    std::vector<Model> models;
    models.push_back(Model::CreateCornellBox(555));
    models.push_back(box0);
    models.push_back(box1);

    return std::make_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::CornellBoxLucy(CameraInitialSate& camera) {
    camera.ModelView = lookAt(vec3(278, 278, 800), vec3(278, 278, 0), vec3(0, 1, 0));
    camera.FieldOfView = 40;
    camera.Aperture = 0.0f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 500.0f;
    camera.GammaCorrection = true;
    camera.HasSky = false;

    const auto i = mat4(1);
    const auto sphere = Model::CreateSphere(
        vec3(555 - 130, 165.0f, -165.0f / 2 - 65),
        80.0f,
        Material::Dielectric(1.5f),
        true
    );
    auto lucy0 = Model::LoadModel("../assets/models/lucy.obj");

    lucy0.Transform(rotate(
        scale(translate(i, vec3(555 - 300 - 165 / 2, -9, -295 - 165 / 2)), vec3(0.6f)),
        radians(75.0f),
        vec3(0, 1, 0)
    ));

    std::vector<Model> models;
    models.push_back(Model::CreateCornellBox(555));
    models.push_back(sphere);
    models.push_back(lucy0);

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

#define TEXTURE_LOAD(path) textures.push_back(Texture::LoadTexture(path, Base::SamplerConfig()));

SceneAssets SceneList::AbandonedWareHouse(CameraInitialSate& camera) {
    camera.ModelView = lookAt(vec3(13, 2, 3), vec3(0, 1.0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 20;
    camera.Aperture = 0.05f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;
    std::vector<Texture> textures;

    models.push_back(Model::LoadModelGLTF("../assets/models/abandoned_warehouse/scene.gltf"));

#define FOR_LIST_OF_VARIABLES(DO) \
    DO("../assets/models/abandoned_warehouse/textures/briques-ss-fenetre_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/briques-sur-fenetre_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/colonne-beton.001_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/colonne-beton_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/DM_Watch_Your_Step_graffiti_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.002_diffuse.jpeg") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.002_specularGlossiness.png") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.004_diffuse.jpeg") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.004_specularGlossiness.png") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.005_diffuse.jpeg") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.005_specularGlossiness.png") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.006_diffuse.jpeg") \
    DO("../assets/models/abandoned_warehouse/textures/fenetre.006_specularGlossiness.png") \
    DO("../assets/models/abandoned_warehouse/textures/gravas_diffuse.jpeg") \
    DO("../assets/models/abandoned_warehouse/textures/gravas_specularGlossiness.png") \
    DO("../assets/models/abandoned_warehouse/textures/material_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/mur-arriere_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/mur-droite_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/mur-fond_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/mur-gauche_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/palette.001_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/palette_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/pandagun_1280_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/poutre-toit.001_diffuse.png") \
    DO( \
        "../assets/models/abandoned_warehouse/textures/TexturesCom_CardboardPlain0019_2_alphamasked_S_diffuse.png" \
    ) \
    DO( \
        "../assets/models/abandoned_warehouse/textures/TexturesCom_CardboardPlain0022_2_alphamasked_S_diffuse.png" \
    ) \
    DO( \
        "../assets/models/abandoned_warehouse/textures/TexturesCom_CardboardPlain0026_2_alphamasked_S_diffuse.png" \
    ) \
    DO("../assets/models/abandoned_warehouse/textures/TexturesCom_DoorsRollup0044_M_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/TexturesCom_DoorsWoodBarn0017_M_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/toiture-tole.001_diffuse.png") \
    DO("../assets/models/abandoned_warehouse/textures/toiture-tole_diffuse.png")

    FOR_LIST_OF_VARIABLES(TEXTURE_LOAD)
#undef FOR_LIST_OF_VARIABLES

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::DeadByDaylight(CameraInitialSate& camera) {
    camera.ModelView = lookAt(vec3(13, 2, 3), vec3(0, 1.0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 20;
    camera.Aperture = 0.05f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;
    std::vector<Texture> textures;

    models.push_back(Model::LoadModelGLTF("../assets/models/dead_by_daylight/scene.gltf"));

#define FOR_LIST_OF_VARIABLES(DO) \
    DO("../assets/models/dead_by_daylight/textures/AccHeadMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/AccMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/HeadMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/LeftArmMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/LegsMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/RightArmMat_baseColor.png") \
    DO("../assets/models/dead_by_daylight/textures/TorsoMat_baseColor.png")

    FOR_LIST_OF_VARIABLES(TEXTURE_LOAD)
#undef FOR_LIST_OF_VARIABLES

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::GiantHumanoidPlaneMonster(CameraInitialSate& camera) {
    camera.ModelView = lookAt(vec3(23, 15, 30), vec3(0, 1.0, 0), vec3(0, 1, 0));
    camera.FieldOfView = 40;
    camera.Aperture = 0.0f;
    camera.FocusDistance = 10.0f;
    camera.ControlSpeed = 5.0f;
    camera.GammaCorrection = true;
    camera.HasSky = true;

    const bool isProc = true;
    const auto i = mat4(1);
    const float scaleFactor = 10.0f;

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;
    std::vector<Texture> textures;

    AddRayTracingInOneWeekendCommonScene(models, isProc, random);

    auto obj0 = Model::LoadModelGLTF("../assets/models/giant_humanoid_plant_monster/scene.gltf");
    auto obj1 = obj0;
    auto obj2 = obj0;

    // obj0.Transform(rotate(
    //     scale(translate(i, vec3(0.0, 5.0f, 0)), vec3(scaleFactor)),
    //     radians(-90.0f),
    //     vec3(1, 0, 0)
    // ));

    obj1.Transform(rotate(
        scale(translate(i, vec3(-10.0, 5.0f, 0)), vec3(scaleFactor)),
        radians(-90.0f),
        vec3(1, 0, 0)
    ));
    obj1.SetMaterial(Material::Dielectric(1.5f));

    obj2.Transform(rotate(
        scale(translate(i, vec3(10.0f, 5.0f, 0)), vec3(scaleFactor)),
        radians(-90.0f),
        vec3(1, 0, 0)
    ));
    obj2.SetMaterial(Material::Metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f));

    // models.push_back(std::move(obj0));
    models.push_back(std::move(obj1));
    models.push_back(std::move(obj2));

    models.push_back(
        Model::CreateSphere(vec3(4, 4, 0), 4.0f, Material::Metallic(vec3(1.0f), 0.0f, 1), isProc)
    );

    textures.push_back(
        Texture::LoadTexture(
            "../assets/models/giant_humanoid_plant_monster/textures/tripo_mat_d2b8c9f8_baseColor.jpeg",
            Base::SamplerConfig()
        )
    );

    textures.push_back(
        Texture::LoadTexture(
            "../assets/textures/land_ocean_ice_cloud_2048.png",
            Base::SamplerConfig()
        )
    );

    return std::forward_as_tuple(std::move(models), std::move(textures));
}
