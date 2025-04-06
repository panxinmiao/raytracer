struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_index: u32,
};

struct Triangle {
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,

    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,

    material_index: u32,
};

// struct Material {
//     color: vec3f,
//     specular_or_ior: f32,
// };

struct Material {
    color: vec3<f32>,           // 基础颜色
    metallic: f32,              // 金属度
    emissive: vec3<f32>,        // 自发光颜色
    roughness: f32,             // 粗糙度
    ior: f32,                // 折射率
};

struct Scatter {
    attenuation: vec3f,
    ray: Ray,
}

fn scatter(ray: Ray, hit: Intersection, material: Material) -> Scatter {
    let incident = normalize(ray.direction);
    let view_dir = -incident;
    
    let incident_dot_normal = dot(incident, hit.normal);
    let is_front_face = incident_dot_normal < 0.;

    let normal = select(-hit.normal, hit.normal, is_front_face);

    var scatter_dir: vec3<f32>;
    var attenuation: vec3<f32>;

    if material.ior > 0.0 {
        // 处理折射
        let ior = material.ior;
        let ref_ratio = select(ior, 1. / ior, is_front_face);
        let cos_theta = abs(incident_dot_normal);

        let cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
        if cannot_refract || schlick(cos_theta, ref_ratio) > rand_f32() {
            scatter_dir = reflect(incident, normal);
        } else {
            scatter_dir = refract(incident, normal, ref_ratio);
        }
        attenuation = material.color;

    } else {
        // 对于镜面材质使用完美反射
        if material.roughness < 0.001 {
            if material.metallic > 0.999 {
                scatter_dir = reflect(incident, normal);
                attenuation = material.color;
            } else {
                scatter_dir = reflect(incident, normal);
                let cos_theta = abs(dot(view_dir, normal));
                let r0 = vec3f(0.04);
                attenuation = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
            }
        } else {
            if material.metallic > 0.5{
                // 使用GGX分布来采样方向而不是余弦分布
                let alpha = max(material.roughness * material.roughness, 0.001);
                // 使用基于反射方向的采样
                let reflected = reflect(incident, normal);
                
                // 添加基于roughness的随机扰动，但确保扰动更集中在反射方向周围
                scatter_dir = normalize(reflected + alpha * sample_sphere());
                
                // 确保散射方向在正确的半球
                if dot(scatter_dir, normal) < 0.0 {
                    scatter_dir = scatter_dir - 2.0 * dot(scatter_dir, normal) * normal;
                }
            }else{
                // 随机采样光线方向（基于粗糙度）
                scatter_dir = normalize(normal + material.roughness * sample_sphere());
            }

            scatter_dir = select(scatter_dir, normal, all(scatter_dir == vec3f(0.0)));
            // 使用 Disney BRDF 计算衰减值
            let brdf = disney_brdf(normal, view_dir, scatter_dir, material);

            // 计算混合 PDF
            let half_vector = normalize(view_dir + scatter_dir);
            let pdf = mixed_pdf(normal, scatter_dir, half_vector, view_dir, material);

            // 避免除以很小的 PDF 值
            attenuation = brdf / max(pdf, EPSILON);

        }

    }

    let output_ray = Ray(point_on_ray(ray, hit.t), scatter_dir);
    return Scatter(attenuation, output_ray);
}

fn disney_diffuse(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32) -> f32 {
    let nl = max(dot(normal, light_dir), 0.0);
    if nl <= 0.0 {
        return 0.0;
    }
    let nv = max(dot(normal, view_dir), 0.0);
    let lh = max(dot(light_dir, view_dir), 0.0);

    let fd90 = 0.5 + 2.0 * roughness * lh * lh;
    let light_scatter = mix(1.0, fd90, pow(1.0 - nl, 5.0));
    let view_scatter = mix(1.0, fd90, pow(1.0 - nv, 5.0));

    return light_scatter * view_scatter * nl / PI;
}

fn disney_specular(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    
    let half_vector = normalize(view_dir + light_dir);
    let nh = max(dot(normal, half_vector), 0.0);
    let nv = max(dot(normal, view_dir), 0.0);
    let nl = max(dot(normal, light_dir), 0.0);
    let vh = max(dot(view_dir, half_vector), 0.0);

    // GGX Normal Distribution Function
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = nh * nh * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * denom * denom);

    // Schlick Fresnel Approximation
    let F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);

    // Smith Geometry Function
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));

    return (D * F * G) / (4.0 * nv * nl + 0.001);
}

fn disney_brdf(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, material: Material) -> vec3<f32> {
    let f0 = mix(vec3f(0.04), material.color, material.metallic); // 菲涅尔反射率
    let diffuse_color = material.color * (1.0 - material.metallic); // 漫反射颜色

    let diffuse = disney_diffuse(normal, view_dir, light_dir, material.roughness) * diffuse_color;
    let specular = disney_specular(normal, view_dir, light_dir, material.roughness, f0);

    return diffuse + specular;
}

fn lambertian_pdf(normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    let cos_theta = max(dot(normal, direction), 0.0);
    return cos_theta / PI;
}

fn ggx_pdf(normal: vec3<f32>, half_vector: vec3<f32>, view_dir: vec3<f32>, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let nh = max(dot(normal, half_vector), 0.0);
    let vh = max(dot(view_dir, half_vector), 0.0);

    // GGX Normal Distribution Function
    let D = alpha2 / (PI * pow(nh * nh * (alpha2 - 1.0) + 1.0, 2.0));

    // PDF for GGX sampling
    return (D * nh) / (4.0 * vh + 0.001);
}

fn mixed_pdf(normal: vec3<f32>, direction: vec3<f32>, half_vector: vec3<f32>, view_dir: vec3<f32>, material: Material) -> f32 {
    let lambertian = lambertian_pdf(normal, direction);
    let ggx = ggx_pdf(normal, half_vector, view_dir, max(material.roughness, 0.001));

    // Fresnel term (Schlick approximation)
    // let F = fresnel_schlick(max(dot(view_dir, half_vector), 0.0), vec3f(0.04));

    // 根据材质属性调整PDF混合权重
    let metallic_factor = material.metallic;
    
    // 对于金属材质，更多地倾向于使用GGX PDF
    let weight = mix(0.5, 0.9, metallic_factor);

    // Mix the two PDFs
    return max(mix(lambertian, ggx, weight), 0.001);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// fn scatter(input_ray: Ray, hit: Intersection, material: Material) -> Scatter {
//     // let scattered = reflect(input_ray.direction, hit.normal);
//     let incident = normalize(input_ray.direction);
//     let incident_dot_normal = dot(incident, hit.normal);
//     let is_front_face = incident_dot_normal < 0.;
//     let N = select(-hit.normal, hit.normal, is_front_face);

//     let cos_theta = abs(incident_dot_normal);

//     // `ior` only has meaning if the material is transmissive.
//     let is_transmissive = material.specular_or_ior < 0.;
//     let is_specular = material.specular_or_ior > 0.;


//     var scattered: vec3f;
//     if is_specular {
//     // if is_specular || (is_transmissive && cannot_refract) {
//         scattered = reflect(incident, N);
//         scattered = normalize(scattered) + sample_sphere() * (1. - EPSILON) * (1- material.specular_or_ior);
//         if dot(scattered, N) < 0. {
//             // stop the ray from going inside the object
//             return Scatter(vec3f(0.), Ray(vec3f(0.), vec3f(0.)));
//         }
//     } else if is_transmissive {
//         let ior = abs(material.specular_or_ior);
//         let ref_ratio = select(ior, 1. / ior, is_front_face);
//         let cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
//         if cannot_refract || schlick(cos_theta, ref_ratio) > rand_f32() {
//             scattered = reflect(incident, N);
//         } else {
//             scattered = refract(incident, N, ref_ratio);
//         }

//     } else {
//         scattered = sample_lambertian(N);
//     }
//     let output_ray = Ray(point_on_ray(input_ray, hit.t), scattered);
//     let attenuation = material.color;


//     return Scatter(attenuation, output_ray);
// }

fn schlick(cosine: f32, ref_ratio: f32) -> f32 {
    var r0 = (1.0 - ref_ratio) / (1.0 + ref_ratio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn sample_lambertian(normal: vec3f) -> vec3f {
    return normal + sample_sphere() * (1. - EPSILON);
}