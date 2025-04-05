struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_index: u32,
};

struct Material {
    color: vec3f,
    specular_or_ior: f32,
};

struct Scatter {
    attenuation: vec3f,
    ray: Ray,
}

fn scatter(input_ray: Ray, hit: Intersection, material: Material) -> Scatter {
    // let scattered = reflect(input_ray.direction, hit.normal);
    let incident = normalize(input_ray.direction);
    let incident_dot_normal = dot(incident, hit.normal);
    let is_front_face = incident_dot_normal < 0.;
    let N = select(-hit.normal, hit.normal, is_front_face);

    let cos_theta = abs(incident_dot_normal);

    // `ior` only has meaning if the material is transmissive.
    let is_transmissive = material.specular_or_ior < 0.;
    let is_specular = material.specular_or_ior > 0.;


    var scattered: vec3f;
    if is_specular {
    // if is_specular || (is_transmissive && cannot_refract) {
        scattered = reflect(incident, N);
        scattered = normalize(scattered) + sample_sphere() * (1. - EPSILON) * (1- material.specular_or_ior);
        if dot(scattered, N) < 0. {
            // stop the ray from going inside the object
            return Scatter(vec3f(0.), Ray(vec3f(0.), vec3f(0.)));
        }
    } else if is_transmissive {
        let ior = abs(material.specular_or_ior);
        let ref_ratio = select(ior, 1. / ior, is_front_face);
        let cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
        if cannot_refract || schlick(cos_theta, ref_ratio) > rand_f32() {
            scattered = reflect(incident, N);
        } else {
            scattered = refract(incident, N, ref_ratio);
        }

    } else {
        scattered = sample_lambertian(N);
    }
    let output_ray = Ray(point_on_ray(input_ray, hit.t), scattered);
    let attenuation = material.color;


    return Scatter(attenuation, output_ray);
}

fn schlick(cosine: f32, ref_ratio: f32) -> f32 {
    var r0 = (1.0 - ref_ratio) / (1.0 + ref_ratio);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn sample_lambertian(normal: vec3f) -> vec3f {
    return normal + sample_sphere() * (1. - EPSILON);
}