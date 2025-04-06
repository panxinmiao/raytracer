import wgpu
import numpy as np
import random
import math
from wgpu.utils.imgui import Stats
from wgpu.gui.auto import WgpuCanvas, run
from pathlib import Path
from camera import OrbitCamera
from scene import Sphere, Material


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

COMPUTE_WORKGROUP_SIZE_X = 8
COMPUTE_WORKGROUP_SIZE_Y = 8
MAX_BOUNCES = 50


# class Sphere(np.ndarray):
#     def __new__(cls, center=(0, 0, 0), radius=1, material_index=0):
#         na = np.zeros(1, dtype=np.dtype([
#             ('center', np.float32, 3),
#             ('radius', np.float32),
#             ('material_index', np.uint32),
#             ('__padding', np.uint32, 3),
#         ])).view(cls)

#         na['center'] = center
#         na['radius'] = radius
#         na['material_index'] = material_index
#         return na
    
# class Material(np.ndarray):
#     def __new__(cls, color=(1, 1, 1), specular_or_ior=0):
#         na = np.zeros(1, dtype=np.dtype([
#             ('color', np.float32, 3),
#             ('specular_or_ior', np.float32),
#         ])).view(cls)

#         na['color'] = color
#         na['specular_or_ior'] = specular_or_ior
#         return na
    

def create_scene():
    ground = Sphere((0, -1000, 0.0), 1000.0, 0)
    ground_mat = Material((0.5, 0.5, 0.5), 0.0)

    ball_1 = Sphere((0, 0.5, 0), 0.5, 1)
    mat_1 = Material((1.0, 1.0, 1.0), ior=1.5)

    ball_2 = Sphere((-2, 0.5, 0), 0.5, 2)
    mat_2 = Material((0.4, 0.2, 0.1), 0.0)

    ball_3 = Sphere((2, 0.5, 0), 0.5, 3)
    mat_3 = Material((0.7, 0.6, 0.5), 1.0, roughness=0.0)

    spheres = [ground, ball_1, ball_2, ball_3]
    mats = [ground_mat, mat_1, mat_2, mat_3]

    for i in range(400):
        while True:
            sphere = Sphere()
            r = np.random.uniform(0.05, 0.1)
            r = 0.1
            sphere['radius'] = r

            sphere['center'] = [np.random.uniform(-4, 4), r, np.random.uniform(-4, 4)]
            sphere['material_index'] = i + 4

            overlap = False
            for existing_sphere in spheres:
                distance = np.linalg.norm(sphere['center'] - existing_sphere['center'])
                if distance < sphere['radius'] + existing_sphere['radius']:
                    overlap = True
                    break
            
            if not overlap:
                spheres.append(sphere)
                break

        mat = Material()

        choose_mat = random.random()
        if choose_mat < 0.6:
            # diffuse
            mat['color'] = np.random.uniform(0.0, 1.0, 3) * np.random.uniform(0.0, 1.0, 3)
        elif choose_mat < 0.8:
            # metal
            mat['color'] = np.random.uniform(0.5, 1.0, 3)
            mat['roughness'] = np.random.uniform(0.0, 0.5)
            mat['metallic'] = 1.0
        else:
            # glass
            mat['color'] = (1.0, 1.0, 1.0)
            mat['ior'] = np.random.uniform(1.0, 2.5)
        
        mats.append(mat)

    
    spheres_b = np.concatenate(spheres)
    materials_b = np.concatenate(mats)

    return spheres_b, materials_b


def load_wgsl(shader_name):
    shader_dir = Path(__file__).parent / "shaders"
    shader_path = shader_dir / shader_name
    with open(shader_path, "rb") as f:
        return f.read().decode()


canvas = WgpuCanvas(title="Raytracing", size=(IMAGE_WIDTH, IMAGE_HEIGHT))

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(required_features=["texture-adapter-specific-format-features","float32-filterable"])
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
# render_texture_format = wgpu.TextureFormat.rgba8unorm
present_context.configure(device=device, format=render_texture_format)


frame_texture = device.create_texture(
    size=(IMAGE_WIDTH, IMAGE_HEIGHT, 1),
    usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    format=wgpu.TextureFormat.rgba32float,
    dimension="2d",
)

spheres, materials = create_scene()

sphere_buffer = device.create_buffer_with_data(
    data=spheres.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)

materials_buffer = device.create_buffer_with_data(
    data=materials.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)

# random seed buffer
# rng_state= np.arange(0, IMAGE_WIDTH*IMAGE_HEIGHT, dtype=np.uint32)
# rng_state_buffer = device.create_buffer_with_data(
#     data=rng_state.tobytes(),
#     usage=wgpu.BufferUsage.STORAGE,
# )


common_buffer_data = np.zeros(
    (),
    dtype=[
        ("width", "uint32"),
        ("height", "uint32"),
        ("frame_counter", "uint32"),
        ("__padding2", "uint32"),  # padding to 16 bytes
    ],
)


common_buffer = device.create_buffer(
    size=common_buffer_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

camera = OrbitCamera(
    canvas=canvas,
    fov=45.0,
    focal_length=5.0,
    defocus_angle=0.4,
    center=(0.0, 0.0, 0.0),
    distance=5.0,
    attitude=math.pi / 24,
    azimuth=math.pi / 2 * 1.2,
    up=(0.0, 1.0, 0.0),
)

camera_buffer_data = camera.buffer_data

camera_buffer = device.create_buffer_with_data(
    data=camera_buffer_data.tobytes(),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

#Create raytracing compute shader

UTIL = load_wgsl("util.wgsl")
MATERIAL = load_wgsl("material.wgsl")
RAY = load_wgsl("ray1.wgsl")
COMPUTE_SHADER = load_wgsl("ray_tracing1.wgsl")
CAMERA = load_wgsl("camera.wgsl")

ray_tracing_src = "\n".join([UTIL, MATERIAL, RAY, CAMERA, COMPUTE_SHADER])


ray_tracing_pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={
        "module": device.create_shader_module(code=ray_tracing_src),
        "entry_point": "main",
        "constants": {
            "WORKGROUP_SIZE_X": COMPUTE_WORKGROUP_SIZE_X,
            "WORKGROUP_SIZE_Y": COMPUTE_WORKGROUP_SIZE_Y,
            "OBJECTS_COUNT_IN_SCENE": len(spheres),
            "MAX_BOUNCES": MAX_BOUNCES,
        },
    
    },
)


ray_tracing_bind_group = device.create_bind_group(
    layout=ray_tracing_pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": sphere_buffer}},
        {"binding": 1, "resource": {"buffer": materials_buffer}},
        {"binding": 2, "resource": {"buffer": common_buffer}},
        {"binding": 3, "resource": {"buffer": camera_buffer}},
        {"binding": 4, "resource": frame_texture.create_view()},
    ],
)


# Render to the screen
RENDER_SHADER = load_wgsl("render.wgsl")

render_src = "\n".join([RENDER_SHADER])

render_module = device.create_shader_module(
    code=render_src,
)

render_pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": render_module,
        "entry_point": "vs_main",
        "buffers": [],
    },
    fragment={
        "module": render_module,
        "entry_point": "fs_main",
        "targets": [
            {
                "format": render_texture_format,
            }
        ],
    },
)


sampler = device.create_sampler(
    mag_filter="linear",
    min_filter="linear",
    mipmap_filter="linear",
)

render_bind_group = device.create_bind_group(
    layout=render_pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": sampler},
        {"binding": 1, "resource": frame_texture.create_view()},
    ],
)



def do_draw():

    canvas_texture = present_context.get_current_texture()

    # Update camera buffer
    if camera._need_calculate_buffer:
        camera.calculate_buffer_data()

    if camera._need_update_buffer:
        device.queue.write_buffer(camera_buffer, 0, camera_buffer_data.tobytes())
        common_buffer_data["frame_counter"] = 0
        camera._need_update_buffer = False

    # Update uniform buffer
    common_buffer_data["width"] = IMAGE_WIDTH
    common_buffer_data["height"] = IMAGE_HEIGHT
    common_buffer_data["frame_counter"] += 1

    device.queue.write_buffer(common_buffer, 0, common_buffer_data.tobytes())


    command_encoder = device.create_command_encoder()
    

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(ray_tracing_pipeline)
    compute_pass.set_bind_group(0, ray_tracing_bind_group)
    compute_pass.dispatch_workgroups( math.ceil(IMAGE_WIDTH / COMPUTE_WORKGROUP_SIZE_X), math.ceil(IMAGE_HEIGHT / COMPUTE_WORKGROUP_SIZE_Y), 1)
    compute_pass.end()


    render_pass = command_encoder.begin_render_pass(
        color_attachments=[{
            "view": canvas_texture.create_view(),
            "load_op": "clear",
            "store_op": "store",
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, render_bind_group)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()


    device.queue.submit([command_encoder.finish()])


stats = Stats(device, canvas)

def loop():
    # with stats:
    do_draw()
    canvas.request_draw()

if __name__ == "__main__":
    canvas.request_draw(loop)
    run()