import bpy
import sys

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Load the .glb file (you can also use .gltf or .obj)
bpy.ops.import_scene.gltf(filepath="render/f16-c_falcon.glb")

# Set render resolution and camera location
scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = True  # optional: transparent BG

# Optional: add a camera if the model doesn't include one
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (5, -5, 3)
cam_obj.rotation_euler = (1.1, 0, 0.8)  # pitch, roll, yaw

# Optional: lighting
light = bpy.data.lights.new(name="Light", type='POINT')
light_obj = bpy.data.objects.new(name="Light", object_data=light)
light_obj.location = (5, -5, 5)
scene.collection.objects.link(light_obj)

# Set output path
scene.render.filepath = "rendered_jet.png"

# Render the scene
bpy.ops.render.render(write_still=True)
