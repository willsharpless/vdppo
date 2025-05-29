import trimesh

scene = trimesh.load("render/f16-c_falcon.glb")

# Render scene to an image (returns a PNG byte buffer)
png_data = scene.save_image(resolution=(800, 600), visible=True)

# Save to file
with open("jet_render.png", "wb") as f:
    f.write(png_data)