from pythreejs import *
from IPython.display import display
from urllib.request import pathname2url
import os

if __name__ == "__main__":
    model_path = "render/f16-c_falcon.glb"
    url = "file://" + pathname2url(os.path.abspath(model_path))

    gltf_loader = GLTFLoader()
    gltf_loader.load(url)

    scene = Scene(children=[gltf_loader], background="#222222")

    camera = PerspectiveCamera(position=[5, 5, 5], up=[0, 0, 1], fov=60)
    controller = OrbitControls(controlling=camera)
    renderer = Renderer(camera=camera, scene=scene, controls=[controller],
                        width=800, height=600)
    display(renderer)
