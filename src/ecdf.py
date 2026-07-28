from collections import defaultdict
from pathlib import Path
import pickle
from torchvision.transforms import CenterCrop
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt
import numpy as np

from utils import compute_similarity, load_all_models
from image_processing import linear_to_srgb_ldr
from scenes import setup_views, dragon_scene, dog_scene, suzanne_scene, lion_statue_scene, Envmap
from model import Resnet, ResnetSIN, LPIPSVGG, LPIPS, CLIPVision, DINO
models = load_all_models()

# random objects
path = Path('./renders')
envmap_names = ['aloe_farm_shade_house_1k', 'hallstatt4_hd', 'autumn_field']
files = {envmap: [str(f) for f in (path/envmap).glob('*.jpg')] for envmap in envmap_names}
envmaps = [Envmap.GARDEN, Envmap.HALLSTATT, Envmap.SKYBOX]
scenes = [dragon_scene, dog_scene, lion_statue_scene, suzanne_scene]
names = ['dragon', 'dog', 'lion_statue', 'suzanne']

for scene, name in zip(scenes, names):
    sensor_id = 24 if name != 'suzanne' else 0
    nviews = 25 if name != 'suzanne' else 8
    for i, envmap in enumerate(envmaps):
        envmap_name = envmap_names[i]
        scene['emitter'] = {'type': 'envmap', 'filename': envmap.value}
        view = setup_views(nviews)[sensor_id]
        S = mi.load_dict(scene)
        ref = mi.render(S, spp=1024, sensor=view)
        ref = linear_to_srgb_ldr(ref)
        null = defaultdict(list)
        for f in files[envmap_name]:
            img = linear_to_srgb_ldr(mi.TensorXf(np.array(plt.imread(f))))
            for model in models:
                crop = CenterCrop(224) if model.__class__.__name__ in ['DINO', 'CLIPVision'] else None
                sim = compute_similarity(ref, img, model, crop=crop)
                null[str(model)].append(sim)

        with open(f'./ecdf/{name}-{envmap.name.lower()}-ecdf.pickle', 'wb+') as fp:
            pickle.dump(null, fp)
