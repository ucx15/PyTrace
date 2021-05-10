from Scene_files.All_features_Scene import *
from PIL import Image
from time import time


####_____MAIN____####

img = Image.new("RGB", (W,H))

t1 = time()
img = render_loop(scene_data, img)
t2 = time()
total_t = t2-t1

print(total_t)
file_name = f"Output/ {scene_data.f_name} {total_t}"
out_frmt = "png"
img.save(f"{file_name}.{out_frmt}")
