from rt_lib import *
from Spheres import *
from PIL import Image
from time import time



####__==__RT_LIBRARY___==__####

####_____END_of_LIB_____####


####_____MAIN____####

img = Image.new("RGB", (W,H), color=scene_data.bkg.to_rgb_q(8, 1, scene_data.gamma))

t1 = time()
img = render_loop(scene_data, img)
t2 = time()
total_t = t2-t1

print(total_t)
file_name = "Output/"+ scene_data.f_name +str(total_t)
out_frmt = "png"
img.save(file_name+"."+out_frmt)
