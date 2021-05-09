from rt_lib import *

W, H =  200, 120

#__Objects
BR = 2

ball1 = Sphere(Vec(-2,0,0), BR)
ball2 = Sphere(Vec(2,0,0), BR)


#materials
plastic = Material(Color(1, 0,0))
chalk = Material()

plastic.roughness = 0
plastic.type = "DIFF + GLOSS"


ball1.material = plastic
ball2.material = chalk

obj_lst = [ball1, ball2]


#_lights
l1_loc = Vec(-20,-20, 20)
L1 = Light(l1_loc, 18000)       #Key
L2 = Light(Vec(0,-20,1), 120) #Fill
L1.shadows = 1

light_lst = [L1, L2]

#_camera
c_loc = Vec(0, -30, 10)
c_at = Vec(0,0,0)
c_up = Vec(0,0,1)
c_fov = 8


cam = Camera(c_loc, c_at,c_up,c_fov,W,H)

##__SCENE__##
scene_data = Scene(obj_lst, cam, light_lst, W,H)

scene_data.reflections = 1
#scene_data.samples = 1
