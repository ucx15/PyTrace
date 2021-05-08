from rt_lib import *

W, H =  1000, 600

#__Objects
BR = 1.95
#Gpr = 99999
ball1col = Color(1,0,0)
ball2col = Color(0,0,1)
ball3col = Color(0,0,1)

ball1 = Sphere(Vec(-4,0,0), BR)
ball2 = Sphere(Vec(0,0,0), BR)
ball3 = Sphere(Vec(4,0,0), BR)
#gp = Sphere(Vec(0,0,-Gpr-BR-0.1), Gpr)
gp = Plane(Vec(0,0,-BR-.01), Vec(0,0,1))

#materials
dielec_red = Material(ball1col)
dielec_red.roughness = .01
dielec_red.type = "DIFF + GLOSS"

dielec_blue = Material(ball3col)

metal = Material(ball2col)
metal.type = "GLOSS"
metal.roughness = 0


ball1.material = dielec_red
ball2.material = metal
ball3.material = dielec_blue

obj_lst = [ball1,ball2,ball3, gp]


#_lights
l1_loc = Vec(-20,-20, 20)
L1 = Light(l1_loc, 15000)       #Key
L2 = Light(Vec(0,-20,1), 120) #Fill
L1.shadows = 1

light_lst = [L1, L2]

#_camera
c_loc = Vec(0, -30, 10)
c_at = ball2.loc
c_up = Vec(0,0,1)
c_fov = 8


cam = Camera(c_loc,
							c_at,
							c_up,
							c_fov,
							W,H)

##__SCENE__##
scene_data = Scene(obj_lst, cam, light_lst, W,H)

scene_data.reflections = 1
scene_data.samples = 24
#scene_data.depth = 2
