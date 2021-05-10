from rt_lib import *

W, H =  150, 75

#__OBJECTS

#initialize
BR = 2        #radius
Sp1 = Sphere(Vec(-4.1,0,0), BR)
Sp2 = Sphere(Vec(0,0,0), BR)
Sp3 = Sphere(Vec(4.1,0,0), BR)
Sp4 = Sphere(Vec(0,0,-BR-99999), 99999) #Plane obj not worked; used sphere as a plane


#set_materials
red_shiny = Material(Color(1, 0,0))
white_metal = Material()
white_diffused = Material()
blue_diffused = Material(Color(0,.07, 1))

red_shiny.roughness = 0.1
white_metal.roughness = 0

red_shiny.type = "DIFF + GLOSS"
white_metal.type = "GLOSS"
white_shiny.type = "DIFF + GLOSS"

white_metal.adpt_smpls = True

Sp1.material = white_metal
Sp2.material = red_shiny
Sp3.material = blue_diffused
Sp4.material = white_diffused

obj_lst = [Sp1, Sp2, Sp3, Sp4]


#_LIGHTS
L1 = Light(Vec(-10, -20, 20), 8000)     #Key
L2 = Light(Vec(0,-20,1), 150) #Fill
L1.shadows = 1

light_lst = [L1, L2]

#_CAMERA
c_loc = Vec(0, -10, 2)
c_at = Sp2.loc
c_up = Vec(0,0,1)
c_fov = 24

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##

scene = Scene(obj_lst, cam, light_lst, W,H)

scene.reflections = True
scene.samples = 8  #reflection samples (1-128), can be >128 @cost of speed
scene.depth = 1   #reflection depth (0-2), can be >2 @cost of speed
