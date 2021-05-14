from Library.rt_lib_noC import *

W, H =  2000, 1200

#__OBJECTS

#initialize
BR = 2        #radius
Sp1 = Sphere(Vec(-4.1,0,0), BR)
Sp2 = Sphere(Vec(0,0,0), BR)
Sp3 = Sphere(Vec(4.1,0,0), BR)
Sp4 = Sphere(Vec(0,0,-BR-99999), 99999) #Plane obj not worked; used sphere as a plane


#set_materials
red = Material(Color(1, 0,0))
metal = Material()
white = Material()
blue = Material(Color(0,.07, 1))

white.roughness = metal.roughness = 0
red.roughness = 0


white.type = red.type = "DIFF + GLOSS"
metal.type = "GLOSS"


Sp1.material = red
Sp2.material = metal
Sp3.material = blue
Sp4.material = white

obj_lst = [ Sp1,Sp2, Sp3, Sp4]


#_LIGHTS
L1 = Light(Vec(-20, -20, 20), 12000)     #Key
L2 = Light(Vec(0,-20,1), 150) #Fill
L1.shadows = 1

light_lst = [L1, L2]

#_CAMERA
c_loc = Vec(0, -10, 3)
c_at = Sp2.loc
c_up = Vec(0,0,1)
c_fov = 22

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##

scene = Scene(obj_lst, cam, light_lst, W,H)

scene.reflections = True
scene.samples = 32 #reflection samples (1-128), can be >128 @cost of speed
scene.depth = 1   #reflection depth (0-2), can be >2 @cost of speed
