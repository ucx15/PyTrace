from RTLib import *

W, H =  1920, 1080

#__OBJECTS

#initialize
BR = 2        #radius
Sp1 = Sphere(Vec(-4.1,0,0), BR)
Sp2 = Sphere(Vec(0,0,0), BR)
Sp3 = Sphere(Vec(4.1,0,0), BR)
P = Plane(Vec(0, 0, -BR), Vec(0, 0, 1))

#set_materials
red = Material(Color(1, 0,0), rough=0)
metal = Material(type="GLOSS",rough=0)
white = Material(reflect=False)
blue = Material(Color(0,.07, 1), rough=.1)


Sp1.material = red
Sp2.material = metal
Sp3.material = blue
P.material = white

obj_lst = [Sp1,Sp2, Sp3, P]


#_LIGHTS
L1 = Light(Vec(-20, -20, 20), 2.5E4)     #Key
L2 = Light(Vec(0,-20,1), 300) #Fill
L1.shadows = 1

light_lst = [L1, L2]

#_CAMERA
c_loc = Vec(0, -10, 3)
c_at = Sp2.loc
c_up = Vec(0,0,1)
c_fov = 24

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##

scene = Scene(obj_lst, cam, light_lst, W,H)

scene.reflections = True
scene.samples = 16 #reflection samples (1-128), can be >128 @cost of speed
scene.depth = 2   #reflection depth (0-2), can be >2 @cost of speed
scene.exposure = 1
#scene.crop = {"x":[180,300], "y":[20,100]}