from RTLib import *

W, H =  2160, 1080

#__OBJECTS

#initialize
BR = 1.95        #radius
Sp1 = Sphere(Vec(-4, 0, 0), BR)
Sp2 = Sphere(Vec( 0.0, 0, 0), BR)
Sp3 = Sphere(Vec( 4, 0, 0), BR)
Pln = Plane(Vec(0, 0, -BR), n=Vec(0, 0, 1))

#set_materials
red = Material(Color(1, 0,0))
metal = Material(Color(0,0,0))
blue = Material(Color(0,.07, 1), rough=1, reflect=False)
gry = Material(Color(.6,.6,.6), reflect=False)


Sp1.material = red
Sp2.material = metal
Sp3.material = blue
Pln.material = gry

obj_lst = [Sp1, Sp2, Sp3, Pln]


#_LIGHTS
L1 = Light(Vec(-20, -20, 20), 18000, shadows=True) #Key
L2 = Light(Vec(0,-20,1), 200) #Fill

light_lst = [L1, L2]

#_CAMERA
c_loc = Vec(0, -10, 3)
c_at = Vec(0,0,0)
c_up = Vec(0,0,1)
c_fov = 24 #in degrees

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##

scene = Scene(obj_lst, cam, light_lst, W,H)

scene.reflections = True
scene.samples = 1 #reflection samples (1-128), can be >128 @cost of speed
scene.depth = 1
