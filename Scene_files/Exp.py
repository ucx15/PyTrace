from RTLib import *


W, H =  960, 540

#__OBJECTS

#initialize
r1 = 2
r2 = 1.2
r3 = 1.5
r4 = .8

Sp1 = Sphere(Vec(-3, 0, r1), r1)
Sp2 = Sphere(Vec(3.8, 0, r2), r2)
Sp3 = Sphere(Vec(1,  2.5, r3), r3)
Sp4 = Sphere(Vec(1, -1.8, r4),r4)

P = Plane(Vec(0, 0, 0), Vec(0, 0, 1))

#set_materials
blue = Material(Color(0,.1, .8))
orange = Material(Color(1, .2, 0))
yellow = Material(Color(1,1,.1))
green = Material(Color(0.2,1,0.2))
white = Material(Color(1,1,1), rough=1, reflect=0)

Sp1.material = blue
Sp2.material = orange
Sp3.material = yellow
Sp4.material = green
P.material = white

obj_lst = [Sp1, Sp2,Sp3,Sp4, P]


#_LIGHTS
L1 = Light(Vec(.8,-1,8), 500, shadows=True) #Key

light_lst = [L1]

#_CAMERA
c_loc = Vec(0, -8, 4.5)
c_at = Vec(0, 1.8, 0.1)
c_up = Vec(0,0,1)
c_fov = 22.5

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##

scene = Scene(obj_lst, cam, light_lst, W,H)

scene.reflections = 1