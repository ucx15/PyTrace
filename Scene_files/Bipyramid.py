from RTLib import *


W, H =  720, 720

#__OBJECTS

#initialize
v1 = Vec(0,0,2)
v2 = Vec(-1,0,1)
v3 = Vec(0,-1,1)
v4 = Vec(1,0,1)
v5 = Vec(0,1,1)
v6 = Vec(0,0,0)

T1 = Triangle(v2,v3,v1)
T2 = Triangle(v3,v4,v1)
T3 = Triangle(v4,v5,v1)
T4 = Triangle(v5,v2,v1)
T5 = Triangle(v2,v6,v3)
T6 = Triangle(v3,v6,v4)
T7 = Triangle(v4,v6,v5)
T8 = Triangle(v5,v6,v2)

Gp = Plane(Vec(0,0,0), Vec(0,0,1))
Sp = Sphere(Vec(-2,.5,.5), .5)

objs = [Sp,Gp,
		T1,T2,T3,T4,
		T5,T6,T7,T8]

#set_materials

sphereMat = Material(Color(.8,.8,.8), rough=0)
PlaneMat = Material(Color(1,.2,0), rough=1, reflect=0)
TrisMat = Material(Color(0,.2,1), rough=0)


#assign
for i in objs:
	i.material = TrisMat
Gp.material = PlaneMat
Sp.material = sphereMat

#_LIGHTS
L1 = Light(Vec(10, -8, 10), 5000)     #Key
L1.shadows = 1

light_lst = [L1]

#_CAMERA
c_loc = Vec(0,-8,4)
c_at = Vec(0,0,1)
c_up = Vec(0,0,1)
c_fov = 18

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)

##__SCENE__##

scene = Scene(objs,cam, light_lst, W,H)

scene.reflections = 1