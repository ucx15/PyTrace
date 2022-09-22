from RTLib import *


W, H =  800, 600

#__OBJECTS

#initialize
SP = Sphere(Vec(0,0,0), 1)
SP2 = Sphere(Vec(1,-1,-0.8),0.2)
GP = Plane(Vec(0,0,-1), Vec(0,0,1))

objs = [SP,SP2,GP]


#_materials
ball = Material(Color(1,.15,0))
smball = Material(Color(0,.3,1))
floor = Material(Color(.05,.05,.05) , reflect=0)

SP.material = ball
SP2.material = smball
GP.material = floor

#_LIGHTS
L = Light(Vec(8,-10,20), 20000, shadows=True,
			type="AREA",At=Vec(0,0,0),
			edge=4, sdwsmpls= 16)

light_lst = [L]



#_CAMERA
c_loc = Vec(0,-7.5, 1.5)
c_at = SP.loc
c_up = Vec(0,0,1)
c_fov = 18

cam = Camera(c_loc,c_at,c_up,c_fov,W,H)


##__SCENE__##
scene = Scene(objs,cam, light_lst, W,H)
scene.reflections = True