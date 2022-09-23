from RTLib import *

W, H = 1000, 1000

#____#_Objects_#_______#

Body = Sphere(Vec(0, 0, 0), 1)
Head = Sphere(Vec(0, 0, 1.5), 0.5)
#head_features
Ez, Ey, Ex = 1.6, -0.35, 0.16
LEye = Sphere(Vec(-Ex, Ey, Ez), 0.2)
REye = Sphere(Vec(+Ex, Ey, Ez), 0.2)
LBEye = Sphere(Vec(-Ex-.04, Ey-0.15, Ez), 0.1)
RBEye = Sphere(Vec(+Ex+.04, Ey-0.15, Ez), 0.1)

M1 = Sphere(Vec(-0.02, -0.28, 1.1), 0.04)
M2 = Sphere(Vec(0.02, -0.28, 1.1), 0.04)
M3 = Sphere(Vec(-0.055, -0.3, 1.12), 0.04)
M4 = Sphere(Vec(0.055, -0.3, 1.12), 0.04)
M5 = Sphere(Vec(-0.08, -0.31, 1.14), 0.04)
M6 = Sphere(Vec(0.08, -0.31, 1.14), 0.04)

LHand = Sphere(Vec(-1, 0, 0.35), 0.5)
RHand = Sphere(Vec(1, 0, 0.35), 0.5)

LLeg = Sphere(Vec(-0.5, 0, -0.75), 0.56)
RLeg = Sphere(Vec(0.5, 0, -0.75), 0.56)

Earth = Sphere(Vec(0, 0, -1.4-10), 10)


obj_lst = [RHand, LHand,
			RLeg, LLeg,
			Body, Head,
			LEye, REye, LBEye, RBEye,
			M1,M2,M3,M4,M5,M6,
			Earth]


#_____#_Materials_#______
sd = False

body = Material(Color(0, 0.15, 1), rough=.5, flat=sd)

armLeg = Material(Color(1, 0.5, 0.25), rough=.5, flat=sd)

eye = Material(Color(1, 1, 1),
				rough= 0.2, reflect= False, flat=sd)

pupil = Material(Color(0.05, 0.05, 0.05),
				rough=0, flat=sd)

mouth = Material(Color(1, 0, 0.2),
				rough=.9, flat=sd)

eth = Material(Color(0, 0.8, 0.2),
				type="DIFFUSE", rough=1, reflect=False, flat=sd)


sky = (64, 128, 180)

#assign
Body.material = body
Head.material = LHand.material = RHand.material = LLeg.material = RLeg.material = armLeg
LEye.material = REye.material = eye
LBEye.material = RBEye.material = pupil
M1.material = M2.material = M3.material = M4.material =M5.material = M6.material = mouth

Earth.material = eth

#______#_Lights_#______
L1L = Vec(2, -10, 3); L1it = 1400
L1 = Light(L1L, L1it)
L1.shadows = True
light_lst = [L1]


#_Camera

c_loc = Vec(0, -5, 0)
c_at = Body.loc
c_up = Vec(0, 0, 1)
fov = 30

cam = Camera(c_loc,
			c_at,
			c_up,
			fov,
			W, H)


scene = Scene(obj_lst,
			cam,
			light_lst,
			W, H)

scene.reflections = False
scene.samples = 32
scene.depth = 1
scene.bkg = sky