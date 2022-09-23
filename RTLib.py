from math import radians, tan, pi, log
from random import uniform
from multiprocessing import Process, Manager
from time import time
from PIL import Image


###_____CLASSES_____###

#__vectors__
class Vec:
	def __init__(self, i,j,k):
		self.i = i
		self.j = j
		self.k = k
		
	def __str__(self):
		return f"{round(self.i,2)} {round(self.j,2)} {round(self.k,2)}"
	def __neg__(self):
		return Vec(-self.i, -self.j, -self.k)
	def __add__(self, v):
		return Vec((self.i + v.i), (self.j + v.j), (self.k + v.k))	
	def __iadd__(self, v):
		return Vec((self.i + v.i), (self.j + v.j), (self.k + v.k))	
	def __sub__(self, v):
		return Vec((self.i - v.i), (self.j - v.j), (self.k - v.k))	
	def __mul__(self, scl):
		return Vec((self.i *scl), (self.j *scl), (self.k *scl))	
	def __truediv__(self, scl):
		return Vec((self.i /scl), (self.j /scl), (self.k /scl))
	def __eq__(self, v):
		if (self.i == v.i) and (self.j==v.j) and (self.k==v.k):
			return True
		return False
	
	def mag(self):
		return self.dot(self) ** .5
	def mag_sq(self):
		return self.dot(self)
	
	def normalize(self):
		mag = self.mag()
		return Vec(self.i/mag, self.j/mag, self.k/mag)		
	def dot(self, v):
		return (self.i * v.i) + (self.j*v.j) + (self.k*v.k)	
	def cross(self, v):
		return Vec((self.j*v.k - self.k*v.j), -(self.i*v.k - self.k*v.i), (self.i*v.j - self.j*v.i))	
	@staticmethod
	def RandVect():
		return Vec(uniform(-1,1), uniform(-1,1), uniform(-1,1))


class Ray:
	def __init__(self, loc, dir):
		self.loc = loc
		self.dir = dir.normalize()
	
	def __str__(self):
		return ("loc: " +str(self.loc)+"\ndir: "+str(self.dir))



#____Colors_&_Materials____

class Color:
	def __init__(self, r,g,b):
		self.r = r
		self.g = g
		self.b = b
	
	def __str__(self):
		return f"{self.r} {self.g} {self.b}"
	def __add__(self, c):
		return Color((self.r + c.r), (self.g + c.g), (self.b + c.b))
	def __iadd__(self, c):
		return Color((self.r + c.r), (self.g + c.g), (self.b + c.b))
	def __mul__(self, scl):
		return Color((self.r *scl) , (self.g *scl), (self.b *scl))
	def __truediv__(self, scl):
		return self*(1/scl)
	def __eq__(self, c):
		if ((self.r == c.r) and (self.g==c.g) and (self.b==c.b)):
			return True
		return False
	

class Encoder:
	
	#ACES
	@staticmethod
	def ACES(x):
		a,b,c,d,e = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
		return max(0, (x*(x+a) - b) / (x * (x*c + d) + e))
	
	#_HYBRID-LOG-GAMMA
	@staticmethod	
	def HLG(x):	
		a,b,c = 0.17883277, 0.28466892, 0.55991073
		return (.5*(x)**.5) if (x<= 1) else (a*log(x-b) + c)
	
	#ToneMap
	@staticmethod
	def Tonemap(C, fn):
		return Color(fn(C.r), fn(C.g), fn(C.b))
	
	@staticmethod
	def Gamma(C, gm):
		return Color(C.r**gm, C.g**gm, C.b**gm)
	
	@staticmethod
	def quantize(c, bpc=8):		
		qv = int(2**bpc -1)
		r,g,b= ((min(int(c.r*qv),qv)),
				(min(int(c.g*qv),qv)),
				(min(int(c.b*qv),qv)) )
		return r,g,b

	@staticmethod
	def encode(c, crv, ev, gm):
		if crv:
			c = Encoder.Tonemap(c*ev, crv)
		return Encoder.quantize(Encoder.Gamma(c, gm))	


class Material:
	def __init__(self, col=Color(1,1,1), shade = True, flat=False, reflect = True, rough=0, shadows=True, type = None):
		self.color = col
		self.shade = shade
		self.flat = flat
		self.roughness = rough
		self.reflect = reflect
		self.shadows = shadows
		self.type = type

	def spec_const(self):
		return  (350 * (1 - self.roughness))


#__scene_camera_lights__
class Scene:
	def __init__(self, objs, cam, lights, w,h):
		self.objects = objs
		self.camera = cam
		self.lights = lights
		self.W = w
		self.H = h
		
		self.reflections = False
		self.depth = 1
		self.samples = 1


		self.exposure = 1
		self.curve = Encoder.ACES
		self.gamma = 1/2.2
		self.crop = False
		
		self.bkg = (0,0,0,255)
		self.f_name = "render"


class Camera:
	def __init__(self, loc, v_at,up, fov, W,H):
		self.loc = loc
		self.v_at = v_at
		
		self.forw = (v_at - loc).normalize()
		self.right = self.forw.cross(up).normalize()
		self.up = self.forw.cross(self.right).normalize()
		
		self.fov = radians(fov)
		self.near_clip = 1e-10
		self.far_clip = 1e10

		self.iPH = tan(self.fov)
		self.iPW = self.iPH * W/H		
		
		self.type = "PERS"
				
	def CamRay(self,x,y):
		RD = (x*self.iPW)
		UD = (y*self.iPH)
		
		ray_dir = (self.forw + (self.right*RD) + (self.up*UD))
		
		if self.type == "PERS":
			return Ray(self.loc, ray_dir)
		elif self.type == "ORTHO":
			return Ray(ray_dir,self.forw)
			

class Light:	
	def __init__(self, loc,ints, color=Color(1,1,1),shadows=False,type="POINT",At=None,Up=Vec(0,0,1),edge=1,sdwsmpls=25,length=None,width=None):
		self.loc = loc
		self.ints = ints
		self.color = color
		self.shadows = shadows
		self.type = type
		
		self.At = At
		self.Up = Up
		self.edge = edge
		self.sdwsmpls = sdwsmpls
		self.l = length
		self.w = width

	def Generate(self):
		pW,pH = self.l, self.w
		
		if pW and pH:
			AR = pW/pH
			W = int((self.sdwsmpls*AR)**.5)
			H = int((self.sdwsmpls/AR)**.5)
		else:
			W = H = int(self.sdwsmpls**.5)
			pW = pH = self.edge

		Lights = []
		Sints = self.ints/self.sdwsmpls
		
		F = (self.At - self.loc).normalize()
		R = (F.cross(self.Up)).normalize()
		U = (F.cross(R)).normalize()

		
		for yv in range(H):
			y =  (2*yv/H)-1
			for xv in range(W):
				x = (2*xv/W)-1

				RD = (x*pW)
				UD = (y*pH)
		
				Pos = self.loc + (F + (R*RD) + (U*UD))
				sL = Light(Pos,Sints,color=self.color,shadows=self.shadows)
				Lights.append(sL)
		return Lights


#__GEOMETRY__

#_primitives___
class Sphere:
	def __init__(self, loc, r):
		self.loc = loc
		self.r = r
		self.material = Material()
		self.type = "PRIMITIVE"
		self.name = "Sphere"
		
	def intersect(self, ray):
		oc = ray.loc - self.loc		
		b = 2 * (oc.dot(ray.dir))
		c = oc.dot(oc) - self.r**2		
		d = b**2 - 4*c
		if d >= 0:
			return (-b - (d)**.5)/2
		else:
			return None
			
	def normal(self, v):
		return Vec((v.i - self.loc.i), (v.j - self.loc.j), (v.k - self.loc.k)).normalize()


class Plane:
	def __init__(self, loc, n):
		self.loc = loc
		self.n = n.normalize()
		self.material = Material()
		self.type="PRIMITIVE"
		self.name = "Plane"

	def intersect(self, ray):
		ang = self.n.dot(ray.dir.normalize())
		if ang:
			inPln = ((self.loc - ray.loc).dot(self.n)) / (ang)
			return inPln if inPln >= 0 else None

	def normal(self, _):
		return self.n


class Triangle():
	def __init__(self, a,b,c):
		self.a = a
		self.b = b
		self.c = c
		
		self.material = Material()
		self.type = "PRIMITIVE"
		self.name = "Triangle"
		
		self.AB = (b-a)
		self.B = (c-b)
		self.AC = (c-a)
		
		self.n = ((self.AB).cross(self.AC)).normalize()
	

	def intersect(self, ray):
		pvec = ray.dir.cross(self.AC)	
		det = self.AB.dot(pvec)
	
		if det < 0.000001:
			return None
	
		invDet = 1.0 / det
		tvec = ray.loc - self.a
		u = tvec.dot(pvec) * invDet
	
		if u < 0 or u > 1:
			return None
	
		qvec = tvec.cross(self.AB)
		v = ray.dir.dot(qvec) * invDet
	
		if v < 0 or ((u + v) > 1):
			return None
	
		return self.AC.dot(qvec) * invDet
	

	def normal(self, _):
		return self.n



#_composite____	   (non_direct_intersection_function)
	
class Quad:
	def __init__(self, a,b,c,d):
		self.T1 = Triangle(a,b,d)
		self.T2 = Triangle(d,b,c)
		
		self.Tris = [self.T1,self.T2]
		self.material = Material()
		self.type = "COMPOSE"
		self.name = "Quad"

	def mat_apply(self):
		self.T1.material = self.material
		self.T2.material = self.material


class Cube:
	def __init__(self,loc,r):
		self.loc = loc
		self.r = r
		self.material = Material()
		self.type = "COMPOSE"
		self.name = "Cube"
		
		a = loc + Vec(-r,-r,-r)
		b = loc + Vec(+r,-r,-r)
		c = loc + Vec(+r,-r,+r)
		d = loc + Vec(-r,-r,+r)
		e = loc + Vec(-r,+r,-r)
		f = loc + Vec(+r,+r,-r)
		g = loc + Vec(+r,+r,+r)
		h = loc + Vec(-r,+r,+r)


		#_front
		T1 = Triangle(a,b,c)
		T2 = Triangle(a,c,d)		
		#_left
		T3 = Triangle(e,a,d)
		T4 = Triangle(e,d,h)		
		#_right
		T5 = Triangle(b,f,g)
		T6 = Triangle(b,g,c)		
		#_back
		T7 = Triangle(f,e,h)
		T8 = Triangle(f,h,g)		
		#_top
		T9 = Triangle(d,c,g)
		T10 = Triangle(d,g,h)		
		#_bottom
		T11 = Triangle(e,f,b)
		T12 = Triangle(e,b,a)
		
		self.Tris = [T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12]
	
	def mat_apply(self):
		for T in self.Tris:
			T.material = self.material

	def visibleTris(self,ray):
		return [T for T in self.Tris if (T.n.dot(ray.dir)<=0) ]


		
#__Shader__
class Shader:
	
	def __init__(self):
		self.obj = None
		self.hit_loc = None
		self.objects = None
		self.light_color = None
		self.light_ints = None
	
	@staticmethod
	def diffuse(N,L, color, l_int):
		return color * (l_int * max(0, N.dot(L)) )
	
	@staticmethod	
	def spec(N,H, roughness, l_int, l_col, spec_const):
		return (l_col * (((1- roughness) * l_int)*
				max(0, N.dot(H))**spec_const() ))

	def isShadow(self,L,Ld):
		sdw_ray = Ray(self.hit_loc + L*0.0001, L)
		hit_pos = nearest_hit(self.objects, Ld, sdw_ray, 1)
		
		if hit_pos and hit_pos > 0.0001:
			return 1

		
	def reflect(self,N,V):
	
		R = ( (-V) -  (N *( 2*N.dot(-V))) ).normalize() #reflection_vector	
		
		R_Dir = (R * (1- self.obj.material.roughness))
		if self.obj.material.roughness:
			R_Dir += (N+Vec.RandVect())*self.obj.material.roughness
		R_Loc = self.hit_loc + (R_Dir*0.0001)	
		
		ref_ray = Ray(R_Loc, R_Dir)
				
		min_hit = 1e90
		min_o = None
		for obj in self.objects:
			hit_pos = obj.intersect(ref_ray)
			if hit_pos and min_hit>= hit_pos > 0.0001:
				min_hit = hit_pos
				min_o = obj
				
		if min_o and min_hit != 1e90:			
			return min_o, self.hit_loc + (R_Dir *min_hit)
		else:
			return None, None


###______FUNCTIONS_____###
#_Utility_functions
def Minutes(t):
	s = round((t%60), 4)
	m = int(t)//60
	return (f"\n{m}m : {s}s")

def DivideRanges(strt, end, parts):
	n = end//parts
	return [range(i, min(end, i+n)) for i in range(strt, end, n)]


#_Main_Functions
def Ray_Trace(scene, obj,hit_pt,V, shader, depth=0):
	
	if obj and (obj.material.shade) and (depth <= scene.depth):
		
		Total_Color = Color(0,0,0)
		shader.obj,shader.hit_loc = obj, hit_pt
		NormalVec = obj.normal(hit_pt).normalize()
		
		
		for LightSr in scene.lights:
	
			L = (LightSr.loc - hit_pt)
			D = L.mag()
			HalfVec = (L + V).normalize()
			L=L.normalize()
	
			light_color = LightSr.color
			light_ints = (LightSr.ints/(4*pi*D*D))
	
			#_Shadow_Detection
			if (LightSr.shadows and shader.isShadow(L,D)):
				Total_Color += obj.material.color *.01
			
			#_Flat_Surface
			elif obj.material.flat:
				
				Total_Color += obj.material.color
			
			#_Shaded_Surface
			else:
				if obj.material.type == "GLOSS":
					Total_Color += shader.spec(NormalVec,
												HalfVec,
												obj.material.roughness,
												light_ints,
												light_color,
												obj.material.spec_const)
					
				elif obj.material.type == "DIFFUSE":
					Total_Color += shader.diffuse(NormalVec,
													L,
													obj.material.color,
													light_ints)
				
				elif obj.material.type == "EMMIT":
					Total_Color += obj.material.color
				else:
					Total_Color += (shader.spec(NormalVec,
												HalfVec,
												obj.material.roughness,
												light_ints,
												light_color,
												obj.material.spec_const) + 
									
									shader.diffuse(NormalVec,
													L,
													obj.material.color,
													light_ints))
				
					
		
		ref_cond = (depth < scene.depth and
						scene.reflections and
						obj.material.reflect and
						not (obj.material.flat) and
						obj.material.type!="EMMIT")
		if ref_cond:
			if scene.samples == 1:
				adpt_smpls = 1
			else:			
				adpt_smpls = max(1, int(scene.samples * obj.material.roughness))
			ref_col = Color(0,0,0)
			
			for _ in range(adpt_smpls):
				Ref_O, Ref_V = shader.reflect(NormalVec, V)
				if Ref_O:
					smpl_col = Ray_Trace(scene, Ref_O, Ref_V,(hit_pt - Ref_V).normalize(), shader, depth+1)
					ref_col+=smpl_col if smpl_col else Color(0,0,0)
			Total_Color += (ref_col/adpt_smpls)
				
		return Total_Color		

	return None


def nearest_hit(objs, f_clip, ray, dist=False):
	min_hit = f_clip
	n_obj = None
	
	for obj in objs:
		t = obj.intersect(ray)
				
		if t and t <= min_hit:
			min_hit = t
			n_obj = obj
	
	if not dist:
		return n_obj, min_hit
	else:
		return min_hit if (n_obj and min_hit != f_clip) else False


def ColorAt(scene,Shdr,x,y):		
	x_ray, y_ray = (2*x)/scene.W -1, (2*y)/scene.H -1
	cam_ray = scene.camera.CamRay(x_ray, y_ray)

	nearest_obj, hit_dist = nearest_hit(scene.objects, scene.camera.far_clip, cam_ray)
					
	if nearest_obj:
		hit_pt = (cam_ray.loc + (cam_ray.dir*hit_dist))		
		PixelColor = Ray_Trace(scene, nearest_obj, hit_pt, (cam_ray.loc - hit_pt), Shdr)
		if PixelColor:
			return PixelColor


def Renderer(w,h, rw,rh, ImgList, scene, shader):	
	Block = Image.new("RGBA", (w,h) )

	for y in rh:
		Prog = round((100*(y)/(rh[-1])),2)
		print(f"{Prog}", end="\r")	

		for x in rw:
			col = ColorAt(scene, shader, x,y)
			if col:
				Block.putpixel((x,y), Encoder.encode(col, scene.curve, scene.exposure, scene.gamma))
	ImgList.append(Block)



def Render(scene, thds=8):
	#__RENDER_BODY__

	#_resolving_AreaLights_to_pointLights
	LightLst = []
	for L in scene.lights:
		if L.type == "AREA":
			LightLst.extend(L.Generate())
		else:
			LightLst.append(L)
	
	scene.lights = LightLst

	#_resolving_objects_to_primitives
	Object_List = []
	for O in scene.objects:
		if O.type == "COMPOSE":
			O.mat_apply()
			Object_List.extend(O.Tris)
		elif O.type == "PRIMITIVE":
			Object_List.append(O)

	#_Assign_Shader	
	shader = Shader()			
	scene.objects = shader.objects = Object_List


	#_Dividing_height_to_smallerChunks
	if scene.crop:
		CrpW, CrpH = scene.crop["x"], scene.crop["y"]
		ImgW,ImgH = (CrpW[1]-CrpW[0]), (CrpH[1]-CrpH[0])
		Render_W = range(CrpW[0], CrpW[1])
		RngLst = DivideRanges(CrpH[0],CrpH[1], thds)
	else:
		ImgW,ImgH = scene.W, scene.H
		Render_W = range(scene.W)	
		RngLst = DivideRanges(0, scene.H, thds) #divisions of height


	#_Assign_Chunks_to_different_processes
	TaskLst = []
	ImgList = Manager().list()
	Img = Image.new("RGBA", (ImgW, ImgH), scene.bkg)	


	for Render_H in (RngLst):
		p = Process(target=Renderer, args=(ImgW,ImgH,Render_W,Render_H, ImgList, scene, shader))
		TaskLst.append(p)
	
	T1 = time()
	#_Start_Rendering
	for task in TaskLst:
		task.start()
	
	for task in TaskLst:
		task.join()
	
	for chnk in ImgList:
		Img.paste(chnk, (0,0), chnk)
	
	T2 = time()

	#_Timers	
	TTS = T2 - T1 #TotalTimeSeconds
	TTM = Minutes(TTS) #TotalTimeMinutes	
	print(f"\nTotal: \t{TTS}\n{TTM}")
	
	Img.save(f"Output/{scene.f_name} {TTS}.png")
