import json
from math import radians, tan, pi
from multiprocessing import Pool
from random import uniform
from time import perf_counter
from PIL import Image
import pygame


###_____CLASSES_____###

Vec = pygame.Vector3
Color = pygame.Vector3

class Ray:
	def __init__(self, loc, dir):
		self.loc = loc
		self.dir = dir.normalize()

	def __str__(self):
		return ("loc: " +str(self.loc)+"\ndir: "+str(self.dir))


#____Material____
class Encoder:

	#ACES
	@staticmethod
	def ACES(x):
		a,b,c,d,e = 0.0245786, 0.000090537, 0.983729, 0.4329510, 0.238081
		return max(0, (x*(x+a) - b) / (x * (x*c + d) + e))

	@staticmethod
	def Gamma(C, gm):
		return Color(C.x**gm, C.y**gm, C.z**gm)

	@staticmethod
	def quantize(c, bpc=8):
		qv = int(2**bpc -1)
		c *= qv
		return ((min(int(c.x), qv)),
				(min(int(c.y), qv)),
				(min(int(c.z), qv)) )


	@staticmethod
	def encode(c, crv = "", ev = 1, gm = 2.2):
		c *= ev

		if crv == "ACES":
			c = Color(Encoder.ACES(c.x), Encoder.ACES(c.y), Encoder.ACES(c.z))

		return Encoder.quantize(Encoder.Gamma(c, gm))


class Material:
	properties = [
		"assign",
		"uuid",
		"color",
		"shade",
		"flat",
		"roughness",
		"reflect",
		"shadows",
		"type"]

	def __init__(self, color=Color(1,1,1), shade = True, flat=False, reflect = True, roughness=0, shadows=True, type = None):
		self.uuid = ""
		self.color = color
		self.shade = shade
		self.flat = flat
		self.reflect = reflect
		self.roughness = roughness
		self.shadows = shadows
		self.type = type

	def spec_const(self):
		return  (350 * (1 - self.roughness))


# Camera Lights and Scene
class Camera:
	properties = [
		"loc",
		"v_at",
		"v_up",
		"fov",
		"near_clip",
		"far_clip",
		"type"
	]

	def __init__(self, loc = Vec(), v_at = Vec(0,1,0), v_up = Vec(0, 0, 1), fov = 50):
		self.loc  = loc
		self.v_at = v_at
		self.v_up = v_up
		self.fov  = fov
		self.near_clip = 1e-10
		self.far_clip  = 1e10
		self.type = "PERS"

		self.forw = None
		self.right = None
		self.up = None
		self.iPH = None
		self.iPW = None


	def calc_vectors(self, Ar):
		self.forw = (self.v_at - self.loc).normalize()
		self.right = self.forw.cross(self.v_up).normalize()
		self.up = self.forw.cross(self.right).normalize()

		self.fov = radians(self.fov)
		self.iPH = tan(self.fov)
		self.iPW = self.iPH * Ar


	def CamRay(self,x,y, W,H):
		if not self.forw:
			self.calc_vectors(W/H)

		RD = x* self.iPW
		UD = y* self.iPH

		ray_dir = (self.forw + (self.right*RD) + (self.up*UD))

		if self.type == "PERS":
			return Ray(self.loc, ray_dir)
		elif self.type == "ORTHO":
			return Ray(ray_dir, self.forw)


class Light:
	properties = [
		"loc",
		"ints",
		"color",
		"shadows",
		"type",
		"At",
		"Up",
		"edge",
		"sdwsmpls",
		"length",
		"width"]

	def __init__(self, loc = Vec(0,0,0), ints = 100, color=Color(1,1,1),shadows=False,type="POINT",At=None,Up=Vec(0,0,1),edge=1,sdwsmpls=25,length=None,width=None):
		self.loc = loc
		self.ints = ints
		self.color = color
		self.shadows = shadows
		self.type = type

		self.At = At
		self.Up = Up
		self.edge = edge
		self.sdwsmpls = sdwsmpls
		self.length = length
		self.width = width

	def Generate(self):
		pW,pH = self.length, self.width

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


class Scene:
	properties = [
		"size",
		"reflections",
		"depth",
		"samples",
		"exposure",
		"curve",
		"gamma",
		"crop",
		"bkg",
		"name"]

	def __init__(self, objects = [], camera = None, lights = [], size = [1, 1]):
		self.objects = objects
		self.camera = camera
		self.lights = lights
		self.W, self.H = size
		self.reflections = False
		self.depth = 1
		self.samples = 1
		self.exposure = 1
		self.curve = "ACES"
		self.gamma = 1/2.2
		self.crop = False
		self.bkg = Color(0,0,0)
		self.name = "render"

	def Serialize(self):
		# TODO: To be implemented
		pass

	@staticmethod
	def DeSerialize(path : str):
		print(f"\nINFO : Loading File '{path}'")
		
		with open(path, "r") as file:
			try:
				data = json.loads(file.read())
			except json.decoder.JSONDecodeError:
				print(f"ERROR: File '{path}' could not be loaded!\nFATAL: Aborted!\n")
				exit(1)

		# Scene Settings
		scene = Scene()
		raw_scene = data["Scene"]
		for property, val in raw_scene.items():
			match property:
				case "size":
					scene.W, scene.H = val
				case "reflections":
					scene.reflections = val
				case "depth":
					scene.depth = val
				case "samples":
					scene.samples = val
				case "exposure":
					scene.exposure = val
				case "curve":
					scene.curve = val
				case "gamma":
					scene.gamma = 1/val
				case "crop":
					scene.crop = val
				case "bkg":
					scene.bkg = Color(val)
				case "name":
					scene.name = val

		# Camera
		raw_cam = data["Camera"]
		scene.camera = Camera()
		for property, val in raw_cam.items():
			match property:
				case "loc":
					scene.camera.loc = Vec(val)
				case "at":
					scene.camera.v_at = Vec(val)
				case "up":
					scene.camera.v_up = Vec(val)
				case "fov":
					scene.camera.fov = val

		# Objects
		for raw_obj in data["Objects"]:
			obj_type = raw_obj["type"]
			match obj_type:
				case "sphere":
					obj = Sphere(Vec(raw_obj["loc"]), raw_obj["rad"])

				case "plane":
					obj = Plane(Vec(raw_obj["loc"]), Vec(raw_obj["nor"]))

				case "triangle":
					obj = Triangle(Vec(raw_obj["a"]), Vec(raw_obj["b"]), Vec(raw_obj["c"]))

				case "quad":
					obj = Triangle(Vec(raw_obj["a"]), Vec(raw_obj["b"]), Vec(raw_obj["c"]), Vec(raw_obj["d"]))

				case "cube":
					obj = Cube(Vec(raw_obj["loc"]), raw_obj["rad"])

			obj.uuid = raw_obj["uuid"]
			scene.objects.append(obj)

		# Object Materials
		for raw_mat in data["Materials"]:
			mat = Material()
			for property, val in raw_mat.items():
				match property:
					case "uuid":
						mat.uuid = val
					case "color":
						mat.color  = Color(val)
					case "shade":
						mat.shade     = val
					case "flat":
						mat.flat      = val
					case "roughness":
						mat.roughness = val
					case "reflect":
						mat.reflect   = val
					case "shadows":
						mat.shadows   = val
					case "type":
						mat.type      = val

			try:
				assign_grp = raw_mat["assign"]
				for obj in scene.objects:
					if obj.uuid in assign_grp: obj.material = mat
			
			except KeyError: print(f"WARN: Material '{mat.uuid}' is not applied to any object!")

		# Lights
		for raw_lgt in data["Lights"]:
			lgt = Light()
			for property, val in raw_lgt.items():
				match property:
					case "loc":
						lgt.loc = Vec(val)
					case "power":
						lgt.ints = val
					case "color":
						lgt.color = Color(val)
					case "shadows":
						lgt.shadows = val
					case "type":
						lgt.type = val
					case "At":
						lgt.At = Vec(val)
					case "Up":
						lgt.Up = Vec(val)
					case "edge":
						lgt.edge = val
					case "sdwsmpls":
						lgt.sdwsmpls = val
					case "l":
						lgt.l = val
					case "w":
						lgt.w = val
			scene.lights.append(lgt)
		
		print("Scene Loaded.")
		return scene


#__GEOMETRY__

#_primitives___
class Sphere:
	properties = [
		"loc",
		"r"]

	def __init__(self, loc, r):
		self.uuid = ""
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
		return (v - self.loc).normalize()


class Plane:
	properties = [
		"loc",
		"n"]
	def __init__(self, loc, n):
		self.uuid = ""
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
	properties = ["a", "b", "c"]
	def __init__(self, a, b, c):
		self.uuid = ""
		self.a = a
		self.b = b
		self.c = c

		self.material = Material()
		self.type = "PRIMITIVE"
		self.name = "Triangle"

		self.AB = None
		self.BC = None
		self.AC = None


	def intersect(self, ray):
		if not self.AB:
			self.AB = (self.b - self.a)
			self.BC = (self.c - self.b)
			self.AC = (self.c - self.a)

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
		return ((self.AB).cross(self.AC)).normalize()



#_composite____	   (non_direct_intersection_function)

class Quad:
	properties = [
		"a",
		"b",
		"c",
		"d"]

	def __init__(self, a,b,c,d):
		self.uuid = ""
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
	properties = [
		"loc",
		"r"]

	def __init__(self,loc, r):
		self.uuid = ""
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
			R_Dir += (N+RandVect())*self.obj.material.roughness
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

# Utilities
def Minutes(t):
	s = round((t%60), 4)
	m = int(t)//60
	return (f"{m}m : {s}s")

def DivideRanges(strt, end, parts):
	if end - strt < 20:
		return [range(strt, end)]

	n = end//parts
	return [range(i, min(end, i+n)) for i in range(strt, end, n)]

def RandVect():
	return Vec(uniform(-1,1), uniform(-1,1), uniform(-1,1))


# Main Functions
def Ray_Trace(scene, obj,hit_pt,V, shader, depth=0):

	if obj and (obj.material.shade) and (depth <= scene.depth):

		Total_Color = Color(0,0,0)
		shader.obj,shader.hit_loc = obj, hit_pt
		NormalVec = obj.normal(hit_pt).normalize()


		for LightSr in scene.lights:

			L = (LightSr.loc - hit_pt)
			D = L.magnitude()
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
	cam_ray = scene.camera.CamRay(x_ray, y_ray, scene.W, scene.H)

	nearest_obj, hit_dist = nearest_hit(scene.objects, scene.camera.far_clip, cam_ray)

	if nearest_obj:
		hit_pt = (cam_ray.loc + (cam_ray.dir*hit_dist))
		if PixelColor := Ray_Trace(scene, nearest_obj, hit_pt, (cam_ray.loc - hit_pt), Shdr):
			return PixelColor


def Renderer(arglst):
	w,h, rw,rh, scene, shader = arglst

	Block = Image.new("RGBA", (w,h) )
	for y in rh:
		Prog = round((100*(y)/(rh[-1])),2)
		print(f"{Prog}", end="\r")

		for x in rw:
			col = ColorAt(scene, shader, x,y)
			if col:
				Block.putpixel((x,y), Encoder.encode(col, scene.curve, scene.exposure, scene.gamma))
	return Block


def Render(path, thds=4):
	#__RENDER_BODY__

	# Reading scene from file
	T_LOAD = perf_counter()
	scene = Scene.DeSerialize(path)
	T_LOAD = perf_counter() - T_LOAD


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


	# Assigning chunks for processing
	if scene.crop:
		CrpW, CrpH = scene.crop["x"], scene.crop["y"]
		ImgW,ImgH = (CrpW[1]-CrpW[0]), (CrpH[1]-CrpH[0])
		Render_W = range(CrpW[0], CrpW[1])
		RngLst = DivideRanges(CrpH[0],CrpH[1], thds)
	else:
		ImgW,ImgH = scene.W, scene.H
		Render_W = range(scene.W)
		RngLst = DivideRanges(0, scene.H, thds) #divisions of height


	print(f"\nRendering Scene:\t'{scene.name}'")
	print(f"Using {thds} CPU Threads\n")
	if not Object_List:
		print("WARN:\tNo objects found to render!")
		return


	Img = Image.new("RGBA", (ImgW, ImgH),Encoder.encode(scene.bkg, scene.curve, gm=scene.gamma))

	T_RENDER = perf_counter()
	argsList = [[ImgW,ImgH,Render_W, Render_H, scene, shader] for Render_H in RngLst]

	# Rendering
	with Pool(thds) as p:
		ImgList = p.map(Renderer, argsList)

	# Combining chunks
	for chnk in ImgList:
		Img.paste(chnk, (0,0), chnk)

	T_RENDER = perf_counter() - T_RENDER

	#_Timers

	print(f"\n\nLoading Time:\t{T_LOAD:0.4f}s")
	print(f"Total Time:\t{T_RENDER:0.4f}s\n\n{Minutes(T_RENDER)}")

	Img.save(f"Output/{scene.name}_{T_RENDER}.png")
	
	return
