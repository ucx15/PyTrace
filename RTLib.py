from math import tan, cosh, radians, pi, log
from random import uniform
from PIL import Image
from multiprocessing import Process, cpu_count, Manager



###_____CLASSES_____###

#__vectors__
class Vec:
	def __init__(self, i,j,k):
		self.i = i
		self.j = j
		self.k = k
		self.mag = (i*i  +  j*j  +  k*k)**.5
		
	def __str__(self):
		return f"{round(self.i,2)} {round(self.j,2)} {round(self.k,2)}"
	def __neg__(self):
		return Vec(-self.i, -self.j, -self.k)
	def __add__(self, v):
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
			
	def normalize(self):
		return Vec(self.i/self.mag, self.j/self.mag, self.k/self.mag)		
	def dot(self, v):
		return (self.i * v.i) + (self.j*v.j) + (self.k*v.k)	
	def cross(self, v):
		return Vec((self.j*v.k - self.k*v.j), -(self.i*v.k - self.k*v.i), (self.i*v.j - self.j*v.i))	
	def angle_bw(self, v):
		return (cosh((self.dot(v)/ (self.mag()*v.mag())) ) )


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
		return Color((self.r /scl) , (self.g /scl), (self.b /scl))	
	def __pow__(self, scl):
		return Color((self.r **scl), (self.g**scl), (self.b**scl))
	def __eq__(self, c):
		if self.r == c.r and self.g==c.g and self.b==c.b:
			return True
		return False

	def clip(self, a=0, b=1):
		return Color(min(b, max(a, self.r)), min(b, max(a, self.g)), min(b, max(a, self.b)) )
		
	def mix_mul(self, c):
		return Color((self.r * c.r), (self.g*c.g), (self.b*c.b))

	def to_hex_q(self, chnl):
		bpc = 4*chnl
		rq, gq, bq = self.quantize(bpc)
		rh = max(hex(rq)[2:], "0"*chnl)
		gh = max(hex(gq)[2:], "0"*chnl)
		bh = max(hex(bq)[2:], "0"*chnl)
		return f"#{rh}{gh}{bh}"
	
	
	def quantize(self,bpc=8):
		
		qv = 2**bpc -1
		return ( min(int(self.r*qv),qv),
				min(int(self.g*qv),qv),
				min(int(self.b*qv),qv) )
		
	
	def HLG_Curve(self, exp=1):
	
		def logc(E):
			a,b,c = 0.17883277, 0.28466892, 0.55991073
			
			if E <= 1:
				Ed = .5*(E)**.5	
			else:
				Ed = a*log(E-b) + c
			return Ed
		
		self = self*exp
		self.r = logc(self.r)
		self.g = logc(self.g)
		self.b = logc(self.b)
	
		return self

	
	def Gamma_Curve(self, exp=1, gamma=.45):
		return (self*exp)**gamma



class Material:
	def __init__(self, col=Color(1,1,1), shade = True, flat=False, reflect = True, rough=1, shadows=True, type = None):
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
		
		self.bpc = 8
		self.exposure = 1
		self.gamma = 1/2.2
		self.curve = "HLG"
		self.crop = False
		
		self.bkg = Color(0,0,0)
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

		self.img_PlaneH = tan(self.fov)
		self.img_PlaneW = self.img_PlaneH * W/H		
	
	def CamRay(self,scene,x,y):					
		ray_dir = self.forw  +  (self.right*(x*self.img_PlaneW))  +  (self.up*(y*self.img_PlaneH))		
		return Ray(self.loc, ray_dir.normalize())


class Light:	
	def __init__(self, loc,ints, col=Color(1,1,1),shadows=False):
		self.loc = loc
		self.ints = ints
		self.color = col
		self.shadows = shadows


#__GEOMETRY__
class Sphere:
	def __init__(self, loc, r):
		self.loc = loc
		self.r = r
		self.material = Material()
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
		self.name = "Plane"

	def intersect(self, ray):
		ang = self.n.dot(ray.dir.normalize())
		if ang:
			inPln = ((self.loc - ray.loc).dot(self.n)) / (ang)
			return inPln if inPln >= 0 else None


	def normal(self, v):
		return self.n


class Triangle():
	def __init__(self, a,b,c):
		self.a = a
		self.b = b
		self.c = c
		
		self.A = (b-a)
		self.B = (c-b)
		self.C = (a-c)
		
		self.n = ((b-a).cross(c-a)).normalize()
		self.plane = Plane(a, self.n)
			
		self.material = Material()
		self.name = "Triangle"
	
	def inTris(self,point):
			
			pA = point-self.a
			pB = point-self.b
			pC = point-self.c
			inT =  (((self.A.cross(pA)).dot(self.n))>=0 and
					((self.B.cross(pB)).dot(self.n))>=0 and
					((self.C.cross(pC)).dot(self.n))>=0 )
			if inT:
				return True
			else:
				return False

		
	def intersect(self, ray):		
		p_dist = self.plane.intersect(ray)
		
		if p_dist:
			pt = ray.loc + ray.dir*p_dist
			
			if self.inTris(pt):
				return p_dist
			return None
		return None
	
	def normal(self, v):
		return self.n


class Cube:
	def __init__(self,loc,r):
		self.loc = loc
		self.r = r
		self.material = Material()
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
	
	def init(self):
		self.obj = None
		self.hit_loc = None
		self.objects = None
		self.light_color = None
		self.light_ints = None
	
	def diffuse(self, N,L):
		diff_col = self.obj.material.color * (self.light_ints * max(0, N.dot(L)) )
		return diff_col
	
	def spec(self, N,H):
		spec_col = self.light_color * (((1-self.obj.material.roughness)*self.light_ints) * max(0, N.dot(H))**self.obj.material.spec_const() )
		return spec_col
	
	def isShadow(self,L):
		sdw_ray = Ray(self.hit_loc + L*0.0001, L)		
		for obj in self.objects:
			hit_pos = obj.intersect(sdw_ray)
			if hit_pos and hit_pos > 0.0001:
				return 1
		else:
			return None

		
	def reflect(self,N,V):
		Rv = Vec(uniform(-1,1), uniform(-1,1), uniform(-1,1))
		
		R = ( (-V) -  (N *( 2*N.dot(-V))) ).normalize()
		
		R_Dir = ((N+Rv)*(self.obj.material.roughness)) + (R * (1-self.obj.material.roughness))
		R_Dir.normalize()
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


def Ray_Trace(scene, obj,hit_pt,V, shader, depth=0):
	
	if obj and (obj.material.shade) and (depth <= scene.depth):
		
		Total_Color = Color(0,0,0)
		shader.obj,shader.hit_loc = obj, hit_pt
		NormalVec = obj.normal(hit_pt).normalize()
		
		
		for Light in scene.lights:
	
			L = (Light.loc - hit_pt)
			D = L.mag
			HalfVec = (L + V).normalize()
			L=L.normalize()
	
			shader.light_color = Light.color
			shader.light_ints = (Light.ints/(4*pi*D*D))
	
			#_Shadow_Detection
			if (Light.shadows and shader.isShadow(L)):
				Total_Color += obj.material.color *.005
			
			#_Flat_Surface
			elif obj.material.flat:
				
				Total_Color += obj.material.color
			
			#_Shaded_Surface
			else:
				if obj.material.type == "GLOSS":
					Total_Color += shader.spec(NormalVec, HalfVec)
					
				elif obj.material.type == "DIFFUSE":
					Total_Color += shader.diffuse(NormalVec, L)
				else:
					Total_Color += (shader.spec(NormalVec, HalfVec) + shader.diffuse(NormalVec, L))

		
		ref_cond = (depth < scene.depth and
					scene.reflections and
					obj.material.reflect and
					not (obj.material.flat))
		if ref_cond:			
			adpt_smpls = max(1, int(scene.samples * obj.material.roughness))
			ref_col = Color(0,0,0)
			
			for _ in range(adpt_smpls):
				Ref_O, Ref_V = shader.reflect(NormalVec, V)
				if Ref_O:
					smpl_col = Ray_Trace(scene, Ref_O, Ref_V,(Ref_V - hit_pt).normalize(), shader, depth+1)
					ref_col+=smpl_col if smpl_col else Color(0,0,0)
			Total_Color += (ref_col/adpt_smpls)
				
		return Total_Color		
	else:
		return None




def Minutes(t):
	s = round((t%60), 4)
	m = int(t)//60
	return (f"\n{m}m : {s}s")


def  nearest_hit(objs, f_clip, ray, dist=True):
	min_hit = f_clip
	n_obj = None
	
	for obj in objs:
		t = obj.intersect(ray)
				
		if t and t <= min_hit:
			min_hit = t
			n_obj = obj
	
	if dist:
		return n_obj, min_hit
	else:
		return True if n_obj else False



def ColorAt(scene,Shdr,x,y):
	
	x_ray, y_ray = (2*x)/scene.W -1, (2*y)/scene.H -1	
	cam_ray = scene.camera.CamRay(scene, x_ray, y_ray)
		
	nearest_obj, hit_dist = nearest_hit(scene.objects, scene.camera.far_clip, cam_ray)
				
	if nearest_obj:
		hit_pt = (cam_ray.loc + (cam_ray.dir*hit_dist))
		col = Ray_Trace(scene, nearest_obj, hit_pt, (cam_ray.loc - hit_pt), Shdr)
		
		if col:
			if scene.curve == "HLG":
				return (col.HLG_Curve(exp=scene.exposure).quantize(scene.bpc))
			elif scene.curve == "GAMMA":
				return (col.Gamma_Curve(exp=scene.exposure).quantize(scene.bpc))			
			elif scene.curve == "LINEAR":
				return (col.quantize(scene.bpc))
			else:
				return (col.quantize(scene.bpc))
	else:
		return None


def Render(scene, thds=8):
	def DivideRanges(rS, rE, parts):
		pSize = (rE-rS)//parts
		Div_lst = []
		for s in range(rS,rE,pSize):		
			e = s+pSize	
			if (e>rE): e=rE
	
			Div_lst.append(range(s,e))
		return Div_lst, pSize
		
	
	#MultiProcessingStuff_for_rendering_parts_of_image
	def RangeRender(y_lim,x_lim, t_id, scene, shader):
			
		H = 1 + (y_lim[-1] - y_lim[0])
		W = 1 + (x_lim[-1] - x_lim[0])
		i_temp = Image.new("RGB", (W, H) )
		
		for y in y_lim:
			Prog = round((100*(y)/(y_lim[-1])),2)
			print(f"{t_id}\t{Prog}", end="\r")
			
			for x in x_lim:
				col = ColorAt(scene, shader, x,y)
				if col:
					xi,yi = (x-x_lim[0]), (y-y_lim[0])
					i_temp.putpixel((xi,yi), (col))
		ImgDict[t_id] = i_temp
	
	
	#RenderBody
	print(f"Number of Processes: {thds}")
	
	#_resolving_objects_to_primitives	
	for O in scene.objects:
		if O.name == "Cube":
			O.mat_apply()
			scene.objects.extend(O.Tris)
			scene.objects.remove(O)
	
	#_Dividing_height_to_smallerChunks
	if scene.crop:
		CrpW, CrpH = scene.crop["x"], scene.crop["y"]
		ImgW,ImgH = (CrpW[1]-CrpW[0]), (CrpH[1]-CrpH[0])
		Render_W = range(CrpW[0], CrpW[1])
		RngLst, blockH = DivideRanges(CrpH[0],CrpH[1], thds)
	else:
		ImgW,ImgH = scene.W, scene.H
		Render_W = range(scene.W)	
		RngLst,blockH = DivideRanges(0, scene.H, thds) #divisions of height
	
	#_Assign_Shader	
	shader = Shader()			
	shader.objects = scene.objects
	
	#_Assign_Main_Returnable_Image
		
	Img = Image.new("RGB", (ImgW, ImgH))	
	
	#_Assign_Chunks_to_different_processes
	TaskLst = []
		
	Link = Manager()
	ImgDict = Link.dict()
		
	for idx, Render_H in enumerate(RngLst):
		p = Process(target=RangeRender, args=(Render_H,Render_W, idx, scene, shader))
		TaskLst.append(p)

	for task in TaskLst:
		task.start()
	
	for task in TaskLst:
		task.join()
	
	#_Merging_Chunks_to_single_Image	
	for key in range(len(RngLst)):
		tImg = ImgDict[key]
		Img.paste(tImg, (0,key*blockH))	

	return Img
