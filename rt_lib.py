from math import tan, cosh
from random import uniform
from PIL import Image
from multiprocessing import Process, cpu_count
import os

##__CONSTANT__##
PI_4 = 12.566370614359172


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
	
	def to_rgb_q(self,bpc,exp, gamma):
		qv = 2**bpc -1
		col =( ((self*exp).clip())**gamma )*qv
		return min(qv,int(col.r)), min(qv,int(col.g)), min(qv, int(col.b))



class Material:
	def __init__(self, col=Color(1,1,1)):
		self.color = col
		self.roughness = 1
		self.type = None
		self.adpt_smpls = False
		
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
		
		self.bkg = Color(0,0,0)
		self.f_name = "render"
		self.exposure = 1
		self.gamma = 1/2.2
	

class Camera:
	def __init__(self, loc, v_at,up, fov, W,H):
		self.loc = loc
		self.v_at = v_at
		
		self.forw = (v_at - loc).normalize()
		self.right = self.forw.cross(up).normalize()
		self.up = self.forw.cross(self.right).normalize()
		
		self.fov = fov
		self.near_clip = 1e-3
		self.far_clip = 1e5

		self.img_PlaneH = tan((0.01745329251994 * self.fov)) #deg to radians(math module slow)
		self.img_PlaneW = self.img_PlaneH * W/H		
	
	def cast(self,scene,x,y):					
		ray_dir = self.forw  +  (self.right*(x*self.img_PlaneW))  +  (self.up*(y*self.img_PlaneH))		
		return Ray(self.loc, ray_dir.normalize())


class Light:	
	def __init__(self, loc,ints, col=Color(1,1,1) ):
		self.loc = loc
		self.ints = ints
		self.color = col
		self.shadows = 0


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
		self.n = n
		self.material = Material()
		self.name = "Plane"

	def intersect(self, ray):
		ang = self.n.dot(ray.dir)
		if ang:
			return (((self.loc - ray.loc).dot(self.n)) / ang)
		else:
			return None

	def normal(self, v):
		return self.n


			
#__Shader__			
class Shader:
	
	bias = 0.001
	
	def init(self):
		self.obj = None
		self.hit_loc = None
		self.objects = None
		self.light_color = None
		self.light_ints = None

	
	def diffuse(self, N,L):
		diff_col = self.obj.material.color * (self.light_ints * max(0, N.dot(L)) ) + (self.obj.material.color*.001)
		return diff_col
	
	def spec(self, N,H):
		spec_col = self.light_color * (((1-self.obj.material.roughness)*self.light_ints) * max(0, N.dot(H))**self.obj.material.spec_const() )
		return spec_col
	
	def shadows(self,L):
		sdw_ray = Ray(self.hit_loc + L*self.bias, L)		
		for obj in self.objects:
			hit_pos = obj.intersect(sdw_ray)
			if hit_pos and hit_pos > self.bias:
				return 1
		else:
			return None

		
	def reflect(self,N,L):
		rn1, rn2, rn3 = uniform(-1,1),uniform(-1,1),uniform(-1,1)
		Rd = Vec(rn1, rn2, rn3).normalize()
		R = ( (-L) -  (N *( 2*N.dot(-L))) ).normalize()
		
		R_Dir = ((N+Rd)*(self.obj.material.roughness)) + (R * (1-self.obj.material.roughness))
		R_Dir.normalize()
		R_Loc = self.hit_loc + (R_Dir*self.bias)	
		ref_ray = Ray(R_Loc, R_Dir)
		
		for obj in self.objects:
			hit_pos = obj.intersect(ref_ray)
			if hit_pos and hit_pos > self.bias:
				return obj, self.hit_loc + (R_Dir *hit_pos)
		else:
			return None, None



###______FUNCTIONS_____###
def Ray_Trace(scene, obj, shader, hit_v, V, depth):
	
	if obj and depth <= scene.depth:
		
		shader.obj, shader.hit_loc, shader.mat = obj, hit_v, obj.material
		N = obj.normal(hit_v)
		V = V.normalize()
		surf_col = Color(0,0,0)
	
		for light in scene.lights:

			#_vectors_and_conts_for_calc
			L = (light.loc - hit_v)
			Light_dist = L.mag
			L= L.normalize()
			light_ints_at = light.ints / (PI_4 *Light_dist**2)
				
			#_shader_obj
			shader.light_color, shader.light_ints = (light.color, light_ints_at)
			
			#_shadows
			if light.shadows and shader.shadows(L):
				return surf_col + obj.material.color * .01

			#_actually_calculating_color
			mat_type = obj.material.type
			if mat_type == "DIFF + GLOSS":
				H = (L+V).normalize()
				surf_col += (shader.diffuse(N,L) + shader.spec(N,H))
			
			elif mat_type == "GLOSS":
				H = (L+V).normalize()
				surf_col += shader.spec(N,H)
			
			else:
				surf_col += shader.diffuse(N,L)
		
		#_Reflections
		cond = scene.reflections and (depth < scene.depth)
		if cond:
			refl_col = Color(0,0,0)			
			smpls_ct = max(1, scene.samples*obj.material.roughness)
						
			for _ in range(smpls_ct):
				refl_O, refl_V = shader.reflect(N,L)
				if refl_O:
					ic_col = Ray_Trace(scene, refl_O, shader, refl_V, V, depth+1)
					if ic_col:
						refl_col += ic_col
			surf_col += (refl_col/smpls_ct)
		return surf_col
	else:
		return None

def DivideRanges(Q, parts):
	pSize = Q//parts	
	st, end = 0, pSize
	Div_lst = []
	for _ in range(parts):
		Div = range(st, end)
		Div_lst.append(Div)
		st, end = end, end+pSize
	if Q-st:
		Div_lst[parts-1] = range(st-pSize, Q)
	return Div_lst, pSize


def Minutes(t):
	s = round((t%60), 4)
	m = int(t)//60
	return (f"\n{m}m : {s}s")


def ColorAt(scene,Shdr,x,y):	
	def  nearest_hit(objs, f_clip, ray):
		min_hit = f_clip
		n_obj = None
	
		for obj in objs:
			t = obj.intersect(ray)
				
			if t and t <= min_hit:
				min_hit = t
				n_obj = obj
				
		return n_obj, min_hit

	
	x_ray, y_ray = (2*x)/scene.W -1, (2*y)/scene.H -1	
	cam_ray = scene.camera.cast(scene, x_ray, y_ray)
		
	nearest_obj, hit_dist = nearest_hit(scene.objects, scene.camera.far_clip, cam_ray)
				
	if nearest_obj:
		hit_vec = (cam_ray.loc + (cam_ray.dir*hit_dist))
		col = Ray_Trace(scene, nearest_obj, Shdr, hit_vec, (cam_ray.loc - hit_vec), 0)
		return (col.to_rgb_q(8, scene.exposure, scene.gamma))
	else:
		return None



def Render(scene, thds=8):
	
	
	#MultiProcessingStuff_for_rendering_parts_of_image
	def RangeRender(y_lim, t_id, scene, shader):
		
		H = 1 + (y_lim[-1] - y_lim[0])
		i_temp = Image.new("RGB", (scene.W, H))
		for y in y_lim:
			
			for x in range(scene.W):
				col = ColorAt(scene, shader, x,y)
				if col:
					i_temp.putpixel((x, (y-y_lim[0])), (col))					
		i_temp.save(f".Temp/temp{t_id}.png")


	#RenderBody
	Img = Image.new("RGB", (scene.W, scene.H))	
	shader = Shader()
	shader.objects = scene.objects
	
	RngLst,blockH = DivideRanges(scene.H, thds) #divisions of height
	TaskLst = [] #all_processes
	
	print(RngLst)
	for idx, RO in enumerate(RngLst):
		p = Process(target=RangeRender, args=(RO, idx, scene, shader))
		TaskLst.append(p)
	
	for task in TaskLst:
		task.start()
	for task in TaskLst:
		task.join()
	
	for i in range(len(TaskLst)):
		file_path = f".Temp/temp{i}.png"
		im = Image.open(file_path)
		Img.paste(im,(0, i*blockH))
		os.remove(file_path)
	
	return Img
