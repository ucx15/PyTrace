#!python
#cython: language_level=3

from math import cosh, tan
from random import uniform


class Vec:
	def __init__(self, double i, double j, double k):
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
	def __mul__(self, double scl):
		return Vec((self.i *scl), (self.j *scl), (self.k *scl))	
	def __truediv__(self, double scl):
		return Vec((self.i /scl), (self.j /scl), (self.k /scl))
	def __eq__(self, v):
		if (self.i == v.i) and (self.j==v.j) and (self.k==v.k):
			return True
		else:
			return False
			
	def normalize(self):
		return Vec(self.i/self.mag, self.j/self.mag, self.k/self.mag)		
	def dot(self, v):
		return ((self.i * v.i) + (self.j*v.j) + (self.k*v.k))	
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


class Color:
	def __init__(self,double r, double g, double b):
		self.r = r
		self.g = g
		self.b = b
	
	def __str__(self):
		return f"{self.r} {self.g} {self.b}"
	def __add__(self, c):
		return Color((self.r + c.r), (self.g + c.g), (self.b + c.b))
	def __iadd__(self, c):
		return Color((self.r + c.r), (self.g + c.g), (self.b + c.b))
	def mix_mul(self, c):
		return Color((self.r * c.r), (self.g*c.g), (self.b*c.b))	

	def __mul__(self, double scl):
		return Color((self.r *scl) , (self.g *scl), (self.b *scl))	
	def __truediv__(self, double scl):
		return Color((self.r /scl) , (self.g /scl), (self.b /scl))	
	def __pow__(self, double scl):
		return Color((self.r **scl), (self.g**scl), (self.b**scl))
	def __eq__(self, c):
		if self.r == c.r and self.g==c.g and self.b==c.b:
			return True
		else:
			return False

	def clip(self, a=0, b=1):
		return Color(min(b, max(a, self.r)), min(b, max(a, self.g)), min(b, max(a, self.b)) )
	
	def to_rgb_q(self, int bpc, double exp, double gamma):
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


class Scene:
	def __init__(self, objs, cam, lights, int w, int h):
		self.objects = objs
		self.camera = cam
		self.lights = lights
		self.W = w
		self.H =  h
		
		self.reflections = False
		self.depth = 1
		self.samples = 1
		
		self.bkg = Color(0,0,0)
		self.f_name = "render"
		self.exposure = 1
		self.gamma = 1/2.2
	

class Camera:
	def __init__(self, loc, v_at,up, double fov, int W, int H):
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
	def __init__(self, loc, double ints, col=Color(1,1,1) ):
		self.loc = loc
		self.ints = ints
		self.color = col
		self.shadows = 0

class Sphere:
	def __init__(self, loc, double r):
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
