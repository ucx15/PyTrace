from PIL import Image
from multiprocessing import Process, Manager

import pyximport; pyximport.install()

from All_Classes import *


##__CONSTANT__##
PI_4 = 12.566370614359172


###______FUNCTIONS_____###

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
			sd_flg = False
			if light.shadows and shader.shadows(L):
				surf_col += obj.material.color * .01
				sd_flg = True
			
			if sd_flg == False:
				#_actually_calculating_color
				mat_type = obj.material.type
				if mat_type == "DIFFUSE":
					surf_col += shader.diffuse(N,L)
				
				elif mat_type == "GLOSS":
					H = (L+V).normalize()
					surf_col += shader.spec(N,H)
				
				else:
					H = (L+V).normalize()
					surf_col += (shader.diffuse(N,L) + shader.spec(N,H))

			
		#_Reflections
		cond = scene.reflections and (depth < scene.depth)
		if cond:
			refl_col = Color(0,0,0)			
			smpls_ct = max(1, round(scene.samples*obj.material.roughness))
						
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
			Prog = round(100*(y)/(y_lim[-1]),2)
			print(t_id,Prog,end="\r")
			for x in range(scene.W):
				col = ColorAt(scene, shader, x,y)
				if col:
					i_temp.putpixel((x, (y-y_lim[0])), (col))
		ImgDict[t_id] = i_temp


	#RenderBody
	Img = Image.new("RGB", (scene.W, scene.H))	
	shader = Shader()
	shader.objects = scene.objects
	
	RngLst,blockH = DivideRanges(scene.H, thds) #divisions of height
	TaskLst = [] #all_processes
	
	Link = Manager()
	ImgDict = Link.dict()
	
	for idx, RO in enumerate(RngLst):
		p = Process(target=RangeRender, args=(RO, idx, scene, shader))
		TaskLst.append(p)

	for task in TaskLst:
		task.start()

	for task in TaskLst:
		task.join()
	
	for key in range(thds):
		tImg = ImgDict[key]
		Img.paste(tImg, (0,key*blockH))

	return Img
