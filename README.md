# Python_RayTracer

This is a simple implementation of a Ray Tracer in Python 3.

## Dependencies
* PILLOW

## Updates
* Now rt_lib.py is separated to ___All_Classes.pyx___ and ___rt_lib.py___ in _Library_ folder and will be compiled to Cython (May 14,2021 Update)
* Included support for Multi-core Rendering (May 13,2021 Update)


## Instructions
If you want to complie this,just download/clone this repo and run ___main.py___, and if you want to make changes to the scene, edit ___Spheres.py/All_features_Scene.py___ in ___Scene_files___ folder

## Features
### Shading
* Diffuse Shading (Lambert)
* Specular Highlights(Blinn-Phong)
* Specular and Diffused Reflections
* Recursive Reflections with unlimited depth (to recursion limit)
* Sharp Shadows
* Stochastic Sampling for Reflections

### Architecture
* Multiple Lights
* Multiple Objects
* Spheres
* Planes
* Perspective Camera
* Multi-core Rendering (May 13,2021 Update)

## Current Limitations
* Sampling available only for reflections, not for shadows and anti-aliasing
* Only point lights supported
* No soft shadows
* No area Lights
* No GUI

Some Limitations will be removed in coming days..
I did this as a fun project to increase my knowledge in Python(especially OOP)

_THIS IS NOT A SERIOUS PROJECT THAT WILL OVERTAKE ARNOLD OR CYCLES!_
I've got plans for that too ;) 

___Thanks for reading!___
You can also contribute and improve by pulling a request
