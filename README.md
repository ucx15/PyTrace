# Python_RayTracer

This is a simple implementation of a Ray Tracer in Python 3.

## Dependencies
* PILLOW

## Instructions
If you want to complie this,just download/clone this repo and run ___main.py___, and if you want to make changes to the scene, edit ___Spheres.py/All_features_Scene.py___ in ___Scene_files___ folder

## ChangeLog

### May 31,2021
* Included Support for Area Lights
* Support for Soft Shadows
* Added Quad as a Composite Object

### May 22,2021 
* Added Triangle as a Primitive Object
* Added Cube as a Composite Object
* Included Support for High Dynamic Range Rendering using Hybrid-Log Gamma Curve
* Included Support for Region Rendering

### May13,2021
* Included support for Multi-core Rendering

## Features
### Shading
* Diffuse Shading (Lambert)
* Specular Highlights(Blinn-Phong)
* Specular and Diffused Reflections
* Recursive Reflections with unlimited depth (to recursion limit)
* Soft Shadows
* Stochastic Sampling for Reflections and Shadows

### Architecture
* Multiple Lights
* Multiple Objects
* Spheres
* Planes
* Triangles
* Perspective and Orthographic Camera
* Multi-core Rendering

## Current Limitations
* Sampling not available for anti-aliasing (SuperSampling)
* No GUI


Some Limitations will be removed in coming days..
I did this as a fun project to increase my knowledge in Python(especially OOP)

_THIS IS NOT A SERIOUS PROJECT THAT WILL OVERTAKE ARNOLD OR CYCLES!_
I've got plans for that too ;) 

___Thanks for reading!___
You can also contribute and improve
