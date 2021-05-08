# Python_RayTracer

This is a simple implementation of a Ray Tracer in Python 3.

## Dependencies
i. PILLOW

## Instructions
If you want to complie this, just run ___main.py___, and if you want to make changes to the scene, edit ___Spheres.py___

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

## Current Limitations
* Sampling available only for reflections, not for shadows and anti-aliasing
* Only point lights supported
* No soft shadows
* Very simple materials (only Diffuse, Glossy or Diffuse+Glossy)
* No area Lights
* No file system handling 
* No GUI

Some Limitations will be removed in coming days..
I did this as a fun project to increase my knowledge in Python(especially OOP)

_THIS IS NOT A SERIOUS PROJECT THAT WILL OVERTAKE ARNOLD OR CYCLES!_
I've got plans for that too ;) 

___Thanks for reading!___
You can also contribute and improve by pulling a request
