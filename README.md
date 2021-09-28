# Python_RayTracer

This is a simple implementation of a Ray Tracer in Python 3.

## Dependencies
* PILLOW

## Instructions
If you want to complie this,just download/clone this repo and run ___main.py___, and if you want to make changes to the scene, import chosen scene in main.py file and edit chosen file in ___Scene_files___ folder

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

![All Features](https://github.com/udit-uc828/Python_RayTracer/blob/ae673d7ec9ad4e5cfbf493ff1a8e786e0bd46cd1/Output/%20render%20461.82015800476074.png)
![Area Light,Soft Shadows](https://github.com/udit-uc828/Python_RayTracer/blob/ae673d7ec9ad4e5cfbf493ff1a8e786e0bd46cd1/Output/AreaLight%20Test%201586.15s.png)
![Teddy, Diffused Reflections](https://github.com/udit-uc828/Python_RayTracer/blob/ae673d7ec9ad4e5cfbf493ff1a8e786e0bd46cd1/Output/Teddy(Diffused%20Reflections)%206242.99s.png)
![Diamond, Triangles](https://github.com/udit-uc828/Python_RayTracer/blob/ae673d7ec9ad4e5cfbf493ff1a8e786e0bd46cd1/Output/Diamond(Triangle%20test)%20310.76.png)
![Teddy Flat, 2D Capabilities)(https://github.com/udit-uc828/Python_RayTracer/blob/ae673d7ec9ad4e5cfbf493ff1a8e786e0bd46cd1/Output/Teddy(2D)%20101.68s.png)


## Current Limitations
* Sampling not available for anti-aliasing (SuperSampling)
* No GUI


Some Limitations will be removed in coming days..
I did this as a fun project to increase my knowledge in Python(especially OOP)

___Thanks for reading!___
You can also contribute and improve
