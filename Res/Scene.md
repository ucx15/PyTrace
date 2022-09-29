# Usage of a Scene File
A scene file is a representation of scene which is to be rendered.
</br>
It can be named as anything with a __`.json `__ extension.

### `Structure.json`:

```
{	
	"Scene":
	{...},

	"Objects":
	[
		{...},
		...
	],

	"Materials":
	[
		{...},
		...
	],

	"Lights":
	[
		{...},
		...
	],

	"Camera":
	{...}
}
```

### `Example.json`:
```
{	
	"Scene":
	{
		"size"       : [800, 800],
		"name"       : "Example",
		"reflections": 1,
		"depth"      : 1
	},

	"Objects":
	[
		{
			"type" : "sphere",
			"uuid" : "sp",
			"loc"  : [0, 0, 0],
			"rad"  : 1.95
		},
		{
			"type" : "plane",
			"uuid" : "pln",
			"loc"  : [0, 0, -1.95],
			"nor"  : [0, 0, 1]
		}
	],

	"Materials":
	[

		{
			"uuid"  : "metal",
			"color" : [0,0,0],
			"assign": ["sp"]
		},
		{
			"uuid"   : "gry" ,
			"color"  : [0.6, 0.6, 0.6],
			"reflect": 0,
			"assign" : ["pln"]
		}
	],

	"Lights":
	[
		{
			"loc"    : [-20, -20, 20],
			"power"  : 18000,
			"shadows": 1
		},
		{
			"loc"  : [0, -20, 1],
			"power": 200
		}
	],

	"Camera": 
	{
		"loc": [0, -10, 3],
		"at" : [0,0,0],
		"up" : [0,0,1],
		"fov": 24
	}
}
```

## Properties:
Given are the different properties that can be used to define a scene and its components-

<hr>

⚠️ `uuid` cannot be left empty!
### Scene
* `size : list[int]` 
* `reflections : bool`
* `depth : int`
* `samples : int`
* `exposure : float`
* `curve : str`
* `gamma : float`
* `crop : map` _e.g._ `{"x":[180,300], "y":[20,100]}`
* `name : str`

### Material
`assign` _is list of_ `uuids` _of objects to which this material is to be applied)_

* `assign : list[str]`
* `uuid : str`
* `color : list[float]`
* `shade : bool`
* `flat : bool`
* `roughness : float`
* `reflect : bool`
* `shadows : bool`
* `type : str`


### Light
* `loc : list[float]`
* `ints : float`
* `color : list[float]`
* `shadows : bool`
* `type : str`
* `At : list[float]`
* `Up : list[float]`
* `edge : float`
* `sdwsmpls : int`
* `length : int`
* `width : int`


### Camera
* `loc : list[float]`
* `v_at : list[float]`
* `v_up : list[float]`
* `fov : float`
* `near_clip : float`
* `far_clip : float`
* `type : str`

<hr>

## Object
### Global Settings
* `uuid : str`
* `type : str`

### Sphere
* `loc : list[float]`
* `r : float`

### Plane
* `loc : list[float]`
* `nor : list[float]`

### Triangle 
* `a : list[float]`
* `b : list[float]`
* `c : list[float]`

### Quad
* `a : list[float]`
* `b : list[float]`
* `c : list[float]`
* `d : list[float]`



### Cube
* `loc : list[float]`
* `r : float`

Above properties can also be find in respective classes.
