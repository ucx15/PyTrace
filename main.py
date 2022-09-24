from RTLib import Render
from Res.Scene_files.All_features_Scene import scene

if __name__ == "__main__":
	print("\nRendering")
	Render(scene)


# TODO: Raw output of floating point buffer
# TODO: JSON serialization for storing scenes
# TODO: Maybe use numpy arrays
# TODO: Energy Conservation
# TODO: Documentation
# TODO: Show output as pygame surface for progress tracking
# TODO: GUI for scene creation and editing
# TODO: Fresnel effect for Reflection and refraction