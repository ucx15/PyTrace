from RTLib import Render


def main():
	path = "Res/Scene_files/All_features_Scene.json"
	# path = "Res/Scene_files_local/example.json"
	
	Render(path)


if __name__ == "__main__":
	main()


# TODO: Implement a basic logging class
# TODO: Raw output of floating point buffer
# TODO: Fresnel effect for Reflection and refraction
# TODO: Show output as pygame surface for progress tracking
# TODO: GUI for scene creation and editing
# TODO: Energy Conservation
# TODO: Maybe use numpy arrays