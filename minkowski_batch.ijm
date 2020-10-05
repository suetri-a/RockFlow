// For getting Minkowski funcationals from GLOW generated images

// Open source directory, ex. "RockFlow\\results\\run_200518-004041\\imgs_stack"
// Will read each folder in this directory (50 folders for 50 different images stacks)
dir1 = getDirectory("Choose source directory");
list = getFileList(dir1)

// Go through each folder (containing a series of 2D images)
for (i=0; i<list.length; i++) {
	// Pick first image to start reading from
	path = dir1 + list[i] + "xy_image000.png";

	// Stack images into 3D volume
	run("Image Sequence...", "open=path number=128 convert sort");

	// 3D opening/closing filter
	//run("Morphological Filters (3D)", "operation=Opening element=Ball x-radius=2 y-radius=2 z-radius=2");

	// Median 3D blur
	run("Median 3D...", "x=1 y=1 z=1");

	// Otsu thresholding on volume, invert
	run("Auto Threshold", "method=Otsu stack");
	run("Invert", "stack");
	
	// Uncomment below to save thresholded image
	//title = getTitle();
	//saveAs("Tiff", dir1+"\\"+title+".tif");

	// Morphological analysis
	run("Analyze Regions 3D", "volume surface_area mean_breadth sphericity euler_number bounding_box centroid equivalent_ellipsoid ellipsoid_elongations max._inscribed surface_area_method=[Crofton (13 dirs.)] euler_connectivity=C26");

	// Save
	title = getTitle();
	saveAs("Results", dir1+"\\"+ title +"-inv-morpho-open.csv");
	run("Close");
	close();
}