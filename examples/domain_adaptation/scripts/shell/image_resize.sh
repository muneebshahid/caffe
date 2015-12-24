image_dir=$PROJ_HOME/"data/train/"
original_dir=$image_dir"/original_images"
resized_dir=$image_dir"/processed_images"
for image in $original_dir/*.jpg; do
	convert $image -resize 256x256\! $resized_dir/$(basename $image)
done	
