image_dir="../../data/train"
resized_dir=$image_dir"/resized_images"
for image in $image_dir/*.jpg; do
	convert $image -resize 256x256\! $resized_dir/$(basename $image)
done	
