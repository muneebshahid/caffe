if [ "$1" = "finetune" ]
then 
	$CAFFE_ROOT/build/tools/caffe.bin train --solver=/home/shahidm/code/domain_adaptation/network/places_caffenet/places205CNN_solver.prototxt --weights=/home/shahidm/code/domain_adaptation/data/models/alexnet_places/places205CNN_iter_300000_upgraded.caffemodel --log_dir=/home/shahidm/code/domain_adaptation/data/logs/ -gpu all
elif [ "$1" = "scratch" ]
then	
	$CAFFE_ROOT/build/tools/caffe.bin train --solver=/home/shahidm/code/domain_adaptation/network/places_caffenet/places205CNN_solver.prototxt --log_dir=/home/shahidm/code/domain_adaptation/data/logs/ -gpu all
else
	echo "wrong params"
fi
