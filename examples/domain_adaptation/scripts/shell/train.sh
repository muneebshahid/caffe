if [ "$1" = "finetune" ]
then
	if [ "$2" = "default" ]
	then
		model=$CAFFE_ROOT"/data/domain_adaptation_data/models/alexnet/places205CNN_iter_300000_upgraded.caffemodel"
		solver=$CAFFE_ROOT"/examples/domain_adaptation/network/alexnet/pretrained/solver.prototxt"
		log_dir=$CAFFE_ROOT"/data/domain_adaptation_data/logs"
		$CAFFE_ROOT/build/tools/caffe train --solver=$solver --weights=$model --log_dir=$log_dir -gpu all 
	elif [ "$2" = "trained" ]
	then
		model="/home/shahidm/code/domain_adaptation/data/models/alexnet_places/trained/snapshots_iter_1000.caffemodel"
		#snapshot="/home/shahidm/code/domain_adaptation/data/models/alexnet_places/trained/snapshots_iter_1000.solverstate"
		$CAFFE_ROOT/build/tools/caffe.bin train --solver=/home/shahidm/code/domain_adaptation/network/places_caffenet/places205CNN_solver.prototxt --weights=$model --log_dir=/home/shahidm/code/domain_adaptation/data/logs/ --snapshot=$snapshot -gpu all 
	else
		echo "$1"
		echo "$2"
		echo "wrong 2nd params "
	fi	
elif [ "$1" = "scratch" ]
then	
	$CAFFE_ROOT/build/tools/caffe.bin train --solver=/home/shahidm/code/domain_adaptation/network/places_caffenet/places_scratch_solver.prototxt --log_dir=/home/shahidm/code/domain_adaptation/data/logs/ -gpu all
else
	echo "wrong first params"
fi
