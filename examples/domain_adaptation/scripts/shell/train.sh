model=$CAFFE_ROOT"/../data/models/alexnet/pretrained/places205CNN_iter_300000_upgraded.caffemodel"
network=$CAFFE_ROOT"/examples/domain_adaptation/network/alexnet/"
log_dir=$CAFFE_ROOT"/data/domain_adaptation_data/logs"
if [ "$1" = "finetune" ]
then
	if [ "$2" = "default" ]
	then
		solver=$network"pretrained/solver.prototxt"
		$CAFFE_ROOT/build/tools/caffe train --solver=$solver --weights=$model --log_dir=$log_dir -gpu 0
	elif [ "$2" = "trained" ]
	then
		#snapshot="/home/shahidm/code/domain_adaptation/data/models/alexnet_places/trained/snapshots_iter_1000.solverstate"
		$CAFFE_ROOT/build/tools/caffe train --solver=/home/shahidm/code/domain_adaptation/network/places_caffenet/places205CNN_solver.prototxt --weights=$model --log_dir=/home/shahidm/code/domain_adaptation/data/logs/ --snapshot=$snapshot -gpu all 
	else
		echo "wrong 2nd params "
	fi	
elif [ "$1" = "scratch" ]
then	
	solver=$network"scratch/solver.prototxt"
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --log_dir=$log_dir -gpu 0
else
	echo "wrong first params"
fi
