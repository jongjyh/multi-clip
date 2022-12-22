if [ $RLAUNCH_REPLICA -eq 0 ] ;then
	echo `hostname -i` > hostname
	export masterip=`cat "hostname"`
else
	until [  -f "hostname" ] 
	do
		# do whatever
		masterip=0
	done
	export masterip=`cat "hostname"`
	# if [ $RLAUNCH_REPLICA -eq `expr $RLAUNCH_REPLICA_TOTAL - 1`  ] ;then
	# 	rm "hostname"
	# fi
fi

bash process_files_per_node.sh 