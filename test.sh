export USE_MATX_DISTANCES=1

nsys profile -o run8 /home/hphi344/Documents/GS-DBSCAN-Analysis/../GS-DBSCAN-CPP/build-release/GS-DBSCAN --datasetFilename /home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin --outputFilename results.json --n 70000 --d 784 --D 1024 --minPts 50 --k 5 --m 50 --eps 0.11 --alpha 1.2 --distancesBatchSize -1 --distanceMetric COSINE --clusterBlockSize 256 --datasetDType f32 --miniBatchSize 10000 --ABatchSize 10000 --BBatchSize 128 --normBatchSize 10000 --clusterOnCpu --needToNormalize --verbose --useBatchNorm --timeIt



export USE_MATX_DISTANCES=0

nsys profile -o run9 /home/hphi344/Documents/GS-DBSCAN-Analysis/../GS-DBSCAN-CPP/build-release/GS-DBSCAN --datasetFilename /home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin --outputFilename results.json --n 70000 --d 784 --D 1024 --minPts 50 --k 5 --m 50 --eps 0.11 --alpha 1.2 --distancesBatchSize -1 --distanceMetric COSINE --clusterBlockSize 256 --datasetDType f32 --miniBatchSize 10000 --ABatchSize 10000 --BBatchSize 128 --normBatchSize 10000 --clusterOnCpu --needToNormalize --verbose --useBatchNorm --timeIt