# docker build -t openinterpreter --dns 1.1.1.1 -f Dockerfile_openinterpreter . 
IMAGE_NAME=openinterpreter
CONTAINER_NAME=${IMAGE_NAME}_container

docker kill $CONTAINER_NAME
docker rm $CONTAINER_NAME

docker rmi $IMAGE_NAME
docker build -t $IMAGE_NAME --network host -f Dockerfile_openinterpreter . 