EXPORT_FILEPATH=openinterpreter_arm64.tar

bash build_openinterpreter_container.sh
cd docker_images
rm $EXPORT_FILEPATH
docker save openinterpreter -o $EXPORT_FILEPATH
k3s ctr image import $EXPORT_FILEPATH