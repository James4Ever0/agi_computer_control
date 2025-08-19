IMAGE_NAME=cybergod_playwright_server

# shall make a prebuilt image with playwright node module globally installed instead of this
# docker run -p 3000:3000 --rm --init -it mcr.microsoft.com/playwright:v1.50.0-noble /bin/sh -c "npx -y playwright@1.50.0 run-server --port 3000 --host 0.0.0.0"

# run with cybergod built playwright server image
docker run -p 3000:3000 --rm --init -it --ipc=host $IMAGE_NAME /bin/sh -c "playwright run-server --port 3000 --host 0.0.0.0" 