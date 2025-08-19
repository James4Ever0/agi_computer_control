URL=http://127.0.0.1:6080/vnc.html

# https://playwright.dev/docs/docker

# use --ipc=host to avoid chromium oom

# in mcr.microsoft.com/playwright and mcp/playwright, they include xvfb-run for launching headful browsers.

# IMAGE_NAME=mcr.microsoft.com/playwright:v1.50.0-noble

# view help
# docker run -it --rm --entrypoint mcp/playwright --help

# browse files 
docker run -it --rm --entrypoint bash mcp/playwright

# use --network host to connect to the novnc page