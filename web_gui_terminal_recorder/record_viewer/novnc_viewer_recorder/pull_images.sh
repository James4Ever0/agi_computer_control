docker pull bonigarcia/novnc:1.2.0 # no latest tag

# this image has fully installed browser binaries, but without any nodejs or python dependencies.
# docker pull mcr.microsoft.com/playwright:v1.50.0-noble 

# the mcp version of playwright container only has chromium binary installed
# we can further communicate it with python; node_modules (with playwright library) location: /app/node_modules; mcp cli location: /app/cli.js
docker pull mcp/playwright