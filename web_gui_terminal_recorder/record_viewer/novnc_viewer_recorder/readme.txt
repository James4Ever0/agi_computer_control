novnc requires browser to work.
no headless version, requiring rework.

https://hub.docker.com/r/bonigarcia/novnc - light weight, based on alpine

https://hub.docker.com/r/gotget/novnc/

https://hub.docker.com/r/theasp/novnc - bundled with xterm

libvnc is a cpp library. there is libvnc-rs binding.

browser apps may run in phantomjs, or in xvfb with electron

alternatively, use playwright headless docker container instead of electron headless

https://hub.docker.com/r/mcp/playwright

https://hub.docker.com/r/microsoft/playwright

https://hub.docker.com/r/wernight/phantomjs

https://hub.docker.com/r/dannysu/electron-headless

https://hub.docker.com/r/bengreenier/docker-xvfb

https://hub.docker.com/r/linuxserver/xvfb

---

Xvfb (X virtual framebuffer) is a display server that performs graphical operations in memory without showing any screen output. This is particularly useful for running graphical applications in headless environments, such as CI/CD pipelines or Docker containers.

Setting Up Xvfb in Docker

To use Xvfb in Docker, you can create a Dockerfile that installs and configures Xvfb. Here are two examples of how to achieve this:

Example 1: Using bengreenier/docker-xvfb

    Create a Dockerfile:

FROM bengreenier/docker-xvfb:stable

# Install additional packages if needed
RUN apt-get update -y && \
apt-get install --no-install-recommends -y mesa-utils && \
rm -rf /var/lib/apt/lists/*

# Set the command to run your application
CMD glxgears

    Build and run the Docker image:

docker build . -t my-x11-app
docker run -it --rm --name my-running-app my-x11-app

Example 2: Using metal3d/docker-xvfb

    Start an Xvfb container:

docker run --name xvfb metal3d/xvfb

    Launch an application from another image:

docker run -e DISPLAY=xvfb:99 --link xvfb other/image app

Customizing Xvfb

Both images support customization through environment variables:

    Resolution: You can set the screen resolution using the RESOLUTION variable. The default is 1920x1080x24.

    Additional Arguments: You can pass additional arguments to Xvfb using the XARGS variable.

For example:
docker run -e RESOLUTION=1280x720x24 -e XARGS="-screen 0 1280x720x24" --name xvfb metal3d/xvfb

Conclusion

Using Xvfb with Docker allows you to run graphical applications in headless environments efficiently. By leveraging pre-built Docker images like bengreenier/docker-xvfb and metal3d/docker-xvfb, you can quickly set up and customize your Xvfb environment to suit your needs
1
2
.