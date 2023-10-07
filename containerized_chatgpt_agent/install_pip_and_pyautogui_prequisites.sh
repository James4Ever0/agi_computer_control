#!/bin/bash

function install_packages {
  # Install the packages
  apt install -y python3-pip python3-tk python3-dev xvfb xfce4 scrot tesseract-ocr

  # courtesy from ChatGPT
  local status=$?
  # Return the exit status
  return $status
}

install_packages
# Check the status of the package installation
if [ $? -ne 0 ]; then
# If the installation failed, run apt update and try again
echo "Package installation failed. Running apt update and trying again..."
apt update
install_packages
fi