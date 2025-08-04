echo "Adding user $USER to docker group..."
sudo addgroup $USER docker
echo "Remember to logout and login for the changes to take effect."
echo "You can also run 'newgrp docker' to apply the changes to the current session."