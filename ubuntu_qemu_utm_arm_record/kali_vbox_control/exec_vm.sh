# vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run --exe /usr/bin/env
# vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run --exe /bin/loginctl
vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run -- /bin/loginctl unlock-sessions

# Class=user
# Class=greeter