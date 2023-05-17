# vboxmanage controlvm "Ubuntu 16.04" keyboardputscancode
# vboxmanage controlvm "Ubuntu 16.04" keyboardputstring

# cannot perform mouse actions.

# can only do this when vm running.

vboxmanage controlvm "Ubuntu 16.04" vrde on
vboxmanage controlvm "Ubuntu 16.04" vrdeport 8991
vboxmanage controlvm "Ubuntu 16.04" vrdeproperty vrdeauthtype=null

vboxmanage controlvm "Ubuntu 16.04" audioout off
# vboxmanage guestcontrol "Ubuntu 16.04" run -- 
