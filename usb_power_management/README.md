it would be a joke if we use randomly self-detaching drives to run our precious projects.

this directory focus on how to get our storage drives and usb devices alive.

----

the primary cause of disk disconnection could be idle state. 

----

different OSes have different ways to keep devices alive. however, using NAS can be more robust and platform independent. but that only applies to storage devices.

although there have been some "netusb" or "serial over tcp" stuff, first we need to make sure the actual physical devices is online. blame me if these "netusb" related libraries handle device suspension.
