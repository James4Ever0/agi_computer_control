from __future__ import print_function

import hid
import time

# enumerate USB devices

# manufacturer_string -> "winkeyless.kr"
# shall be what we expect.

for d in hid.enumerate():
    keys = list(d.keys())
    keys.sort()
    for key in keys:
        print("%s : %s" % (key, d[key]))
    print()

devices = {
    0: {"vendor_id": 8352, "product_id": 16941, },  # 0, could be mouse.
    1: {"vendor_id": 8352, "product_id": 16941, },  # 1
}

# the same!

# try opening a device, then perform write and read
# may you write keyboard and mouse commands to different devices, by using `h.open(d)`
h = hid.device()
try:
    print("Opening the device")
    h.open(devices[0]["vendor_id"], devices[0]["product_id"])
    # h.open(0x534C, 0x0001)  # TREZOR VendorID/ProductID

    # print(dir(h))
    # 'get_feature_report', 'get_indexed_string', 'get_input_report', 'get_manufacturer_string', 'get_product_string', 'get_serial_number_string',
    # print(h.open_path)

    print("Manufacturer: %s" % h.get_manufacturer_string())
    print("Product: %s" % h.get_product_string())
    print("Serial No: %s" % h.get_serial_number_string())

    # enable non-blocking mode
    h.set_nonblocking(1)

    # write some data to the device
    # what data is this anyway?
    print("Write the data")
    h.write([0, 63, 35, 35] + [0] * 61)

    # wait
    time.sleep(0.05)

    # read back the answer
    print("Read the data")
    while True:
        d = h.read(64)
        if d:
            print(d)
        else:
            break

    print("Closing the device")
    h.close()

except IOError as ex:
    print(ex)
    print("hid error:")
    print(h.error())
    print("")
    print("You probably don't have the hard-coded device.")
    print("Update the h.open() line in this script with the one")
    print("from the enumeration list output above and try again.")

print("Done")
