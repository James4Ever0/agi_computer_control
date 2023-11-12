import ev3_dc as ev3

protocol_param = dict(protocol=ev3.USB)
my_ev3 = ev3.EV3(**protocol_param) # working!
my_ev3.verbosity = 1
