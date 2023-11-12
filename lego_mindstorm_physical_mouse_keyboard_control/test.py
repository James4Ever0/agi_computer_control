from usb_ev3 import *

ops = b"".join(
    (
        ev3.opSound,
        ev3.TONE,
        ev3.LCX(1),  # VOLUME
        ev3.LCX(440),  # FREQUENCY
        ev3.LCX(1000),  # DURATION
    )
)
my_ev3.send_direct_cmd(ops)


# voice = ev3.Voice(ev3_obj=my_ev3, volume=100)
# voice.speak("hello world").start(thread=False) # gtts failed to start.



mt1 = ev3.Motor(port=ev3.PORT_D, ev3_obj = my_ev3)
print("motor_type:", mt1.motor_type ) # 7 -> large motor
# mt1.start_move_by(10,speed = 100)
# mt1.start_move_by(10,speed = 1)
mt1.start_move_by(100,speed = 100)

# you know it could have problems with the environment
# when it will stop on bad things?