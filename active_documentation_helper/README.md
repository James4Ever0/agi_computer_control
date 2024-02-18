render bytes from shell, into pyte, and then use agent ai to operate in the environment, observe it and execute commands.

some small objectives like using vim, run shell commands, navigate and so on must be completed successfully before going to auto exploration.

the exploration shall be limited by time.

---

godlang is just a name. we choose it because this is cybergod we are working with.

---

duckyscript is keyboard only, and introduces a lot of new syntax

```
STRING STRINGLN END_STRING END_STRINGLN HOLD RELEASE RESET
```

complex control flow used in duckyscript, may be abandoned since we are handcrafters.

introducing: WITH statement (python like indentation, automatically handle HOLD and RELEASE, STRING and END_STRING etc)

introducing: INSPECT statement, showing keyboard keys being held, mouse location and onhold mouse keys

---

mouse controls: (can set jitter interval between command execution, random)

```
MOVETO x y
RELMOVE dx dy
CLICK mouse_button
HOLD mouse_button
RELEASE mouse_button
SCROLL dx dy
```

---

touchpad controls:

```
FINGER[finger_index] MOVETO x y
FINGER[finger_index] RELMOVE dx dy
FINGER[finger_index] TAP
FINGER[finger_index] HOLD
FINGER[finger_index] RELEASE
LIST_FINGERS
CLEAR_FINGERS
```

---

stylus controls: (jitter of the stylus can be different from the mouse, or shared with mouse)

```
TRACK
    x y force
    x y force
    x y force
    x y force
END_TRACK
CLICK stylus_button
HOLD stylus_button
RELEASE stylus_button
```