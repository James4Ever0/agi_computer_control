from pywinauto_recorder.recorder import Recorder
# from pywinauto_recorder.player import UIPath, click, move, playback

# this seems not working.

from pywinauto_recorder.player import playback
recorder = Recorder()
input("START?")
recorder.start_recording()
import time
time.sleep(10)
print("RECORD COMPLETE")
input("PLAY?")
#  |          with UIPath("Untitled - Notepad||Window"):
#  |                  doc = move("Text editor||Document")
#  |                  time.sleep(0.5)
#  |                  click(doc)
#  |                  utf8 = move("||Pane-> UTF-8||Text")
#  |                  time.sleep(0.5)
#  |                  click(utf8)
recorded_python_script = recorder.stop_recording()
recorder.quit()
print("RECORDING FILENAME?", recorded_python_script)
playback(filename=recorded_python_script)