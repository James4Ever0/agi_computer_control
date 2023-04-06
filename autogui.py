import pyautogui, time, sys, os, win32api, win32gui, win32con, datetime, pyHook, pythoncom
from optparse import OptionParser

'''
Python Automated Actions Script by Ian Mckay
Version 0.1 - 20151217
'''

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True
main_thread_id = win32api.GetCurrentThreadId()
events = []
recording = False

def OnMouseEvent(event):
	global events
	global recording
	
	if (event.Message!=512): # 512 is mouse move
		'''
		print('MessageName:',event.MessageName)
		print('Message:',event.Message)
		print('Time:',event.Time)
		print('Window:',event.Window)
		print('WindowName:',event.WindowName)
		print('Position:',event.Position)
		print('Wheel:',event.Wheel)
		print('Injected:',event.Injected)
		print('---')
		'''
	
		if (recording==True):
			events.append([event.Position[1],event.Position[0],event.Message,event.Time,"2"])
	
	return True

def OnKeyboardEvent(event):
	global hm
	global events
	global recording
	global starttime
	global main_thread_id
	'''
	print('MessageName:',event.MessageName)
	print('Message:',event.Message)
	print('Time:',event.Time)
	print('Window:',event.Window)
	print('WindowName:',event.WindowName)
	print('Ascii:', event.Ascii, chr(event.Ascii))
	print('Key:', event.Key)
	print('KeyID:', event.KeyID)
	print('ScanCode:', event.ScanCode)
	print('Extended:', event.Extended)
	print('Injected:', event.Injected)
	print('Alt', event.Alt)
	print('Transition', event.Transition)
	print('---')
	'''
	
	if (recording==True):
		if (event.Key=="End"):
			hm.UnhookKeyboard()
			hm.UnhookMouse()
			win32api.PostThreadMessage(main_thread_id, win32con.WM_QUIT, 0, 0);
			events.append(["0","0","1",event.Time,"0"])
			print("Ended recording")
			print('\a')
			return False
		else:
			events.append([event.Extended,event.KeyID,event.Message,event.Time,"1"])
		
	if (recording==False):
		if (event.Key=="Home" and event.Message==257):
			starttime = datetime.datetime.now().time()
			recording = True
			print("Started recording")
			print('\a')
			events.append(["0","0","0",event.Time,"0"])
			return False
	
	return True

def record():
	global hm

	print("Hooking now...")
	hm = pyHook.HookManager()
	hm.MouseAll = OnMouseEvent
	hm.KeyAll = OnKeyboardEvent
	hm.HookMouse()
	hm.HookKeyboard()
	print("Hooked")
	pythoncom.PumpMessages()
	print("Exporting...")
	f = open('recording.txt', 'w+')
	for event in events:
		f.write(str(event.pop()) + ',' + str(event.pop()) + ',' + str(event.pop()) + ',' + str(event.pop()) + ',' + str(event.pop()) + '\n')
	f.close()
	print("Ending...")

def play():
	print("Starting in 2 secs...")
	time.sleep(2)

	lasttime=False

	with open('recording.txt') as fp:
		for line in fp:
			elements = line.split(',')
			eventdata2 = int(elements.pop().replace('\n',''))
			eventdata1 = int(elements.pop())
			eventsubtype = int(elements.pop())
			eventtime = int(elements.pop())
			eventtype = int(elements.pop())
			
			if (lasttime==False):
				if (eventtype==0 and eventsubtype==0):
					lasttime=eventtime
				else:
					print("Source data error! (eventtype=" + str(eventtype) + ", eventsubtype=" + str(eventsubtype) + ")")
					sys.exit(1)
			elif (eventtype==1):
				time.sleep(max((eventtime-lasttime)/1000,0.02)) # At least 20ms between everything
				
				if (eventdata1>32 and eventdata1<127 and eventdata2==0):
					key = chr(eventdata1).lower()
				elif (eventdata1==91 and eventdata2==1):
					key = "winleft"
				elif (eventdata1==9 and eventdata2==0):
					key = "tab"
				elif (eventdata1==20 and eventdata2==0):
					key = "capslock"
				elif (eventdata1==160 and eventdata2==0):
					key = "shiftleft"
				elif (eventdata1==162 and eventdata2==0):
					key = "ctrlleft"
				elif (eventdata1==164 and eventdata2==0):
					key = "altleft"
				elif (eventdata1==32 and eventdata2==0):
					key = "space"
				elif (eventdata1==165 and eventdata2==1):
					key = "altright"
				elif (eventdata1==163 and eventdata2==1):
					key = "ctrlright"
				elif (eventdata1==37 and eventdata2==1):
					key = "left"
				elif (eventdata1==40 and eventdata2==1):
					key = "down"
				elif (eventdata1==39 and eventdata2==1):
					key = "right"
				elif (eventdata1==161 and eventdata2==1):
					key = "shiftright"
				elif (eventdata1==38 and eventdata2==1):
					key = "up"
				elif (eventdata1==34 and eventdata2==1):
					key = "pgdn"
				elif (eventdata1==33 and eventdata2==1):
					key = "pgup"
				elif (eventdata1==8 and eventdata2==0):
					key = "backspace"
				elif (eventdata1==44 and eventdata2==1):
					key = "printscreen"
				elif (eventdata1==46 and eventdata2==1):
					key = "delete"
				elif (eventdata1==27 and eventdata2==0):
					key = "esc"
				elif (eventdata1==13 and eventdata2==0):
					key = "enter"
				elif (eventdata1==112 and eventdata2==0):
					key = "f1"
				elif (eventdata1==113 and eventdata2==0):
					key = "f2"
				elif (eventdata1==114 and eventdata2==0):
					key = "f3"
				elif (eventdata1==115 and eventdata2==0):
					key = "f4"
				elif (eventdata1==116 and eventdata2==0):
					key = "f5"
				elif (eventdata1==117 and eventdata2==0):
					key = "f6"
				elif (eventdata1==118 and eventdata2==0):
					key = "f7"
				elif (eventdata1==119 and eventdata2==0):
					key = "f8"
				elif (eventdata1==120 and eventdata2==0):
					key = "f9"
				elif (eventdata1==121 and eventdata2==0):
					key = "f10"
				elif (eventdata1==122 and eventdata2==0):
					key = "f11"
				elif (eventdata1==123 and eventdata2==0):
					key = "f12"
				else:
					print("Skipping unknown keycode: " + str(eventdata1))
					key = False
				
				if (eventsubtype==256 or eventsubtype==260): # I think 260 is a "virtual keystroke"
					if (key!=False):
						pyautogui.keyDown(key)
				elif (eventsubtype==257):
					if (key!=False):
						pyautogui.keyUp(key)
				else:
					print("Bad keyboard subtype!")
					sys.exit(1)
				
				lasttime=eventtime
			elif (eventtype==2):
				time.sleep(max((eventtime-lasttime)/1000,0.02)) # At least 20ms between everything
				
				if (eventsubtype==513):
					pyautogui.mouseDown(x=eventdata1, y=eventdata2, button='left')
				elif (eventsubtype==514):
					pyautogui.mouseUp(x=eventdata1, y=eventdata2, button='left')
				else:
					print("Bad mouse subtype!")
					sys.exit(1)
				
				lasttime=eventtime
			elif (eventtype==0 and eventsubtype==1):
				print("Done.")
				sys.exit(0)
			else:
				print("Bad source major type!")
				sys.exit(1)
	print("Done playing")
	print('\a')

def main():
	usage = "usage: %prog [options]"
	parser = OptionParser(usage=usage)
	parser.add_option("-r", "--record", action="store_true", dest="do_record", default=False, help="record a session of input")
	parser.add_option("-p", "--play", action="store_true", dest="do_play", default=False, help="play a session of input")
	(options, args) = parser.parse_args()
	
	if (options.do_record==False and options.do_play==False):
		parser.print_help()
		sys.exit(0)
	if (options.do_record==True):
		record()
	if (options.do_play==True):
		play()

if __name__ == "__main__":
	main()

'''
PLAY AREA
w=win32gui
title=w.GetWindowText(w.GetForegroundWindow())
im = None
def capture():
	global im
	#pyautogui.click(1200, 500)
	#pyautogui.typewrite('About to close window!')
	#time.sleep(2)
	#pyautogui.hotkey('alt', 'f4')
	#pyautogui.screenshot()
	posx, posy = pyautogui.position()
	print(str(posx) + "," + str(posy))
	im = pyautogui.screenshot('img.png',region=(posx-20,posy-20,40,40))
def replay():
	global im
	loc = pyautogui.locateOnScreen(im)
	locx, locy = pyautogui.center(loc)
	pyautogui.click(locx, locy)
#capture()
#replay()
def post_keys(hwnd, i):
	win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, i, 0)
	win32api.SendMessage(hwnd, win32con.WM_KEYUP, i, 0)
'''