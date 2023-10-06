from __future__ import unicode_literals
import ptyprocess

# this module is exclusive for windows. to port to linux there should be extra steps.
# i mean, android.
# hey! do not run this shit outside of sandbox, unless you want to get me killed.
import threading
import pyte

# can you format things into colorful output?
# or just raw terminal string which can be transformed into html.
import traceback
import tornado.ioloop
import tornado.web
import requests
import base64
import signal

# no watchdog for this?
LF_CRLF = b"\n"
maxbark = 2
maxbark_granual = 5
maxterm = 3
maxterm_granual = 5
bark = 0
term = 0
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=8788, help="port number")
args = parser.parse_args()
port = args.port
assert port > 0 and port < 65535
print("server running on port %d" % port)
# you can turn off the barking dog sometimes.
# we can use a big dog every since then.
def kill(pipe):
    try:
        pipe.terminate()
        # here.
        pipe.kill(signal.SIGKILL)
    except:
        print("_____process kill error_____")
        traceback.print_exc()


# signal.signal(signal.SIGINT, signal_handler)
display = ""
lag = 0.05
executable = "bash"  # this is wrong. could get your computer in danger.
# unless you want to take the risk. everything worth the try?
cols, rows = 80, 25
import time

watch_rate = 0.5
screen = pyte.Screen(cols, rows)
stream = pyte.ByteStream(screen)
process = ptyprocess.PtyProcess.spawn([executable], dimensions=(rows, cols))


def read_to_term():
    global display, stream, screen
    # read a global list?
    # you can start another server. not quite like terminal. like execution shell.
    noerr = True
    while noerr:
        try:
            reading = process.read()
            # will block.
            # will raise error if not good.
            stream.feed(reading)
            display = "\n".join(screen.display)
        except:
            noerr = False
            break


t0 = threading.Thread(target=read_to_term, args=())
t0.setDaemon(True)
t0.start()


def barkdog():
    global bark, maxbark_granual
    while True:
        bark = 0
        time.sleep(maxbark_granual)


tb = threading.Thread(target=barkdog, args=())
tb.setDaemon(True)
tb.start()


def termdog():
    global term, maxterm_granual
    while True:
        term = 0
        time.sleep(maxterm_granual)


tx = threading.Thread(target=termdog, args=())
tx.setDaemon(True)
tx.start()


def watchdog():
    global process, watch_rate, port, bark, maxbark
    alive = True
    while alive:
        alive = process.isalive()
        #        print("alive?",alive)
        time.sleep(watch_rate)
    #    print("bark")
    bark += 1
    if bark > maxbark:
        print("max bark exceed.", bark)
        # what the heck?
        pass
    else:
        #        print("did get to here")
        # if server is down this will cause dead shit.
        requests.get(
            "http://localhost:{}/restart".format(port),
            stream=False,
            verify=False,
            timeout=1,
        )


# does that work?
# if not, call the handler. use requests.
t1 = threading.Thread(target=watchdog, args=())
t1.setDaemon(True)
t1.start()


class RHandler(tornado.web.RequestHandler):
    def get(self):
        global process, screen, stream, t0, t1, executable, display, term, maxterm
        # print(type(process))
        # print(dir(process))
        term += 1
        if term > maxterm:
            self.write("exceeding max termination quota!\n")
        else:
            kill(process)
            # did it stuck here?
            # nope.
            for x in [process, screen, stream, t0, t1]:
                # print("deleting")
                del x
            display = ""
            screen = pyte.Screen(cols, rows)
            stream = pyte.ByteStream(screen)
            process = ptyprocess.PtyProcess.spawn([executable], dimensions=(rows, cols))
            t0 = threading.Thread(target=read_to_term, args=())
            t0.setDaemon(True)
            t0.start()
            t1 = threading.Thread(target=watchdog, args=())
            t1.setDaemon(True)
            t1.start()
            self.write("terminal restart!\n")


class IHandler(tornado.web.RequestHandler):
    def get(self):
        global display, process, lag
        # print("type request received.")
        argument = self.get_argument("type", None)
        argumentx = self.get_argument("b64type", None)
        # that is for argument!
        autoreturn = self.get_argument("autoreturn", None) == "true"
        # print("actual argument",[argument],type(argument))
        # string.
        if not process.isalive():
            self.write("process is dead.\n")
        elif argument is not None:
            # unicode.
            # may encounter error.
            if autoreturn:
                process.write(argument.encode("utf8") + b"\r")
            else:
                process.write(argument.encode("utf8"))
            time.sleep(lag)
            self.write(display)
        elif argumentx is not None:
            # check if correctly formed.
            # check if not dead.
            try:
                arx = base64.b64decode(argumentx)
                # the result is not right.
                # cannot decode here.
                if autoreturn:
                    process.write(arx + b"\r")
                else:
                    process.write(arx)
                    # this is not unicode string.
                time.sleep(lag)
                self.write(display)
            except:
                self.write("incorrect format\n")
                # pass
                # D:\Programs\Python\Python36\lib\site-packages\winpty\winpty_wrapper.py
        else:
            self.write("empty input\n")
            # pass


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        global display
        self.write(display)

    def make_app():
        return tornado.web.Application(
            [(r"/display", MainHandler), (r"/restart", RHandler), (r"/input", IHandler)]
        )


# get a window watcher. if want to lock the winsize better use that.
# why the fuck that the code needs to be compiled? could we just examine the code and prepare for tested binaries?
app = MainHandler.make_app()
app.listen(port)
# here's the shit.
tornado.ioloop.IOLoop.current().start()
# register handler.
exit()
