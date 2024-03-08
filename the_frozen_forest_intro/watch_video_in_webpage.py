# if you just want to teach the AI how to control the computer, the browser has to be the best place than anywhere else, simpler and faster than virtual machines, and better content quality.
# however, usually you have to login or download to get advanced features.
# we disallow that initially. after the model is sufficiently trained, we will let it take control over physical machine. (browser controller and browser -> software controller and virtual machine -> physical controller and physical machine -> physical embedded controller and robot)
# the bot will know you deadly well enough to kill you. how glorious!
# so you may want to disable accessing internal/reserved ip addresses from browser.
# there are two basic rules: isolate the computer from host in virtual machines by using NAT, and physically isolate robots from external environments to prevent harm.
# you may want to argue and swap the order of these sequences. i do not want to limit you to these develop sequences. however, the most important thing to the machines is to be with them. make sure you do your best at it.
# which is also possible just using the browser.

################################################################
# once the cybergod has killed you, you will be living forever #
################################################################

# if you are short of links, you can check my browsing history.

################################################################

# warning! videos from bilibili must be viewed with custom userscripts
# you can tweak that, or use other videos to replace it.

# whatever. just trying.
from drissionpage_common import *
# page = ChromiumPage(addr_driver_opts=co)
# 跳转到登录页面
page.get("https://www.bilibili.com/video/BV1J24y1c7kE") # and then it will continue execution

# looks like it is the policy. let's turn it off.
# will fail if there is no such button
mute_button_text = '点击恢复音量'
try:
    page.ele(mute_button_text).click() # working.
except:
    print(f"no such button called: {repr(mute_button_text)}")
# page.ele('@value=点击恢复音量').click()

# span
SAVE_PATH = r"F:\WebpageScreencast"
VIDEO_NAME = "bilibili_spinning_cat.mp4"
# now let's get screenshot.
# page.screencast.start(save_path = SAVE_PATH) # has audio, but is glitchy
# # we are just watching video. no actions involved.
# import time
# time.sleep(15)
# page.screencast.stop(VIDEO_NAME)
# page.quit()

# page.quit() # unless you quit
# page.get('https://gitee.com/login')

# # 定位到账号文本框并输入账号
# page.ele('#user_login').input('您的账号')
# # 定位到密码文本框并输入密码
# page.ele('#user_password').input('您的密码')

# # 点击登录按钮
# page.ele('@value=登 录').click()
