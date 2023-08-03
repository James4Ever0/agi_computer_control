import sys
sys.append("../")

from conscious_struct import HIDActionTypes
import json

with open("keys.json", 'r') as f:
    content = f.read()
    kcom_keycodes = json.loads(content)

kcom_translation_table = {}



for record in kcom_keycodes:
    keyname = record['Key Name']
    keycode = bytes.fromhex(record['HID Usage ID'])

def KeyLiteralToKCOMKeycode(keyLiteral: HIDActionTypes.keys):
    ...
    