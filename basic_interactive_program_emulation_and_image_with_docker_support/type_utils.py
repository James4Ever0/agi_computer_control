def enforce_bytes(s):
    if isinstance(s, str):
        s = s.encode()
    if not isinstance(s, bytes):
        raise Exception("unknown line content type:", type(s))
    return s

def enforce_str(content):
    if isinstance(content, bytes):
        content = content.decode()
    if not isinstance(content, str):
        raise Exception("Invalid content type: %s\nShould be: str" % type(content))
    return content
