import os 

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = api_base

import litellm

# it is bad to run random commands.
# maybe you should listen to the advice at https://github.com/Significant-Gravitas/AutoGPT/issues/346
# before it is too late.

# we are just prototyping. why so serious.
# trying random stuff!

# let's create a virtual editor.

# you just need to master the diff, the memory and the action

prompt = """
You are an AI agent inside a terminal environment. You can interact with the environment by writing special commands separated by newline. After your actions, the environment will execute the command and return the current terminal view.

Avaliable commands:
type <character sequence>
return
delete

Terminal view:


"""