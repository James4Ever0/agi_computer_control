import torch
import typing
import functools

# import zeta.nn.attention
# no you must change the code.

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
except:
    # let's patch the error.
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper

    def lambda_auto_wrap_policy(
        module: torch.nn.Module,
        recurse: bool,
        unwrapped_params: int,
        lambda_fn: typing.Callable,
    ) -> bool:
        """
        A convenient auto wrap policy to wrap submodules based on an arbitrary user
        function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
        a `wrapper_cls` unit.

        Return if a module should be wrapped during auto wrapping.

        The first three parameters are required by :func:`_recursive_wrap`.

        Args:
        module (nn.Module):
            The module to be considered in this decision.
        recurse (bool):
            Indicate if this is called to make a decision on whether we
            should recurse down a subgraph of the module structure.
            If False, it means this function is called to make a decision
            on whether we should wrap the said module.
        unwrapped_params (int):
            The number of parameters yet to be wrapped in this module.

        lambda_fn (Callable[nn.Module] -> bool):
            If this returns ``True``, this module will be wrapped by
            wrapper_cls individually.
        """
        if recurse:
            # always recurse
            return True
        else:
            # if not recursing, decide whether we should wrap for the leaf node or reminder
            return lambda_fn(module)

    def apply_activation_checkpointing_wrapper(
        model,
        checkpoint_wrapper_fn=torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper,
        check_fn=lambda _: True,
    ):
        """
        Applies :func:`checkpoint_wrapper` to modules within `model` based on a user-defined
        configuration. For each module within `model`, the `check_fn` is used to decide
        whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

        Note::
            This function modifies `model` in place and replaces appropriate layers with
            their checkpoint-wrapped modules.
        Note::
            This function will not wrap the overall root module. If this is needed, please directly use
            :class:`CheckpointWrapper`.
        Usage::
            model = nn.Sequential(
                nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
            )
            check_fn = lambda l: isinstance(l, nn.Linear)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        Args:
            module (nn.Module):
                The model who's submodules (or self) should be wrapped with activation checkpointing.
            checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
                A `Callable` which will wrap modules
            check_fn (Optional[Callable[nn.Module, nn.Module]])
                A lambda function which will be passed current layer and returns
                ``True`` or ``False`` depending on whether input layer should be wrapped.
        Returns: None (`model` is modified inplace)
        """
        # TODO: Importing inside function to avoid circular import issue between FSDP and
        # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
        from torch.distributed.fsdp.wrap import _recursive_wrap

        return _recursive_wrap(
            module=model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy, lambda_fn=check_fn
            ),
            wrapper_cls=checkpoint_wrapper_fn,
            ignored_modules=set(),
            ignored_params=set(),
            only_wrap_children=True,
        )

    setattr(
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper,
        "apply_activation_checkpointing",
        apply_activation_checkpointing_wrapper,
    )
from rtx import RTX2

major_torch_version = int(torch.__version__.split(".")[0])
flash_attention = False
if major_torch_version >= 2:  # use flash attention
    print("Using flash attention")
    flash_attention = True
# when posting ad via email. be sure there is an unsubscribe button to avoid legal issues.

dev = None
device_name = "CPU"
if torch.cuda.is_available():
    print("Trying to use first CUDA device.")
    # dev = torch.cuda.device(0)  # not working on torch 1.11
    dev = "cuda"
    device_name = "CUDA"
else:
    print("`torch.cuda` is not available.")
    try:
        print("Trying DirectML.")
        import torch_directml

        dev = torch_directml.device()
        device_name = "DirectML"
    except:
        print("Could not find DirectML device.")
print(f"Using {device_name}.")


def forward_new(self, img: torch.Tensor, text: torch.Tensor):
    """Forward pass of the model."""
    try:
        _encoded = self.encoder(img, return_embeddings=True)
        print("encoded context shape: {}".format(_encoded.shape))
        # torch.Size([2, 64, 512])
        # b, wtf, 2*dim
        encoded = _encoded
        # the shape is fixed. damn.
        # now we either need addition or some thin nn
        # encoded = torch.zeros((2,128,512)).to(dev)
        # encoded = torch.zeros((2,64,1024)).to(dev)
        # can we use arbitrary input? can we?
        return self.decoder(text, context=encoded)
    except Exception as error:
        print(f"Failed in forward method: {error}")
        raise


RTX2.forward = forward_new
# uninstalling and reinstalling 'timm' 'zetascale' and 'beartype' helps.
# is it data corruption?

# windows is not supported
# 'NoneType' object has no attribute 'cadam32bit_grad_fp32'

batch_size = 1
# batch_size = 2
input_length = 1024  # output_length = input_length - 1

# usage
# it is trying to expand the observation space to infinity, till the model says it is end.
img = torch.randn(
    batch_size, 3, 256, 256
)  # the size of the image is not the same as vit.
text = torch.randint(
    0, 20000, (batch_size, input_length)
)  # one of 20000 logits, 1024 as count

# want classification? you can either use more tokens or use special classifiers.
# for me, just use more tokens. we will train this model in multiple "continuous" scenarios anyway.
# also benefit from llms
# adjust the resolution according to the environment

# what is similar transformer in audio files like the ViT?
# how do we put audio in?

# approach 1: audio -> mel graph -> ViT -> embedding (ref: https://github.com/YuanGongND/ast)
# approach 2: native audio transformers (ref: https://github.com/lucidrains/audiolm-pytorch)

# how to merge?
# approach 1: linear addition
# approach 2: concatenation
# approach 3: concatenation with thin linear nns

# visual: add global & local area for the robot to inspect
# hierarchical cascade structure? like small -> medium -> large

# THE MOST IMPORTANT THING IS TO FIGURE OUT HOW TO CONNECT THE LM WITH VIT

model = RTX2(attn_flash=flash_attention)

# let's use the gpu.
# breakpoint()
if dev is not None:
    model.to(dev)
    output1, output2 = model(img.to(dev), text.to(dev))
else:
    output1, output2 = model(img, text)
# output1: torch.Size([1, 1023, 20000]) (logits)
# output2 is a single number (loss? how come?)
# this is the same as text input. is it?
# or just trying to reduce the loss against input text?
print("output logits:", output1.shape, output2.shape)  # with gradient!
print("device:", output1.device)
# breakpoint()
# output logits: torch.Size([2, 1023, 20000]) torch.Size([])
# device: privateuseone:0

# possible implementations:

# i decide to go with the latter.

# 1. interpret the robot action sequentially in a loop, if separated by any other token the interpretation ends, and it will start over from the beginning
# 2. use indicator/classifier to determine what type of action the robot is taking for every token (token classification)

# not working for DirectML
# memory_usage = torch.cuda.memory_allocated(device=dev) / 1024**3  # in GB

# print("Memory Usage:", memory_usage, "GB")

############ CPU Usage ############

import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024**3  # in GB

print("Memory Usage:", memory_usage, "GB")

# ref: https://github.com/microsoft/DirectML/issues/444