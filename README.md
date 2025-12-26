# chatterbox-test

This is an experiment. It's a voice cloning tool that takes a short input clip, some text you write, and then it generates an output .wav file using the reference voice. It's built on [chatterbox-tts](https://github.com/resemble-ai/chatterbox).

There are a couple models this thing can use:

* `ChatterboxTTS`
* `ChatterboxTurboTTS`

The non-turbo model appears to work fine with a tiny patch to torch, as demonstrated in the example scripts over in the chatterbox-tts repo. What I'm doing here is getting the turbo model to work too. To get it to run, it involves more complicated dependency patching. I did whatever I could to get it to work. It's not quite as efficient as it could be probably, but it does work.

Supported and tested setups:

- Mac M2+ (using mps and cpu)
- Windows 11/WSL2 with Ubuntu 22.04 (using cpu)

Untested:

- anything running directly on Nvidia cards (using cuda)

The idea is we have a small script that demonstrates what this does and you can try using that directly on the CLI.
The second way to use this is via the little web app, created to make the uploading of example audio and downloading
of the generated audio a little easier.

## Setup

### Download Weights

You need a huggingface token to pull down the model weights initially (just once). It just needs to be a read-only token.
Once you do that, you never do anything off the box. Everything's local. You have a couple choices for using that token.

1. `hf auth login` (if you have this installed)
2. `export HF_TOKEN=YOUR_TOKEN_HERE python ./main.py`

The main.py script just runs a quick clone of the voice sample checked into this repo. The first time that runs, it should download all the weights.
Once you have those locally, this demo doesn't need the Internet and it doesn't make any external connections.

### Python virtualenv setup

Set up a virtualenv around the code. The requirements are kinda finicky, FYI so you might require some tweaking of versions.

```
uv venv
source .venv/bin/activate
uv sync
```

To test that the script works on your setup, just run main.py. Otherwise, you can start the web app for a simpler experience.

Run the web app. It's going to come up on localhost:8000

`uvicorn app:app --host 0.0.0.0 --port 8000`

weeeeeee
