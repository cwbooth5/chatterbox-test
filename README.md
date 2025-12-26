# chatterbox-test

This is an experiment.

This is a test to see if I can get chatterbox to run in ubuntu/WSL on CPU. It'd be much faster on the GPU but it's a bit too much
work to try it on AMD. This appears to work.

The idea is we have a small script that demonstrates what this does and you can try using that directly on the CLI.
The second way to use this is via the little web app, created to make the uploading of example audio and downloading
of the generated audio a little easier.

## Setup

You need a huggingface token to pull down the model weights initially (just once). It just needs to be a read-only token.
Once you do that, you never do anything off the box. Everything's local.

Set up a virtualenv around the code.

```
uv venv
source .venv/bin/activate
uv sync
```

The requirements are kinda finicky, FYI so you might require some tweaking of versions.

Run the web app. It's going to come up on localhost:8000

`uvicorn app:app --host 0.0.0.0 --port 8000`

weeeeeee
