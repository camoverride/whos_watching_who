# Who's Watching Who?

# Setup

Rotate the screen:
TODO: make this actually work on boot!
`export DISPLAY=:0`
`WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90`

Clone the repo, then:

`python -m venv .venv`
`source venv/bin/activate`
`pip install -r requirements.txt`


## Run

`python display_screen.py`


## TODO

- [X] Tested on MacBook
- [ ] Tested on Pi 4B with webcam (bad SD card)
- [ ] Tested on Pi 4B with Picam (bad SD card)
- [ ] Tested on Pi 4B with Picam (good SD card)
- [ ] Install in portrait frame


## Note

`jeff_1080-1920/16.png` is a copy of `jeff_1080-1920/15.png`...
The pictures in `jeff_1080-1920` are all 925 x 1645 pixels.

