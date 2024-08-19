# Who's Watching Who?


## Setup & Test

Hide the mouse:

- `sudo apt-get install unclutter`
- `unclutter -idle 0`

Clone the repo, then:

`python -m venv .venv`
`source venv/bin/activate`
`pip install -r requirements.txt`

`export DISPLAY=:0`
`WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90`

`python display_screen.py`


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies:

- `mkdir -p ~/.config/systemd/user`

- Paste the contents of `display.service` into `~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u display.service`


## TODO

- [X] Tested on MacBook
- [X] Tested on Pi 4B with webcam (bad SD card)
- [X] Rotate screen on reboot
- [ ] Tested on Pi 4B with Picam (bad SD card)
- [ ] Install in portrait frame


## Note

`jeff_1080-1920/16.png` is a copy of `jeff_1080-1920/15.png`...
The pictures in `jeff_1080-1920` are all 925 x 1645 pixels.

