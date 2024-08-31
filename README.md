# Who's Watching Who?


## Setup & Test

Install the wide angle camera drivers [link](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/#supported-platforms-and-os).

Camera model [link](https://www.amazon.com/dp/B0C53BBMLG?ref=ppx_yo2ov_dt_b_fed_asin_title#customerReviews)

Hide the mouse:

- `sudo apt-get install unclutter`

Clone the repo, then:

`python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera2` package.)
`source .venv/bin/activate`
`pip install -r requirements.txt`

`export DISPLAY=:0`
`WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-2 --transform 90`
`HDMI-A-1` for HDMI port 1

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

- [X] Test on MacBook
- [X] Test on Pi 4B with webcam (bad SD card)
- [X] Rotate screen on reboot
- [X] Hide mouse
- [X] Test on Pi 4B with Picam (bad SD card)
- [X] Install in portrait frame
- [ ] Speed up gaze
    [ ] hardware option: USB with OS (hardware option)
    [X] software option: only update eyes portion (hold in memory)


## Note

`jeff_1080-1920/16.png` is a copy of `jeff_1080-1920/15.png`...
The pictures in `jeff_1080-1920` are all 925 x 1645 pixels.

