# Who's Watching Who?


## Setup & Test

Hide the mouse [stack overflow post](https://raspberrypi.stackexchange.com/questions/145382/remove-hide-mouse-cursor-when-idle-on-rasbperry-pi-os-bookworm):

- `sudo apt install -y interception-tools interception-tools-compat`
- `sudo apt install -y cmake`
- `cd ~`
- `git clone https://gitlab.com/interception/linux/plugins/hideaway.git`
- `cd hideaway`
- `cmake -B build -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build`
- `sudo cp /home/$USER/hideaway/build/hideaway /usr/bin`
- `sudo chmod +x /usr/bin/hideaway`
- `cd ~`
- `wget https://raw.githubusercontent.com/ugotapi/wayland-pagepi/main/config.yaml`
- `sudo cp /home/$USER/config.yaml /etc/interception/udevmon.d/config.yaml`
- `sudo systemctl restart udevmon`

Open your `~/.config/wayfire.init` file and adjust the following:

```
[core]
plugins = \
        autostart \
        ### We need to add the extra plugin to the list ###
        hide-cursor
```

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

