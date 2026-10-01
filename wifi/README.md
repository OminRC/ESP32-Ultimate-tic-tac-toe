# Ultimate Tic-Tac-Toe AI -- WiFi / phone browser version

Same game engine and on-device AI as [`../main/`](../main), but instead of
the Serial monitor, the ESP32-S3 hosts its own WiFi network and a tiny web
server -- open a page on your phone and play with taps, no cable needed
after flashing.

## How it works

The board starts a **SoftAP** (its own WiFi network, no router/internet
required) and serves a single-page web UI from flash
([`index_html.h`](main/index_html.h)). The phone's browser talks to it over
three JSON endpoints:

- `GET /state` -- current board state
- `POST /move` -- `{"sub":n,"cell":n}`, applies your move then the AI's reply
- `POST /reset` -- starts a new game

All game logic and AI inference still run entirely on the ESP32-S3
(`game.c` / `net_infer.c`, byte-for-byte copies of `../main/`'s) -- WiFi is
just the input/output layer, same role the Serial monitor played before.

## Build and flash

```bash
cd wifi
idf.py set-target esp32s3
idf.py -p <PORT> build flash monitor
```

## Connect and play

1. On your phone, open WiFi settings and connect to:
   - **Network**: `UltimateTTT-AI`
   - **Password**: `tictactoe`
   (change these in `main/main.c` -- `WIFI_SSID` / `WIFI_PASS` -- before
   flashing if you want different ones; set `WIFI_PASS` to `""` for an open
   network)
2. Open a browser and go to **http://192.168.4.1/**
3. Tap a highlighted cell to play. You're `O`; the board plays `X` and
   replies automatically. Tap "New game" to reset.

## Notes

- Only one game/player is tracked at a time (single shared `Game` struct) --
  it's a single-player device, not a multiplayer server.
- `MAX_STA_CONN` in `main.c` allows a few WiFi clients to associate, but
  only whoever's browser last hit `/move` is really "playing" -- there's no
  per-client session. Fine for a demo/portfolio piece, not meant for
  simultaneous multi-user play.
- If the page doesn't load, double-check you're on the ESP32's WiFi (not
  still on your home WiFi/mobile data) -- the SoftAP has no internet uplink
  so phones sometimes auto-switch away from it.
