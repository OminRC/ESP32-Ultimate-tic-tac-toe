# Arduino IDE -- WiFi / phone browser version

Same idea as [`../../wifi/`](../../wifi) (ESP-IDF), ported to Arduino: the
board hosts its own WiFi network and a tiny web server, serving a single-page
UI straight from flash. No cable needed to play after flashing -- just
connect a phone to the board's WiFi and open a page.

Uses only `WiFi.h` and `WebServer.h`, both already bundled with the
**esp32 by Espressif Systems** board package -- no extra libraries to
install beyond what [`../ultimate_ttt/`](../ultimate_ttt)'s setup already
needs.

## Setup

1. Same board-package install as [`../README.md`](../README.md) (ESP32
   boards manager URL, select your board, 8MB flash, PSRAM enabled).
2. Open `ultimate_ttt_wifi.ino` (keep the whole `ultimate_ttt_wifi` folder
   together).
3. Upload (same wiring/IO0-EN dance as the Serial version if your module
   has no auto-reset circuit).
4. Open the Serial Monitor (115200 baud) once after upload to confirm it
   booted and see the exact IP to use:
   ```
   SoftAP up. Connect to WiFi "UltimateTTT-AI" (password "tictactoe"),
   then open http://192.168.4.1/
   ```

## Play

1. On your phone, connect to WiFi network **`UltimateTTT-AI`**, password
   **`tictactoe`** (change `WIFI_SSID`/`WIFI_PASS` at the top of the `.ino`
   before uploading if you want different ones).
2. Open `http://192.168.4.1/` in the phone's browser.
3. Tap a highlighted cell to play. You're `O`; the board plays `X` and
   replies automatically. "New game" resets.

## Files

Same layout/behavior as [`../ultimate_ttt/`](../ultimate_ttt), except the
I/O layer: `ultimate_ttt_wifi.ino` runs a `WebServer` instead of reading
`Serial` line-by-line, and [`index_html.h`](index_html.h) holds the web UI
(byte-for-byte copy of [`../../wifi/main/index_html.h`](../../wifi/main/index_html.h)).
`game.c`/`game.h`/`net_infer.c`/`net_infer.h`/`model_weights.h` are the
usual byte-for-byte copies shared across every variant in this repo.
