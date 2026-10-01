#include <WiFi.h>
#include <WebServer.h>
#include "game.h"
#include "net_infer.h"
#include "index_html.h"

// Phone connects directly to this network -- no router/internet needed.
// WPA2 requires >=8 chars; set WIFI_PASS to "" for an open network instead.
#define WIFI_SSID "UltimateTTT-AI"
#define WIFI_PASS "tictactoe"

static WebServer server(80);
static Game g_game; // single shared game -- this device serves one player at a time

static void ai_move_if_needed() {
    while (!g_game.done && g_game.player == 1) {
        float state[126];
        float policy[81];
        game_encode(&g_game, state);
        net_infer(state, policy);

        float mask[81];
        game_legal_mask81(&g_game, mask);

        int best_a = -1;
        float best_p = -1.0f;
        for (int a = 0; a < 81; a++) {
            if (mask[a] > 0 && policy[a] > best_p) {
                best_p = policy[a];
                best_a = a;
            }
        }
        if (best_a < 0) break;
        game_play(&g_game, best_a / 9, best_a % 9);
    }
}

static int state_to_json(char *buf, size_t buflen, bool error) {
    int n = 0;
    n += snprintf(buf + n, buflen - n, "{\"board\":[");
    for (int s = 0; s < 9; s++) {
        for (int c = 0; c < 9; c++) {
            n += snprintf(buf + n, buflen - n, "%d%s", g_game.board[s][c],
                          (s == 8 && c == 8) ? "" : ",");
        }
    }
    n += snprintf(buf + n, buflen - n, "],\"subResult\":[");
    for (int s = 0; s < 9; s++) {
        n += snprintf(buf + n, buflen - n, "%d%s", g_game.sub_result[s], s == 8 ? "" : ",");
    }
    n += snprintf(buf + n, buflen - n,
                  "],\"activeSub\":%d,\"player\":%d,\"winner\":%d,"
                  "\"done\":%s,\"error\":%s}",
                  g_game.active_sub, g_game.player, g_game.winner,
                  g_game.done ? "true" : "false", error ? "true" : "false");
    return n;
}

static void handleIndex() {
    server.send(200, "text/html", INDEX_HTML);
}

static void handleState() {
    char buf[1600];
    state_to_json(buf, sizeof(buf), false);
    server.send(200, "application/json", buf);
}

static void handleMove() {
    String body = server.arg("plain"); // raw POST body
    int sub = -1, cell = -1;
    sscanf(body.c_str(), "{\"sub\":%d,\"cell\":%d}", &sub, &cell);

    bool error = false;
    if (g_game.done || g_game.player != -1 || sub < 0 || sub > 8 ||
        cell < 0 || cell > 8 || !game_is_legal_move(&g_game, sub, cell)) {
        error = true;
    } else {
        game_play(&g_game, sub, cell); // human (O) move
        ai_move_if_needed();            // board (X) replies immediately
    }

    char buf[1600];
    state_to_json(buf, sizeof(buf), error);
    server.send(200, "application/json", buf);
}

static void handleReset() {
    game_init(&g_game);
    ai_move_if_needed(); // X always moves first
    char buf[1600];
    state_to_json(buf, sizeof(buf), false);
    server.send(200, "application/json", buf);
}

void setup() {
    Serial.begin(115200);
    delay(300);

    game_init(&g_game);
    ai_move_if_needed(); // X (the board) always opens

    WiFi.softAP(WIFI_SSID, WIFI_PASS);
    Serial.print("SoftAP up. Connect to WiFi \"");
    Serial.print(WIFI_SSID);
    Serial.print("\" (password \"");
    Serial.print(WIFI_PASS);
    Serial.println("\"),");
    Serial.print("then open http://");
    Serial.print(WiFi.softAPIP());
    Serial.println("/");

    server.on("/", HTTP_GET, handleIndex);
    server.on("/state", HTTP_GET, handleState);
    server.on("/move", HTTP_POST, handleMove);
    server.on("/reset", HTTP_POST, handleReset);
    server.begin();
}

void loop() {
    server.handleClient();
}
