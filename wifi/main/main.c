#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "game.h"
#include "index_html.h"
#include "net_infer.h"

// Phone connects directly to this network -- no router/internet needed.
// WPA2 requires >=8 chars; set WIFI_PASS to "" for an open network instead.
#define WIFI_SSID "UltimateTTT-AI"
#define WIFI_PASS "tictactoe"
#define WIFI_CHANNEL 1
#define MAX_STA_CONN 4

static const char *TAG = "ttt_wifi";
static Game g_game; // single shared game -- this device serves one player at a time

static void ai_move_if_needed(void) {
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
        if (best_a < 0) break; // shouldn't happen while !done, but stay safe
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

static esp_err_t index_get_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t state_get_handler(httpd_req_t *req) {
    char buf[1600];
    int n = state_to_json(buf, sizeof(buf), false);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, buf, n);
}

static esp_err_t move_post_handler(httpd_req_t *req) {
    char body[64];
    int total = req->content_len;
    if (total <= 0 || total >= (int)sizeof(body)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad body");
        return ESP_FAIL;
    }
    int received = httpd_req_recv(req, body, total);
    if (received <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "recv failed");
        return ESP_FAIL;
    }
    body[received] = '\0';

    int sub = -1, cell = -1;
    sscanf(body, "{\"sub\":%d,\"cell\":%d}", &sub, &cell);

    bool error = false;
    if (g_game.done || g_game.player != -1 || sub < 0 || sub > 8 ||
        cell < 0 || cell > 8 || !game_is_legal_move(&g_game, sub, cell)) {
        error = true;
    } else {
        game_play(&g_game, sub, cell); // human (O) move
        ai_move_if_needed();            // board (X) replies immediately
    }

    char buf[1600];
    int n = state_to_json(buf, sizeof(buf), error);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, buf, n);
}

static esp_err_t reset_post_handler(httpd_req_t *req) {
    game_init(&g_game);
    ai_move_if_needed(); // X always moves first
    char buf[1600];
    int n = state_to_json(buf, sizeof(buf), false);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, buf, n);
}

static httpd_handle_t start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 8;
    config.stack_size = 8192; // handlers use ~1.6KB JSON buffers on the stack

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "failed to start http server");
        return NULL;
    }

    httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_get_handler};
    httpd_uri_t state_uri = {.uri = "/state", .method = HTTP_GET, .handler = state_get_handler};
    httpd_uri_t move_uri = {.uri = "/move", .method = HTTP_POST, .handler = move_post_handler};
    httpd_uri_t reset_uri = {.uri = "/reset", .method = HTTP_POST, .handler = reset_post_handler};
    httpd_register_uri_handler(server, &index_uri);
    httpd_register_uri_handler(server, &state_uri);
    httpd_register_uri_handler(server, &move_uri);
    httpd_register_uri_handler(server, &reset_uri);

    return server;
}

static void wifi_init_softap(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = WIFI_SSID,
            .ssid_len = strlen(WIFI_SSID),
            .channel = WIFI_CHANNEL,
            .password = WIFI_PASS,
            .max_connection = MAX_STA_CONN,
            .authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    if (strlen(WIFI_PASS) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "SoftAP up. Connect to WiFi \"%s\" (password \"%s\"), then open http://192.168.4.1/",
              WIFI_SSID, WIFI_PASS);
}

void app_main(void) {
    ESP_ERROR_CHECK(nvs_flash_init());

    game_init(&g_game);
    ai_move_if_needed(); // X (the board) always opens

    wifi_init_softap();
    start_webserver();
}
