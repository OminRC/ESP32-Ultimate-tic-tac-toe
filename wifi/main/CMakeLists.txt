idf_component_register(
    SRCS "main.c" "game.c" "net_infer.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_http_server
)
