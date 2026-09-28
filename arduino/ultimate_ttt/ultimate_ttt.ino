#include <Arduino.h>
#include "game.h"
#include "net_infer.h"

// Human plays O (-1), typed as "sub cell" e.g. "4 0" over Serial Monitor.
// AI plays X (1), using the trained policy network (greedy: highest
// legal-move probability -- no on-device search, just a single NN forward
// pass per move).

static int ai_choose_move(const Game *g, int *out_sub, int *out_cell) {
    float state[126];
    float policy[81];
    game_encode(g, state);
    float value = net_infer(state, policy);

    float mask[81];
    game_legal_mask81(g, mask);

    int best_a = -1;
    float best_p = -1.0f;
    for (int a = 0; a < 81; a++) {
        if (mask[a] > 0 && policy[a] > best_p) {
            best_p = policy[a];
            best_a = a;
        }
    }
    if (best_a < 0) return 0;
    *out_sub = best_a / 9;
    *out_cell = best_a % 9;
    Serial.printf("AI (X) plays sub=%d cell=%d  (policy=%.3f, value=%.3f)\n",
                  *out_sub, *out_cell, best_p, value);
    return 1;
}

static void print_winner(const Game *g) {
    if (g->winner == 1) Serial.println("=== X (AI) wins! ===");
    else if (g->winner == -1) Serial.println("=== O (you) wins! ===");
    else Serial.println("=== draw ===");
}

// Blocking read of one line from Serial (waits until Enter is pressed).
static void read_line(char *buf, size_t maxlen) {
    size_t idx = 0;
    while (true) {
        if (Serial.available()) {
            char c = (char)Serial.read();
            if (c == '\r') continue;
            if (c == '\n') break;
            if (idx < maxlen - 1) buf[idx++] = c;
        }
    }
    buf[idx] = '\0';
}

void setup() {
    Serial.begin(115200);
    delay(300); // give the Serial Monitor a moment to attach

    Game g;
    game_init(&g);

    Serial.println();
    Serial.println("Ultimate Tic-Tac-Toe -- you are O, AI is X.");
    Serial.println("Enter moves as: sub cell   (0-8 0-8), e.g. '4 4'");
    Serial.println();

    while (!g.done) {
        game_print(&g);

        if (g.player == 1) {
            int sub, cell;
            if (ai_choose_move(&g, &sub, &cell)) {
                game_play(&g, sub, cell);
            }
        } else {
            int subs[9], n;
            game_legal_subboards(&g, subs, &n);
            Serial.print("Your turn (O). Legal sub-boards: ");
            for (int i = 0; i < n; i++) {
                Serial.print(subs[i]);
                Serial.print(' ');
            }
            Serial.println();
            Serial.print("> ");

            char line[64];
            read_line(line, sizeof(line));

            int sub = -1, cell = -1;
            if (sscanf(line, "%d %d", &sub, &cell) != 2) {
                Serial.println("Could not parse input, try again.");
                continue;
            }

            if (sub < 0 || sub > 8 || cell < 0 || cell > 8 ||
                !game_is_legal_move(&g, sub, cell)) {
                Serial.println("Illegal move, try again.");
                continue;
            }
            game_play(&g, sub, cell);
        }
    }

    game_print(&g);
    print_winner(&g);
}

void loop() {
    // game runs to completion inside setup(); nothing to do repeatedly.
    delay(1000);
}
