#include "game.h"
#include <string.h>
#include <stdio.h>

static const int WIN_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};

static int check_winner9(const int8_t cells9[9]) {
    for (int i = 0; i < 8; i++) {
        int a = WIN_LINES[i][0], b = WIN_LINES[i][1], c = WIN_LINES[i][2];
        int s = cells9[a] + cells9[b] + cells9[c];
        if (s == 3) return 1;
        if (s == -3) return -1;
    }
    return 0;
}

void game_init(Game *g) {
    memset(g, 0, sizeof(*g));
    g->player = 1;
    g->active_sub = -1;
    g->winner = 0;
    g->done = false;
}

bool game_legal_subboards(const Game *g, int out_subs[9], int *n_subs) {
    *n_subs = 0;
    if (g->active_sub != -1 && g->sub_result[g->active_sub] == 0) {
        out_subs[(*n_subs)++] = g->active_sub;
        return true;
    }
    for (int s = 0; s < 9; s++) {
        if (g->sub_result[s] == 0) out_subs[(*n_subs)++] = s;
    }
    return *n_subs > 0;
}

bool game_is_legal_move(const Game *g, int sub, int cell) {
    if (g->done) return false;
    if (g->sub_result[sub] != 0) return false;
    if (g->board[sub][cell] != 0) return false;
    int subs[9], n;
    game_legal_subboards(g, subs, &n);
    for (int i = 0; i < n; i++) if (subs[i] == sub) return true;
    return false;
}

void game_legal_mask81(const Game *g, float mask[81]) {
    memset(mask, 0, sizeof(float) * 81);
    int subs[9], n;
    game_legal_subboards(g, subs, &n);
    for (int i = 0; i < n; i++) {
        int s = subs[i];
        for (int c = 0; c < 9; c++) {
            if (g->board[s][c] == 0) mask[s * 9 + c] = 1.0f;
        }
    }
}

bool game_play(Game *g, int sub, int cell) {
    g->board[sub][cell] = (int8_t)g->player;

    int w = check_winner9(g->board[sub]);
    if (w != 0) {
        g->sub_result[sub] = (int8_t)w;
    } else {
        bool full = true;
        for (int c = 0; c < 9; c++) if (g->board[sub][c] == 0) { full = false; break; }
        if (full) g->sub_result[sub] = 2;
    }

    int8_t big_cells[9];
    for (int s = 0; s < 9; s++) big_cells[s] = (g->sub_result[s] == 2) ? 0 : g->sub_result[s];
    int big_w = check_winner9(big_cells);

    if (big_w != 0) {
        g->winner = big_w;
        g->done = true;
    } else {
        bool all_decided = true;
        for (int s = 0; s < 9; s++) if (g->sub_result[s] == 0) { all_decided = false; break; }
        if (all_decided) {
            g->winner = 2;
            g->done = true;
        } else {
            g->active_sub = (g->sub_result[cell] == 0) ? cell : -1;
        }
    }

    g->player *= -1;
    return g->done;
}

void game_encode(const Game *g, float out[126]) {
    int p = g->player;
    for (int s = 0; s < 9; s++)
        for (int c = 0; c < 9; c++)
            out[s * 9 + c] = (float)(g->board[s][c] * p);

    int idx = 81;
    for (int s = 0; s < 9; s++) {
        int r = g->sub_result[s];
        float oh[4] = {0, 0, 0, 0};
        if (r == 0) oh[0] = 1;
        else if (r == p) oh[1] = 1;
        else if (r == -p) oh[2] = 1;
        else oh[3] = 1;
        out[idx++] = oh[0];
        out[idx++] = oh[1];
        out[idx++] = oh[2];
        out[idx++] = oh[3];
    }

    int subs[9], n;
    game_legal_subboards(g, subs, &n);
    float legal_flag[9] = {0};
    for (int i = 0; i < n; i++) legal_flag[subs[i]] = 1.0f;
    for (int s = 0; s < 9; s++) out[idx++] = legal_flag[s];
}

float game_result_for(const Game *g, int player) {
    if (g->winner == 2) return 0.0f;
    return (g->winner == player) ? 1.0f : -1.0f;
}

static char mark_char(int8_t v) {
    if (v == 1) return 'X';
    if (v == -1) return 'O';
    return '.';
}

void game_print(const Game *g) {
    for (int br = 0; br < 3; br++) {
        for (int sr = 0; sr < 3; sr++) {
            for (int bc = 0; bc < 3; bc++) {
                int sub = br * 3 + bc;
                for (int sc = 0; sc < 3; sc++) {
                    int cell = sr * 3 + sc;
                    printf("%c", mark_char(g->board[sub][cell]));
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("sub results: ");
    for (int s = 0; s < 9; s++) printf("%d ", g->sub_result[s]);
    printf("\n");
}
