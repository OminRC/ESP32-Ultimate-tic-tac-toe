#pragma once
#include <stdint.h>
#include <stdbool.h>

// Ultimate Tic-Tac-Toe game engine (C port of train/game.py).
// board[sub][cell]: 0 empty, 1 = player X, -1 = player O.

typedef struct {
    int8_t board[9][9];
    int8_t sub_result[9];   // 0 undecided, 1/-1 winner, 2 drawn
    int player;              // 1 or -1, whose turn it is
    int active_sub;          // -1 = any open sub-board
    int winner;               // 0 ongoing, 1/-1 winner, 2 draw
    bool done;
} Game;

void game_init(Game *g);
bool game_legal_subboards(const Game *g, int out_subs[9], int *n_subs);
bool game_is_legal_move(const Game *g, int sub, int cell);
void game_legal_mask81(const Game *g, float mask[81]);
bool game_play(Game *g, int sub, int cell); // returns g->done
void game_encode(const Game *g, float out[126]); // 126-dim NN input
float game_result_for(const Game *g, int player);
void game_print(const Game *g);
