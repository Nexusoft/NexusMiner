/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once

#define GPU_MAX 8
#define WORD_MAX 32
#define OFFSETS_MAX 24
#define WINDOW_BITS 5
#define WINDOW_SIZE (1 << WINDOW_BITS)
#define WINDOW_SIZE_DIV2 (1 << (WINDOW_BITS - 1))
