
### Strong scaling, N = 4096, kernel = byte

serial byte baseline: 19.285 ms/generation (0.870 Gcell/s)

| decomp | mode | p | grid | ms/gen | speedup | efficiency | vs p=1 | compute % | comm % | reps |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| block | blocking | 1 | 1x1 | 21.453 | 0.90 | 90% | 1.00 | 98 | 2 | 5 |
| block | blocking | 2 | 2x1 | 8.572 | 2.25 | 112% | 2.50 | 97 | 3 | 5 |
| block | blocking | 4 | 2x2 | 6.581 | 2.93 | 73% | 3.26 | 73 | 27 | 5 |
| block | blocking | 8 | 4x2 | 4.194 | 4.60 | 57% | 5.12 | 81 | 19 | 5 |
| block | nonblocking | 1 | 1x1 | 20.269 | 0.95 | 95% | 1.00 | 98 | 2 | 5 |
| block | nonblocking | 2 | 2x1 | 9.095 | 2.12 | 106% | 2.23 | 97 | 3 | 5 |
| block | nonblocking | 4 | 2x2 | 6.685 | 2.88 | 72% | 3.03 | 74 | 26 | 5 |
| block | nonblocking | 8 | 4x2 | 5.603 | 3.44 | 43% | 3.62 | 61 | 39 | 5 |
| strip | blocking | 1 | 1x1 | 20.309 | 0.95 | 95% | 1.00 | 100 | 0 | 5 |
| strip | blocking | 2 | 2x1 | 9.008 | 2.14 | 107% | 2.25 | 98 | 2 | 5 |
| strip | blocking | 4 | 4x1 | 6.241 | 3.09 | 77% | 3.25 | 76 | 24 | 5 |
| strip | blocking | 8 | 8x1 | 4.117 | 4.68 | 59% | 4.93 | 77 | 23 | 5 |
| strip | nonblocking | 1 | 1x1 | 18.916 | 1.02 | 102% | 1.00 | 100 | 0 | 5 |
| strip | nonblocking | 2 | 2x1 | 9.132 | 2.11 | 106% | 2.07 | 98 | 2 | 5 |
| strip | nonblocking | 4 | 4x1 | 6.070 | 3.18 | 79% | 3.12 | 75 | 25 | 5 |
| strip | nonblocking | 8 | 8x1 | 3.904 | 4.94 | 62% | 4.85 | 82 | 18 | 5 |

### Strong scaling, N = 16384, kernel = byte

serial byte baseline: 302.156 ms/generation (0.888 Gcell/s)

| decomp | mode | p | grid | ms/gen | speedup | efficiency | vs p=1 | compute % | comm % | reps |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| block | blocking | 1 | 1x1 | 305.523 | 0.99 | 99% | 1.00 | 99 | 1 | 5 |
| block | blocking | 2 | 2x1 | 142.391 | 2.12 | 106% | 2.15 | 96 | 4 | 5 |
| block | blocking | 4 | 2x2 | 90.180 | 3.35 | 84% | 3.39 | 80 | 20 | 5 |
| block | blocking | 8 | 4x2 | 64.569 | 4.68 | 58% | 4.73 | 89 | 11 | 5 |
| block | nonblocking | 1 | 1x1 | 326.889 | 0.92 | 92% | 1.00 | 99 | 1 | 5 |
| block | nonblocking | 2 | 2x1 | 155.744 | 1.94 | 97% | 2.10 | 96 | 4 | 5 |
| block | nonblocking | 4 | 2x2 | 91.602 | 3.30 | 82% | 3.57 | 81 | 19 | 5 |
| block | nonblocking | 8 | 4x2 | 64.419 | 4.69 | 59% | 5.07 | 91 | 9 | 5 |
| strip | blocking | 1 | 1x1 | 307.025 | 0.98 | 98% | 1.00 | 100 | 0 | 5 |
| strip | blocking | 2 | 2x1 | 138.943 | 2.17 | 109% | 2.21 | 100 | 0 | 5 |
| strip | blocking | 4 | 4x1 | 86.442 | 3.50 | 87% | 3.55 | 82 | 18 | 5 |
| strip | blocking | 8 | 8x1 | 63.303 | 4.77 | 60% | 4.85 | 90 | 10 | 5 |
| strip | nonblocking | 1 | 1x1 | 321.233 | 0.94 | 94% | 1.00 | 100 | 0 | 5 |
| strip | nonblocking | 2 | 2x1 | 131.917 | 2.29 | 115% | 2.44 | 99 | 1 | 5 |
| strip | nonblocking | 4 | 4x1 | 88.979 | 3.40 | 85% | 3.61 | 80 | 20 | 5 |
| strip | nonblocking | 8 | 8x1 | 61.090 | 4.95 | 62% | 5.26 | 94 | 6 | 5 |

### Halo volume per rank per generation (cells sent to other ranks)

| N | kernel | decomp | p | grid | measured cells | model cells | measured / model | measured bytes |
|--:|---|---|--:|---|--:|--:|--:|--:|
| 4096 | byte | block | 1 | 1x1 | 0 | 16384 | 0.00 | 0 |
| 4096 | byte | block | 2 | 2x1 | 8192 | 11585 | 0.71 | 8192 |
| 4096 | byte | block | 4 | 2x2 | 8196 | 8192 | 1.00 | 8196 |
| 4096 | byte | block | 8 | 4x2 | 6148 | 5793 | 1.06 | 6148 |
| 4096 | byte | strip | 1 | 1x1 | 0 | 8192 | 0.00 | 0 |
| 4096 | byte | strip | 2 | 2x1 | 8192 | 8192 | 1.00 | 8192 |
| 4096 | byte | strip | 4 | 4x1 | 8192 | 8192 | 1.00 | 8192 |
| 4096 | byte | strip | 8 | 8x1 | 8192 | 8192 | 1.00 | 8192 |
| 16384 | byte | block | 1 | 1x1 | 0 | 65536 | 0.00 | 0 |
| 16384 | byte | block | 2 | 2x1 | 32768 | 46341 | 0.71 | 32768 |
| 16384 | byte | block | 4 | 2x2 | 32772 | 32768 | 1.00 | 32772 |
| 16384 | byte | block | 8 | 4x2 | 24580 | 23170 | 1.06 | 24580 |
| 16384 | byte | strip | 1 | 1x1 | 0 | 32768 | 0.00 | 0 |
| 16384 | byte | strip | 2 | 2x1 | 32768 | 32768 | 1.00 | 32768 |
| 16384 | byte | strip | 4 | 4x1 | 32768 | 32768 | 1.00 | 32768 |
| 16384 | byte | strip | 8 | 8x1 | 32768 | 32768 | 1.00 | 32768 |
