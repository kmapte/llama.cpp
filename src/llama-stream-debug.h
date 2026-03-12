#pragma once

/**
 * llama-stream-debug.h
 *
 * Compile-time debug flag for streaming weight loader.
 * To enable: add -DLLAMA_STREAM_DEBUG=1 to CMake or just set to 1 here.
 * To disable: set to 0 — zero overhead, all prints compiled out.
 */

#ifndef LLAMA_STREAM_DEBUG
#  define LLAMA_STREAM_DEBUG 1   // change to 0 to silence all stream debug output
#endif

#if LLAMA_STREAM_DEBUG
#  include <cstdio>
#  define STREAM_DBG(fmt, ...) fprintf(stderr, "[STREAM-DBG] " fmt "\n", ##__VA_ARGS__)
#else
#  define STREAM_DBG(fmt, ...) ((void)0)
#endif
