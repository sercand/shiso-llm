#pragma once


// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1] retired groups,
// [2 + group] ticket published to group. Self-resetting, zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)

// Workspace size
#define WORKSPACE_SIZE (16*1024*1024)

#define MAX_DEVICES 16
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  5
