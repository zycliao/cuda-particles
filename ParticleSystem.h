#ifndef INCLUDED_PARTICLE_SYSTEM
#define INCLUDED_PARTICLE_SYSTEM

#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <vector>


#include "particle_system_module.h"

typedef unsigned int uint;

struct Grid
{
	uint x_num;
	uint y_num;
	uint z_num;
	uint cell_num;
	float x_size;
	float y_size;
	float z_size;

	Grid() {}

	Grid(const ParticleSystemParameters& params) {
		x_size = y_size = z_size = params.max_particle_radius * 2;
		x_num = static_cast<int> ((params.bb_max[0] - params.bb_min[0]) / x_size) + 1;
		y_num = static_cast<int> ((params.bb_max[1] - params.bb_min[1]) / y_size) + 1;
		z_num = static_cast<int> ((params.bb_max[2] - params.bb_min[2]) / z_size) + 1;
		cell_num = x_num * y_num * z_num;
	}
};


class ParticleSystem
{
	const std::size_t num_particles;
	const ParticleSystemParameters params;
	float* pos0;
	float* pos, * pos2;  // two position arrays, so that the previous one frame can be saved
	float* last_pos, * last_pos2;
	uint* particle_id, * cell_id;  // for collision detection
	uint* start_id, * end_id;      // start, end of each cell (prefix sum)
	std::uint32_t* col;  // color
	bool use1 = true;    // the 
	bool setColor = false; // it becomes true after set_particles()
	Grid grid;
	// DEBUG
	int cnt = 0;

public:
	ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);

	void reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color);
	void update(float* position, std::uint32_t* color, float dt);
};


void set_particles(float* position, std::uint32_t* color, float* pos, std::uint32_t* col, std::size_t num_particles);
void update_particles(std::uint32_t* color, float* position, float* new_pos, float* new_last_pos, float* cur_pos, float* last_pos, 
	uint* cell_id, uint* particle_id, uint* start_id, uint* end_id, float dt, std::size_t num_particles, uint cell_num);
void copy_params(const ParticleSystemParameters* params, const Grid* grid);

#endif // INCLUDED_PARTICLE_SIMULATION
