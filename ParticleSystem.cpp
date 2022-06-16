#include "ParticleSystem.h"


ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	grid = Grid(params);
	copy_params(&params, &grid);

	cudaMalloc((void**)&pos0, 4 * num_particles * sizeof(float));
	cudaMalloc((void**)&pos, 4 * num_particles * sizeof(float));
	cudaMalloc((void**)&pos2, 4 * num_particles * sizeof(float));
	cudaMalloc((void**)&last_pos, 4 * num_particles * sizeof(float));
	cudaMalloc((void**)&last_pos2, 4 * num_particles * sizeof(float));
	cudaMalloc((void**)&col, num_particles * sizeof(std::uint32_t));
	cudaMalloc((void**)&particle_id, num_particles * sizeof(uint));
	cudaMalloc((void**)&cell_id, num_particles * sizeof(uint));
	cudaMalloc((void**)&start_id, grid.cell_num * sizeof(uint));
	cudaMalloc((void**)&end_id, grid.cell_num * sizeof(uint));
	
	reset(x, y, z, r, color);
}

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color)
{
	// TODO: reset particle system to the given state
	cudaMemcpy(pos + 0 * num_particles, x, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 1 * num_particles, y, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 2 * num_particles, z, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 3 * num_particles, r, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos2, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(last_pos, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(last_pos2, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(col, color, num_particles * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
	use1 = true;
}



void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	// TODO: update particle system by timestep dt (in seconds)
	//       position and color are device pointers to write-only buffers to receive the result
	if (!setColor) {
		setColor = !setColor;
		set_particles(position, color, pos, col, num_particles);
		return;
	}
	cudaMemset(start_id, 0, grid.cell_num * sizeof(uint));
	cudaMemset(end_id, 0, grid.cell_num * sizeof(uint));
	if (use1)
		update_particles(color, position, pos2, last_pos2, pos, last_pos, cell_id, particle_id, start_id, end_id, dt, num_particles, grid.cell_num);
	else
		update_particles(color, position, pos, last_pos, pos2, last_pos2, cell_id, particle_id, start_id, end_id, dt, num_particles, grid.cell_num);
	use1 = !use1;
	cudaMemcpy(pos0, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pos0, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pos0, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pos0, pos, 4 * num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
}
