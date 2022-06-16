#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "cuda_runtime_api.h"
#include "utils/CUDA/helper_math.h"
#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "ParticleSystem.h"

#define THREAD_PER_BLOCK 256
__constant__ ParticleSystemParameters dev_params;
__constant__ Grid dev_grid;


void checkError() {
	cudaDeviceSynchronize();
	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

__global__ void set_particles_kernel(float* position, std::uint32_t* color, const float* input_pos, const std::uint32_t* input_color, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_particles) {
		position[tid * 4 + 0] = input_pos[num_particles * 0 + tid];
		position[tid * 4 + 1] = input_pos[num_particles * 1 + tid];
		position[tid * 4 + 2] = input_pos[num_particles * 2 + tid];
		position[tid * 4 + 3] = input_pos[num_particles * 3 + tid];
		color[tid] = input_color[tid];
	}
}


__device__
int3 get_cell_index(uint cid) {
	uint x = cid % dev_grid.x_num;
	cid = cid / dev_grid.x_num;
	uint y = cid % dev_grid.y_num;
	uint z = cid / dev_grid.y_num;	
	return make_int3(x, y, z);
}


__global__ void 
__launch_bounds__(THREAD_PER_BLOCK)
update_particles_kernel(std::uint32_t* color, float* position, float* new_pos, float* new_last_pos, const float* cur_pos, const float* last_pos,
	const uint* cell_id, const uint* particle_id, const uint* start, const uint* end, float dt, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;
	__shared__ float shared_loc[THREAD_PER_BLOCK][4];
	__shared__ float shared_last[THREAD_PER_BLOCK][3];

	// unwrap parameters
	float x0 = dev_params.bb_min[0], y0 = dev_params.bb_min[1], z0 = dev_params.bb_min[2];
	float x1 = dev_params.bb_max[0], y1 = dev_params.bb_max[1], z1 = dev_params.bb_max[2];
	float bounce = dev_params.bounce;

	// read current particle information
	uint pid = particle_id[tid];
	uint cid;
	uint x_id = pid, y_id = pid + num_particles, z_id = pid + 2 * num_particles;
	// cur_x: the location before the current time step. next.x: the location after
	float cur_x = cur_pos[x_id], cur_y = cur_pos[y_id], cur_z = cur_pos[z_id];
	float3 cur = make_float3(cur_pos[x_id], cur_pos[y_id], cur_pos[z_id]);
	float3 last = make_float3(last_pos[x_id], last_pos[y_id], last_pos[z_id]);
	float r = cur_pos[num_particles * 3 + pid];  // radius

	// acceleration and velocity
	float3 a = make_float3(dev_params.gravity[0], dev_params.gravity[1], dev_params.gravity[2]);
	float3 dx = cur - last;
	float3 v = dx / dt;

	
	float c_spring = dev_params.coll_spring, c_damping = dev_params.coll_damping, c_shear = dev_params.coll_shear;

	cid = cell_id[tid];
	int3 cid_xyz = get_cell_index(cid);

	auto cur_block_min = blockIdx.x * blockDim.x;
	auto cur_block_max = (blockIdx.x + 1) * blockDim.x;
	
	shared_loc[threadIdx.x][0] = cur_x;
	shared_loc[threadIdx.x][1] = cur_y;
	shared_loc[threadIdx.x][2] = cur_z;
	shared_loc[threadIdx.x][3] = r;
	shared_last[threadIdx.x][0] = last.x;
	shared_last[threadIdx.x][1] = last.y;
	shared_last[threadIdx.x][2] = last.z;
	__syncthreads();
	int it;

	for (int cx = max(cid_xyz.x - 1, 0); cx < min(cid_xyz.x + 2, dev_grid.x_num); cx++)
		for (int cy = max(cid_xyz.y - 1, 0); cy < min(cid_xyz.y + 2, dev_grid.y_num); cy++)
			for (int cz = max(cid_xyz.z - 1, 0); cz < min(cid_xyz.z + 2, dev_grid.z_num); cz++) {
				// the particle that may interact with current particle
				uint pid2, cid2;
				float r2;
				float3 cur2, last2, dx2, v2;
				float3 vab, vtan_ab;
				float p_ab_norm;
				float3 fs, fd, ft;
				float3 p_ab, d_ab;


				cid2 = cz * dev_grid.x_num * dev_grid.y_num + cy * dev_grid.x_num + cx;
				for (int i = start[cid2]; i < end[cid2]; i++) {
					pid2 = particle_id[i];

					assert(cell_id[i] == cid2);
					if (i == tid)
						continue;
					if (i > cur_block_min && i < cur_block_max) {
						it = i - cur_block_min;
						cur2 = make_float3(shared_loc[it][0], shared_loc[it][1], shared_loc[it][2]);
						r2 = shared_loc[it][3];
					}
					else {
						cur2 = make_float3(cur_pos[pid2], cur_pos[pid2 + num_particles], cur_pos[pid2 + 2 * num_particles]);
						r2 = cur_pos[pid2 + 3 * num_particles];
					}
					
					p_ab = cur2 - cur;
					p_ab_norm = sqrt(dot(p_ab, p_ab));
					if (p_ab_norm <= r + r2) {
						d_ab = p_ab / p_ab_norm;
						if (i > cur_block_min && i < cur_block_max)
							last2 = make_float3(shared_last[it][0], shared_last[it][1], shared_last[it][2]);
						else
							last2 = make_float3(last_pos[pid2], last_pos[pid2 + num_particles], last_pos[pid2 + 2 * num_particles]);
						dx2 = cur2 - last2;
						v2 = dx2 / dt;
						vab = v2 - v;
						vtan_ab = vab - d_ab * dot(vab, d_ab);

						// forces
						fs = -c_spring * (r + r2 - p_ab_norm) * d_ab;
						fd = c_damping * vab;
						ft = c_shear * vtan_ab;
						a = a + fs + fd + ft;
					}
				}
			}

	// calculate next position
	float3 next = cur + dx + a * dt * dt;

	// collision with the bbox
	float min_t; int min_idx;
	float t[6];
	while (next.x < x0 + r || next.y < y0 + r || next.z < z0 + r || next.x > x1 - r || next.y > y1 - r || next.z > z1 - r) { // if outside the bounding box
		// use parametirc equation of the line to determine the intersection order 
		// min_idx: the first plane the line intersects
		min_idx = -1;
		min_t = 2;
		t[0] = (next.x == cur_x || next.x >= x0 + r) ? 2 : (x0 + r - cur_x) / (next.x - cur_x);
		t[1] = (next.y == cur_y || next.y >= y0 + r) ? 2 : (y0 + r - cur_y) / (next.y - cur_y);
		t[2] = (next.z == cur_z || next.z >= z0 + r) ? 2 : (z0 + r - cur_z) / (next.z - cur_z);
		t[3] = (next.x == cur_x || next.x <= x1 - r) ? 2 : (x1 - r - cur_x) / (next.x - cur_x);
		t[4] = (next.y == cur_y || next.y <= y1 - r) ? 2 : (y1 - r - cur_y) / (next.y - cur_y);
		t[5] = (next.z == cur_z || next.z <= z1 - r) ? 2 : (z1 - r - cur_z) / (next.z - cur_z);
#pragma unroll
		for (int i = 0; i < 6; ++i) {
			if (t[i] >= 0 && t[i] <= 1 && t[i] < min_t) {
				min_idx = i;
				min_t = t[i];
			}
		}
		assert(min_idx != -1);

		switch (min_idx) {
		case 0:
			next.x = x0 + r + bounce * (x0 + r - next.x);
			cur_x = x0 + r - bounce * (cur_x - x0 - r);
			break;
		case 1:
			next.y = y0 + r + bounce * (y0 + r - next.y);
			cur_y = y0 + r - bounce * (cur_y - y0 - r);
			break;
		case 2:
			next.z = z0 + r + bounce * (z0 + r - next.z);
			cur_z = z0 + r - bounce * (cur_z - z0 - r);
			break;
		case 3:
			next.x = x1 - r - bounce * (next.x - x1 + r);
			cur_x = x1 - r + bounce * (x1 - r - cur_x);
			break;
		case 4:
			next.y = y1 - r - bounce * (next.y - y1 + r);
			cur_y = y1 - r + bounce * (y1 - r - cur_y);
			break;
		case 5:
			next.z = z1 - r - bounce * (next.z - z1 + r);
			cur_z = z1 - r + bounce * (z1 - r - cur_z);
			break;
		}
	}
	/*assert(next.x <= x1 - r);
	assert(next.x <= dev_params.bb_max[0]);
	assert(next.y <= dev_params.bb_max[1]);
	assert(next.z <= dev_params.bb_max[2]);*/
	position[pid * 4 + 0] = next.x;
	position[pid * 4 + 1] = next.y;
	position[pid * 4 + 2] = next.z;
	new_pos[x_id] = next.x;
	new_pos[y_id] = next.y;
	new_pos[z_id] = next.z;
	new_last_pos[x_id] = cur_x;
	new_last_pos[y_id] = cur_y;
	new_last_pos[z_id] = cur_z;
}


__global__
__launch_bounds__(THREAD_PER_BLOCK)
void calc_cell_id(float* position, uint* cell_id, uint* particle_id, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;
	float x, y, z;
	x = position[tid * 4 + 0];
	y = position[tid * 4 + 1];
	z = position[tid * 4 + 2];
	int xx, yy, zz;

	xx = static_cast<int> ((x - dev_params.bb_min[0]) / dev_grid.x_size);
	yy = static_cast<int> ((y - dev_params.bb_min[1]) / dev_grid.y_size);
	zz = static_cast<int> ((z - dev_params.bb_min[2]) / dev_grid.z_size);
	assert(xx < dev_grid.x_num);

	cell_id[tid] = zz* dev_grid.x_num * dev_grid.y_num + yy * dev_grid.x_num + xx;
	particle_id[tid] = tid;
}

__global__
__launch_bounds__(THREAD_PER_BLOCK)
void find_start(uint* start, uint* end, uint* cell_id, uint* particle_id, std::size_t num_particles) {
	auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_particles) return;
	__shared__ int shared_cell_id[THREAD_PER_BLOCK+1];

	shared_cell_id[threadIdx.x+1] = cell_id[tid];
	if (threadIdx.x == 0) {
		if (tid != 0)
			shared_cell_id[0] = cell_id[tid - 1];
		else
			shared_cell_id[0] = 0;
	}
	__syncthreads();
	uint prev = shared_cell_id[threadIdx.x], cur = shared_cell_id[threadIdx.x + 1];
	if (prev != cur) {
		start[cur] = tid;
		end[prev] = tid;
	}
	if (tid == num_particles - 1)
		end[cur] = num_particles;
	
}


void copy_params(const ParticleSystemParameters* host_params, const Grid* host_grid) {
	cudaMemcpyToSymbol(dev_params, host_params, sizeof(ParticleSystemParameters));
	cudaMemcpyToSymbol(dev_grid, host_grid, sizeof(Grid));
}

void set_particles(float* position, std::uint32_t* color, float* pos, std::uint32_t* col, std::size_t num_particles) {
	int thread_per_block = 512;
	set_particles_kernel <<<(num_particles + thread_per_block - 1) / thread_per_block, thread_per_block >>> (position, color, pos, col, num_particles);
	cudaDeviceSynchronize();
}

void update_particles(std::uint32_t* color, float* position, float* new_pos, float* new_last_pos, float* cur_pos, float* last_pos, 
	uint* cell_id, uint* particle_id, uint* start, uint* end, float dt, std::size_t num_particles, uint cell_num) {
	int num_block = (num_particles + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	calc_cell_id <<<num_block, THREAD_PER_BLOCK >>> (position, cell_id, particle_id, num_particles);
	
	thrust::sort_by_key(thrust::device_ptr<uint>(cell_id), thrust::device_ptr<uint>(cell_id + num_particles), thrust::device_ptr<uint>(particle_id));
	find_start <<<num_block, THREAD_PER_BLOCK >>> (start, end, cell_id, particle_id, num_particles);
	
	update_particles_kernel <<<num_block, THREAD_PER_BLOCK >>> (color, position, new_pos, new_last_pos, cur_pos, last_pos, cell_id, particle_id, start, end, dt, num_particles);
	
	cudaDeviceSynchronize();
}


// assets\ref\task4a\12345_42.particlereplay
// assets\123_42.particles