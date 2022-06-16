#ifndef INCLUDED_PARTICLE_SYSTEM_MODULE
#define INCLUDED_PARTICLE_SYSTEM_MODULE

#pragma once

#include <cstddef>
#include <cstdint>


extern "C"
{
	struct ParticleSystemParameters
	{
		float bb_min[3];
		float bb_max[3];
		float min_particle_radius;
		float max_particle_radius;
		float gravity[3];
		//float damping;
		float bounce;
		float coll_attraction;
		float coll_damping;
		float coll_shear;
		float coll_spring;
	};

	struct ParticleSystem;

	using create_particles_func = ParticleSystem*(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);
	using reset_particles_func = void(ParticleSystem* particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color);
	using update_particles_func = void(ParticleSystem* particles, float* position, std::uint32_t* color, float dt);
	using destroy_particles_func = void(ParticleSystem* particles);
}

#endif // INCLUDED_PARTICLE_SYSTEM_MODULE
