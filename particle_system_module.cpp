#include "ParticleSystem.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C"
{
	EXPORT ParticleSystem* create_particles(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	{
		return new ParticleSystem(num_particles, x, y, z, r, color, params);
	}

	EXPORT void reset_particles(ParticleSystem* particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color)
	{
		particles->reset(x, y, z, r, color);
	}

	EXPORT void update_particles(ParticleSystem* particles, float* position, std::uint32_t* color, float dt)
	{
		particles->update(position, color, dt);
	}

	EXPORT void destroy_particles(ParticleSystem* particles)
	{
		delete particles;
	}
}
