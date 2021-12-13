typedef struct {
	float3 pos;
	float3 vel;
} Particle;

__kernel
void timestep(__global Particle *particles) {
	size_t id = get_global_id(0);

	particles[id].pos.x += particles[id].vel.x;
	particles[id].pos.y += particles[id].vel.y;
	particles[id].pos.z += particles[id].vel.z;
}