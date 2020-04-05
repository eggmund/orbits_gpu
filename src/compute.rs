use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;

use glsl_layout::AsStd140;
use glsl_layout::vec2;
use glsl_layout::float;

use std::sync::Arc;

use crate::STAR_DENSITY;

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
#version 450

#define G 0.001

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) uniform TimeData {
    float dt;
};

struct Body
{
    vec2 position; // 1 2
    vec2 velocity; // 3 4
    float radius;  // 5
    float mass;    // 6
    // vec4 color;    // 7 8 9 10
};

layout(set = 0, binding = 1) buffer Data {
    Body body_data[];
} buf;


vec2 calculate_grav_force(uint idx) {
    uint max_index = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    vec2 resultant_force = vec2(0.0, 0.0);

    for (uint i = 0; i < max_index; i++) {
        if (i != idx) { // If not itself, get grav force between
            // Get vector going from this planet to the other
            vec2 dist_vec = buf.body_data[i].position - buf.body_data[idx].position;
            // F = GMm/r^2 * r_norm, r_norm = r/|r| -> GMm/r^3 * r
            float distance = length(dist_vec);
            // If they aren't colliding further than half into each other
            if (distance > buf.body_data[i].radius + buf.body_data[i].radius) {
                resultant_force += dist_vec * (G * buf.body_data[i].mass * buf.body_data[idx].mass)/pow(distance, 3);
            }
        }
    }

    return resultant_force;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (buf.body_data[idx].radius > 0.0) {
        vec2 acc = calculate_grav_force(idx)/buf.body_data[idx].mass;
        buf.body_data[idx].velocity += acc * dt;
        buf.body_data[idx].position += buf.body_data[idx].velocity * dt;
    }
}
",
        //path: "src/motion.cs",
    }
}

#[derive(Copy, Clone, Debug, AsStd140)]
pub struct Body {
    pub pos: vec2,
    pub vel: vec2,
    pub radius: float,
    pub mass: float,
}

impl Body {
    pub fn new(pos: [f32; 2], vel: Option<[f32; 2]>, radius: f32, mass: Option<f32>, density: Option<f32>) -> Body {
        Body {
            pos: pos.into(),
            vel: (vel.unwrap_or_else(|| [0.0; 2])).into(),
            radius: radius.into(),
            mass: mass.unwrap_or_else(|| {
                Self::mass_from_radius(radius, density.unwrap_or_else(|| STAR_DENSITY))
            }).into(),
        }
    }

    #[inline]
    pub fn mass_from_radius(r: f32, density: f32) -> f32 {
        use std::f32::consts::PI;
        // V = 4/3 pi r^3
        // d = m/v -> vd = m
        4.0/3.0 * PI * r.powi(3) * density
    }
}

// Body needs to have variables aligned properly for reading by GLSL
// type GLSLBody = <Body as AsStd140>::Std140;

pub struct Buffers {
    pub time: Arc<CpuAccessibleBuffer<f32>>,
    pub bodies: Arc<CpuAccessibleBuffer<[Body]>>,
}

pub struct VulkanInstance {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub compute_pipeline: Arc<ComputePipeline<PipelineLayout<cs::Layout>>>,
    pub descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    pub buffers: Buffers,
}

impl VulkanInstance {
    pub fn new(bodies_data: Vec<Body>) -> VulkanInstance {
        let (device, queue) = Self::setup_vulkan_device_and_queue();
        let buffers = Self::setup_data_buffers(device.clone(), bodies_data);
        let (compute_pipeline, descriptor_set) = Self::setup_compute_pipeline_and_descriptors(device.clone(), &buffers);

        VulkanInstance {
            device,
            queue,
            compute_pipeline,
            descriptor_set,
            buffers,
        }
    }

    fn setup_vulkan_device_and_queue() -> (Arc<Device>, Arc<Queue>) {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .expect("failed to create instance");
    
        let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
        
        let queue_family = physical.queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a compute queue family");
    
        let (device, mut queues) = {
            let mut extensions = DeviceExtensions::none();
            extensions.khr_storage_buffer_storage_class = true;

            Device::new(physical, &Features::none(), &extensions,
                        [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
        };
    
        (device, queues.next().unwrap())
    }
    
    fn setup_data_buffers(device: Arc<Device>, data: Vec<Body>) -> Buffers {
        let bodies = data.into_iter();
            // .map(|b| b.std140());

        // Buffer for dt. Starting value of 60fps
        let time_buf = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::uniform_buffer(), false, 1.0/60.0)
            .expect("failed to create time data buffer");
    
        // Positions and velocities buffer
        let bodies_buf = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            bodies
        ).expect("failed to create positions buffer");
    
        Buffers {
            time: time_buf,
            bodies: bodies_buf,
        }
    }
    
    fn setup_compute_pipeline_and_descriptors(
        device: Arc<Device>,
        buffers: &Buffers,
    ) -> (
        Arc<ComputePipeline<PipelineLayout<cs::Layout>>>,
        Arc<dyn DescriptorSet + Send + Sync>,
    )
    {
        let compute_shader = cs::Shader::load(device.clone())
            .expect("failed to create shader module");
        
        let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &compute_shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"));
    
    
        let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
        let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(buffers.time.clone()).unwrap()
            .add_buffer(buffers.bodies.clone()).unwrap()
            .build().unwrap()
        );
    
        (compute_pipeline, descriptor_set)
    }

    pub fn set_bodies_buffer(&mut self, new_bodies: Vec<Body>) {
        self.buffers.bodies = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::all(),
            true,
            new_bodies.into_iter()
        ).expect("failed to create positions buffer");

        self.rebuild_pipeline();
    }

    fn rebuild_pipeline(&mut self) {
        let (pipeline, descriptor_set) = Self::setup_compute_pipeline_and_descriptors(self.device.clone(), &self.buffers);
        self.compute_pipeline = pipeline;
        self.descriptor_set = descriptor_set;
    }
}
