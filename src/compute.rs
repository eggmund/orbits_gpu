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

use ggez::nalgebra::{Point2, Vector2};

use std::sync::Arc;

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform TimeData {
    float dt;
};

struct BodyData
{
    vec2 position;
    vec2 velocity;
    vec2 acc;
    float mass;
};

layout(set = 0, binding = 1) buffer Data {
    BodyData body_data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.body_data[idx].velocity += buf.body_data[idx].acc * dt;
    buf.body_data[idx].position += buf.body_data[idx].velocity * dt;
}"
    }
}

#[derive(Copy, Clone)]
pub struct Body {
    pub pos: Point2<f32>,
    vel: Vector2<f32>,
    acc: Vector2<f32>,
    mass: f32,
}

impl Body {
    pub fn new(pos: Point2<f32>, vel: Vector2<f32>, acc: Vector2<f32>) -> Body {
        Body {
            pos,
            vel,
            acc,
            mass: 10.0,
        }
    }
}

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
        let data_len = data.len();
        // Buffer for dt. Starting value of 60fps
        let time_buf = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::uniform_buffer(), false, 1.0/60.0)
            .expect("failed to create time data buffer");
    
        // Positions and velocities buffer
        let bodies_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true, data.into_iter())
            .expect("failed to create positions buffer");
    
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
}
