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

struct PosVelData
{
    vec2 position;
    vec2 velocity;
};

layout(set = 0, binding = 1) buffer Data {
    PosVelData pos_vel_data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.pos_vel_data[idx].position += buf.pos_vel_data[idx].velocity * dt;
}"
    }
}

#[derive(Copy, Clone)]
pub struct PosVelData {
    pub pos: [f32; 2],
    vel: [f32; 2],
}

impl PosVelData {
    pub fn new(pos: [f32; 2], vel: [f32; 2]) -> PosVelData {
        PosVelData {
            pos,
            vel,
        }
    }
}

pub struct Buffers {
    pub time: Arc<CpuAccessibleBuffer<f32>>,
    pub pos_vel: Arc<CpuAccessibleBuffer<[PosVelData]>>,
}

pub struct VulkanInstance {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub compute_pipeline: Arc<ComputePipeline<PipelineLayout<cs::Layout>>>,
    pub descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    pub buffers: Buffers,
}

impl VulkanInstance {
    pub fn new(pos_vel_data: Vec<PosVelData>) -> VulkanInstance {
        let (device, queue) = Self::setup_vulkan_device_and_queue();
        let buffers = Self::setup_data_buffers(device.clone(), pos_vel_data);
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
    
    fn setup_data_buffers(device: Arc<Device>, data: Vec<PosVelData>) -> Buffers {
        let data_len = data.len();
        // Buffer for dt. Starting value of 60fps
        let time_buf = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::uniform_buffer(), false, 1.0/60.0)
            .expect("failed to create time data buffer");
    
        // Positions and velocities buffer
        let pos_vel_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true, data.into_iter())
            .expect("failed to create positions buffer");
    
        Buffers {
            time: time_buf,
            pos_vel: pos_vel_buf,
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
            .add_buffer(buffers.pos_vel.clone()).unwrap()
            .build().unwrap()
        );
    
        (compute_pipeline, descriptor_set)
    }
}
