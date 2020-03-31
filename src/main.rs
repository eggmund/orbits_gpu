mod compute;

use ggez::event;
use ggez::graphics::{self, Mesh, DrawParam, DrawMode};
use ggez::{Context, GameResult};
use ggez::nalgebra::{Point2};

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;

use compute::VulkanInstance;
use compute::PosVelData;

const WORK_GROUP_SIZE: f32 = 64.0;

struct MainState {
    vk_instance: VulkanInstance,
    body_count: usize,
}

impl MainState {
    fn new(_ctx: &mut Context) -> GameResult<MainState> {
        let start_pos_vel = vec![PosVelData::new([100.0, 100.0], [50.0, 0.0])];
        let body_count = start_pos_vel.len();

        let s = MainState {
            vk_instance: VulkanInstance::new(start_pos_vel),
            body_count,
        };
        Ok(s)
    }
}

impl event::EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        use ggez::timer;

        let dt = timer::duration_to_f64(timer::delta(ctx)) as f32;
        // Update time buffer
        {
            let mut time_data = self.vk_instance.buffers.time.write().unwrap();
            *time_data = dt;
        }

        // Run compute pipeline
        let command_buffer = AutoCommandBufferBuilder::new(self.vk_instance.device.clone(), self.vk_instance.queue.family()).unwrap()
            .dispatch([(self.body_count as f32/WORK_GROUP_SIZE).ceil() as u32, 1, 1], self.vk_instance.compute_pipeline.clone(), self.vk_instance.descriptor_set.clone(), ()).unwrap()
            .build().unwrap();
        
        let finished = command_buffer.execute(self.vk_instance.queue.clone()).unwrap();


        finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        use ggez::timer;

        graphics::clear(ctx, [0.0, 0.0, 0.0, 1.0].into());
        // Read positions from buffer
        {
            let mut pos_vel_data = self.vk_instance.buffers.pos_vel.read().unwrap();

            for p in pos_vel_data.iter() {
                let circ = Mesh::new_circle(
                    ctx,
                    DrawMode::stroke(2.0),
                    p.pos,
                    10.0,
                    1.0,
                    [0.9, 0.9, 0.9, 1.0].into(),
                )?;

                graphics::draw(ctx, &circ, DrawParam::new())?;
            }
        }

        let fps_text = graphics::Text::new(format!("FPS: {}", timer::fps(ctx)));
        graphics::draw(ctx, &fps_text, 
            DrawParam::new()
                .dest(Point2::from([10.0, 10.0]))
                .color([0.0, 1.0, 0.0, 1.0].into()
            )
        )?;

        graphics::present(ctx)?;
        Ok(())
    }
}

pub fn main() -> GameResult {
    let cb = ggez::ContextBuilder::new("super_simple", "ggez");
    let (ctx, event_loop) = &mut cb.build()?;
    let state = &mut MainState::new(ctx)?;
    event::run(ctx, event_loop, state)
}