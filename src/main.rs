mod compute;

use ggez::event::{self, KeyCode, KeyMods};
use ggez::{Context, GameResult};
use ggez::nalgebra::Point2;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;

use rand::prelude::*;

use std::f32::consts::PI;

use compute::VulkanInstance;
use compute::Body;

const G: f32 = 0.001;
pub const STAR_DENSITY: f32 = 50.0;
pub const BLACK_HOLE_DENSITY: f32 = 1.0e7;
const WORK_GROUP_SIZE: f32 = 64.0;
const TWO_PI: f32 = PI * 2.0;
const SCREEN_DIMS: (f32, f32) = (1000.0, 800.0);
const GALAXY_SIZE: usize = (1024 * 4) - 1;


macro_rules! starting_setup {       // The starting setup when you open the program and when you reset
    ($rand_thread: expr, $start_bodies: expr) => {
        let mut first_galaxy = Self::spawn_galaxy(
            $rand_thread,
            [SCREEN_DIMS.0/3.0, SCREEN_DIMS.1/2.0],
            Some([0.0, 20.0]),
            None,
            2.0,
            GALAXY_SIZE,
            [0.7, 1.5],
            [10.0, 150.0],
            true,
        );
        let mut second_galaxy = Self::spawn_galaxy(
            $rand_thread,
            [SCREEN_DIMS.0 - SCREEN_DIMS.0/3.0, SCREEN_DIMS.1/2.0],
            Some([0.0, -20.0]),
            None,
            2.0,
            GALAXY_SIZE,
            [0.7, 1.5],
            [10.0, 150.0],
            false,
        );
        $start_bodies.append(&mut first_galaxy);
        $start_bodies.append(&mut second_galaxy);

        // let start_bodies = vec![
        //     Body::new([300.0, SCREEN_DIMS.1/2.0], Some([0.0, 0.0]), 10.0, None, None),
        //     Body::new([600.0, SCREEN_DIMS.1/2.0], Some([0.0, 0.0]), 10.0, None, None),
        // ];

        // vec![Body::new([50.0, 100.0], Some([50.0, 0.0]), 10.0, None, None); 7000];
    };
}


mod tools {
    use ggez::nalgebra::Vector2;
    #[inline]
    pub fn get_components(mag: f32, angle: f32) -> Vector2<f32> {
        Vector2::new(mag * angle.cos(), mag * angle.sin())
    }
}

struct MainState {
    vk_instance: VulkanInstance,
    body_count: u32,
    rand_thread: ThreadRng,
    starting_update_num: usize,   // So that dt can stabilise
}

impl MainState {
    fn new(_ctx: &mut Context) -> GameResult<MainState> {
        let mut rand_thread = rand::thread_rng();
        let mut start_bodies = Vec::with_capacity(GALAXY_SIZE * 2 + 2);

        starting_setup!(&mut rand_thread, start_bodies);
        let body_count = start_bodies.len() as u32;

        let s = MainState {
            vk_instance: VulkanInstance::new(start_bodies),
            body_count,
            rand_thread,
            starting_update_num: 0,
        };
        Ok(s)
    }

    fn reset(&mut self) {
        let mut start_bodies = Vec::with_capacity(GALAXY_SIZE * 2 + 2);
        starting_setup!(&mut self.rand_thread, start_bodies);
        self.body_count = start_bodies.len() as u32;

        self.vk_instance.set_bodies_buffer(start_bodies);
    }

    fn spawn_galaxy(
        rand_thread: &mut ThreadRng,
        pos: [f32; 2],
        vel: Option<[f32; 2]>,
        black_hole_mass: Option<f32>,
        black_hole_radius: f32,
        star_num: usize,
        star_radius_range: [f32; 2],
        star_orbit_radius_range: [f32; 2],
        orbit_direction_clockwise: bool,
    ) -> Vec<Body> 
    {
        let mut bodies = Vec::with_capacity(star_num + 1);
        let black_hole_mass = black_hole_mass
            .unwrap_or_else(|| Body::mass_from_radius(black_hole_radius, BLACK_HOLE_DENSITY));
        
        let black_hole = Body::new(pos, vel, black_hole_radius, Some(black_hole_mass), Some(BLACK_HOLE_DENSITY));
        let frame_vel = vel.unwrap_or_else(|| [0.0; 2]);

        for _ in 0..star_num {
            let orbit_radius = black_hole.radius + rand_thread.gen_range(star_orbit_radius_range[0], star_orbit_radius_range[1]);
            let orbit_speed = Self::circular_orbit_speed(black_hole.mass, orbit_radius);
            let start_angle = rand_thread.gen_range(0.0, TWO_PI);
            let start_pos = tools::get_components(orbit_radius, start_angle);
            let start_velocity = tools::get_components(
                orbit_speed,
                if orbit_direction_clockwise {  // Velocity is perpendicular to position
                    start_angle + PI/2.0
                } else {
                    start_angle - PI/2.0
                }
            );
            let radius = rand_thread.gen_range(star_radius_range[0], star_radius_range[1]);

            bodies.push(Body::new(
                [pos[0] + start_pos.x, pos[1] + start_pos.y],
                Some([frame_vel[0] + start_velocity.x, frame_vel[1] + start_velocity.y]),
                radius,
                None,
                None,
            ));
        }

        bodies.push(black_hole);
        bodies
    }

    // Returns the magnitude of the velocity (speed) needed for a circular orbit around another body
    // Orbit is circular when the kinetic energy does not change.
    // K = GMm/2r  -- Derived from centripetal force (in circular motion) = gravitational force
    // GMm/2r = 1/2 mv^2
    // GM/2r = 1/2 v^2
    // sqrt(GM/r) = v
    #[inline]
    fn circular_orbit_speed(host_mass: f32, radius: f32) -> f32 {
        (G * host_mass/radius).sqrt()
    }
}

impl event::EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        use ggez::timer;
        const DT_UPDATE_THRESHOLD: usize = 10;  // Number of updates before dt will be updated to gpu

        let dt = timer::duration_to_f64(timer::delta(ctx)) as f32;
        // Update time buffer
        // This threshold is needed since dt varies wildly at the start and causes bodies to fly everywhere
        if self.starting_update_num > DT_UPDATE_THRESHOLD {  
            let mut time_data = self.vk_instance.buffers.time.write().unwrap();
            *time_data = dt;
        } else {
            self.starting_update_num += 1;
        }

        // Run compute pipeline
        let command_buffer = AutoCommandBufferBuilder::new(self.vk_instance.device.clone(), self.vk_instance.queue.family()).unwrap()
            .dispatch(
                [(self.body_count as f32/WORK_GROUP_SIZE).ceil() as u32, 1, 1],
                self.vk_instance.compute_pipeline.clone(),
                self.vk_instance.descriptor_set.clone(),
                ()
            ).unwrap()
            .build().unwrap();
        
        let finished = command_buffer.execute(self.vk_instance.queue.clone()).unwrap();


        finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        use ggez::timer;
        use ggez::graphics::{self, DrawParam, DrawMode, MeshBuilder};

        let mut render_mesh_builder = MeshBuilder::new();

        graphics::clear(ctx, [0.1, 0.1, 0.15, 1.0].into());
        // Read positions from buffer
        {
            let bodies_data = self.vk_instance.buffers.bodies.read().unwrap();

            for b in bodies_data.iter() {
                let pos = {
                    let pos_ref: &[f32; 2] = b.pos.as_ref();
                    Point2::new(pos_ref[0], pos_ref[1])
                };
                render_mesh_builder.circle(
                    DrawMode::fill(),
                    pos,
                    b.radius,
                    1.0,
                    [0.9, 0.9, 0.9, 1.0].into(),
                );
            }
        }

        let render_mesh = render_mesh_builder.build(ctx)?;
        graphics::draw(ctx, &render_mesh, DrawParam::new())?;

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

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        keycode: KeyCode,
        _keymod: KeyMods,
        _repeat: bool,
    ) {
        match keycode {
            KeyCode::R => self.reset(),
            _ => ()
        }
    }
}

pub fn main() -> GameResult {
    use ggez::conf::WindowMode;

    let cb = ggez::ContextBuilder::new("super_simple", "ggez")
        .window_mode(WindowMode::default().dimensions(SCREEN_DIMS.0, SCREEN_DIMS.1));
    let (ctx, event_loop) = &mut cb.build()?;
    let state = &mut MainState::new(ctx)?;
    event::run(ctx, event_loop, state)
}
