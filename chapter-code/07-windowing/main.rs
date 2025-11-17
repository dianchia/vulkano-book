//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

mod app;
mod fs;
mod vs;

use app::App;
use winit::event_loop::EventLoop;

pub fn main() {
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app).unwrap();
}
