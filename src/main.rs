extern crate ffmpeg_next as ffmpeg;
use derivative::Derivative;
use eframe::NativeOptions;
use egui::{CentralPanel, Grid};
use egui_video::VideoStream;
fn main() {
    ffmpeg::init().unwrap();
    eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    )
}
#[derive(Derivative)]
#[derivative(Default)]
struct App {
    #[derivative(Default(
        value = "\"F:/Archive/veryold/Docs/Custom Office Templates/moveit/fe.mp4\".to_string()"
    ))]
    media_path: String,
    video_stream: Option<VideoStream>,
}

impl eframe::App for App {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(streamer) = self.video_stream.take() {
            streamer.cleanup();
        }
    }
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.text_edit_singleline(&mut self.media_path);
            if ui.button("load").clicked() {
                match VideoStream::new(ctx, &self.media_path.replace("\"", "")) {
                    Ok(video_streamer) => self.video_stream = Some(video_streamer),
                    Err(e) => println!("failed to make stream: {e}"),
                }
            }
            ctx.request_repaint();
            if let Some(streamer) = self.video_stream.as_mut() {
                streamer.process_state();
                ui.label(format!("frame rate: {}", streamer.framerate));
                ui.label(format!("size: {}x{}", streamer.width, streamer.height));
                ui.label(streamer.duration_text());
                ui.label(format!("{:?}", streamer.player_state.try_lock().as_deref()));

                ui.checkbox(&mut streamer.looping, "loop");
                if ui.button("start playing").clicked() {
                    streamer.start()
                }
                if ui.button("play").clicked() {
                    streamer.unpause();
                }
                if ui.button("pause").clicked() {
                    streamer.pause();
                }
                if ui.button("stop").clicked() {
                    streamer.stop();
                }
                Grid::new("h").show(ui, |ui| {
                    streamer.ui(
                        ui,
                        [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5],
                    );
                });
            }
        });
    }
}
