extern crate ffmpeg_next as ffmpeg;
use eframe::NativeOptions;
use egui::{CentralPanel, Grid, Sense, Slider, TextEdit};
use egui_video::VideoStream;
fn main() {
    ffmpeg::init().unwrap();
    eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    )
}
struct App {
    media_path: String,
    stream_size_scale: f32,
    video_stream: Option<VideoStream>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            media_path: String::new(),
            stream_size_scale: 1.,
            video_stream: None,
        }
    }
}

impl eframe::App for App {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(streamer) = self.video_stream.take() {
            streamer.cleanup();
        }
    }
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            let tedit_resp = ui.add(
                TextEdit::singleline(&mut self.media_path)
                    .hint_text("click to set path")
                    .interactive(false),
            );

            if ui
                .interact(tedit_resp.rect, tedit_resp.id.with("click_sense"), Sense::click())
                .clicked()
            {
                if let Some(path_buf) = rfd::FileDialog::new().pick_file() {
                    self.media_path = path_buf.as_path().to_string_lossy().to_string();
                }
            }
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
                ui.label(format!("{:?}", streamer.player_state.get()));

                ui.checkbox(&mut streamer.looping, "loop");
                ui.add(Slider::new(&mut self.stream_size_scale, 0.0..=1.));
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
                        [
                            streamer.width as f32 * self.stream_size_scale,
                            streamer.height as f32 * self.stream_size_scale,
                        ],
                    );
                });
            }
        });
    }
}
