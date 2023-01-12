extern crate ffmpeg_next as ffmpeg;
use eframe::NativeOptions;
use egui::{CentralPanel, Grid, Sense, Slider, TextEdit};
use egui_video::{Player, AudioStreamerCallback, AudioStreamerDevice};
fn main() {
    ffmpeg::init().unwrap();
    eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    )
}
struct App {
    audio_device: AudioStreamerDevice,
    media_path: String,
    stream_size_scale: f32,
    player: Option<Player>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            audio_device: AudioStreamerCallback::init(&sdl2::init().unwrap().audio().unwrap()).unwrap(),
            media_path: String::new(),
            stream_size_scale: 1.,
            player: None,
        }
    }
}

impl eframe::App for App {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(streamer) = self.player.take() {
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
                .interact(
                    tedit_resp.rect,
                    tedit_resp.id.with("click_sense"),
                    Sense::click(),
                )
                .clicked()
            {
                if let Some(path_buf) = rfd::FileDialog::new().pick_file() {
                    self.media_path = path_buf.as_path().to_string_lossy().to_string();
                }
            }
            if ui.button("load").clicked() {
                match Player::new(ctx, &self.media_path.replace("\"", "")).and_then(|p| p.with_audio(&mut self.audio_device)) {
                    Ok(player) => {
                        self.player = Some(player);
                    }
                    Err(e) => println!("failed to make stream: {e}"),
                }
            }
            ctx.request_repaint();
            if let Some(player) = self.player.as_mut() {
                player.process_state();
                ui.label(format!("frame rate: {}", player.framerate));
                ui.label(format!("size: {}x{}", player.width, player.height));
                ui.label(player.duration_text());
                ui.label(format!("{:?}", player.player_state.get()));

                ui.checkbox(&mut player.looping, "loop");
                ui.add(Slider::new(&mut self.stream_size_scale, 0.0..=1.));
                if ui.button("start playing").clicked() {
                    player.start()
                }
                if ui.button("play").clicked() {
                    player.unpause();
                }
                if ui.button("pause").clicked() {
                    player.pause();
                }
                if ui.button("stop").clicked() {
                    player.stop();
                }
                Grid::new("h").show(ui, |ui| {
                    player.ui(
                        ui,
                        [
                            player.width as f32 * self.stream_size_scale,
                            player.height as f32 * self.stream_size_scale,
                        ],
                    );
                });
            }
        });
    }
}
