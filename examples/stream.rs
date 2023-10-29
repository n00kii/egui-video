use eframe::NativeOptions;
use egui::CentralPanel;
use egui_video::Player;

fn main() {
    let _ = eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    );
}
struct App {
    player: Option<Player>,
}

impl Default for App {
    fn default() -> Self {
        Self { player: None }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| match self.player.as_mut() {
            None => {
                println!("Starting stream");
                self.player =
                    Some(Player::from_udp(ctx, "127.0.0.1:1234").expect("Media not found."));
            }
            Some(p) => {
                frame.set_window_size(p.size);
                p.ui(ui, p.size);
            }
        });
    }
}
