use eframe::NativeOptions;
use egui::{CentralPanel, Vec2};
use egui_video::Player;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

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
                self.player = Some(
                    Player::new_udp(
                        ctx,
                        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 1234),
                    )
                    .expect("Media not found."),
                );
            }
            Some(p) => {
                frame.set_window_size(Vec2 {
                    x: p.width as f32,
                    y: p.height as f32,
                });
                p.ui(ui, [p.width as f32, p.height as f32]);
            }
        });
    }
}