extern crate ffmpeg_next as ffmpeg;

use eframe::NativeOptions;
use egui::{CentralPanel, Color32, ColorImage, ImageData, TextureHandle, TextureFilter, Ui, Widget, Response, Sense, vec2, Rounding};
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use std::fs::File;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::env;
use timer::{Guard, Timer};

use anyhow::Result;
use chrono::Duration;

#[derive(Default)]
struct App {
    media_path: String,
    tex_id: Option<TextureHandle>,
    frame_size: Option<[f32; 2]>,
    busy: bool,
    video_stream: Option<Arc<Mutex<VideoStream>>>,
    timer_guard: Option<Guard>,
    timer: Option<Timer>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| {
            ui.text_edit_singleline(&mut self.media_path);
            if ui.button("load").clicked() {
                self.busy = false;
                match VideoStream::start(ctx, &self.media_path) {
                    Ok(video_streamer) => {
                        self.frame_size = None;
                        self.video_stream = Some(Arc::new(Mutex::new(video_streamer)))
                    }
                    Err(e) => println!("failed to make stream: {e}"),
                }
            }
            if let Some(timer) = self.timer.as_mut() {
                if let Some(texture_handle) = self.tex_id.as_mut() {
                    if self.frame_size.is_none() {
                        if let Some(frame_size) = self.video_stream.as_ref().and_then(|s| {
                            s.try_lock()
                                .ok()
                                .and_then(|s| Some([s.width as f32, s.height as f32]))
                        }) {
                            self.frame_size = Some(frame_size);
                        }
                    }
                    if !self.busy {
                        if let Some(streamer_arc) = self.video_stream.as_ref() {
                            if let Ok(mut streamer) = streamer_arc.try_lock().as_deref_mut() {
                                ui.add(streamer);
                                ui.label(format!("frame index: {}", streamer.frame_index));
                                ui.label(format!("frame rate: {}", streamer.framerate));
                                ui.label(format!("size: {}x{}", streamer.width, streamer.height));
                                ui.label(format!("is eof? {}", streamer.eof));
                                if streamer.eof {
                                    self.timer_guard = None;
                                }
                                // texture_handle.set(image, filter)
                                if ui.button("next packet").clicked() {
                                    if let Err(e) = streamer.recieve_next_packet() {
                                        println!("failed to load next packet: {e}")
                                    }
                                }
                                if ui.button("next frame").clicked() {
                                    match streamer.recieve_next_frame() {
                                        Err(e) => println!("failed to load next frame: {e}"),
                                        Ok(frame) => {
                                            texture_handle.set(frame, egui::TextureFilter::Linear)
                                        }
                                    }
                                }
                                if ui.button("all packets till frame").clicked() {
                                    match streamer.recieve_next_packet_until_frame() {
                                        Err(e) => println!("failed to load aptf: {e}"),
                                        Ok(frame) => {
                                            texture_handle.set(frame, egui::TextureFilter::Linear)
                                        }
                                    }
                                }
                                if ui.button("play").clicked() {
                                    self.busy = true;
                                    println!("{:?}", streamer.framerate);
                                    let dur =
                                        Duration::milliseconds((1e+3 / streamer.framerate) as i64);
                                    let str = Arc::clone(streamer_arc);
                                    let mut handle = texture_handle.clone();
                                    let timer_guard = timer.schedule_repeating(dur, move || {
                                        let mut strr = str.lock().unwrap();
                                        match strr.recieve_next_packet_until_frame() {
                                            Ok(frame) => {
                                                handle.set(frame, strr.texture_fiter)
                                            }
                                            _ => (),
                                        }
                                    });
                                    self.timer_guard = Some(timer_guard)
                                }
                            }
                        } else {
                        }
                    }
                    // ui.image(texture_handle.id(), self.frame_size.unwrap_or([100., 100.]));

                } else {
                    self.tex_id = Some(ui.ctx().load_texture(
                        "vid_text",
                        ImageData::Color(ColorImage::example()),
                        egui::TextureFilter::Linear,
                    ));
                }
            } else {
                self.timer = Some(Timer::new());
            }
        });
    }
}

struct VideoStream {
    decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    frame_index: usize,
    input_context: Input,
    eof: bool,
    framerate: f64,
    scaler: Context,
    width: u32,
    texture_fiter: TextureFilter,
    texture_id: Option<TextureHandle>,
    height: u32,
}

impl Widget for &mut VideoStream {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        let size = [self.width as f32, self.height as f32];
        if let Some(texture_id) = self.texture_id.as_ref() {
            ui.image(texture_id.id(), size)
        } else {
            let (rect, response) = ui.allocate_at_least(size.into(), Sense::click());
            ui.painter().rect_filled(rect, Rounding::none(), Color32::BLACK);
            response
        }
    }
}

impl VideoStream {

    fn start(ctx: &egui::Context, input_path: &String) -> Result<Self> {
        let input_context = input(&input_path)?;
        let video_stream = input_context
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = video_stream.index();

        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let video_decoder = context_decoder.decoder().video()?;
        let framerate = (video_stream.avg_frame_rate().numerator() as f64)
            / video_stream.avg_frame_rate().denominator() as f64;

        let (width, height) = (video_decoder.width(), video_decoder.height());
        let frame_scaler = Context::get(
            video_decoder.format(),
            video_decoder.width(),
            video_decoder.height(),
            Pixel::RGB24,
            video_decoder.width(),
            video_decoder.height(),
            Flags::BILINEAR,
        )?;

        let mut streamer = Self {
            decoder: video_decoder,
            eof: false,
            video_stream_index,
            texture_fiter: TextureFilter::Linear,
            input_context,
            framerate,
            texture_id: None,
            width,
            height,
            frame_index: 0,
            scaler: frame_scaler,
        };

        if let Ok(first_frame) = streamer.recieve_next_packet_until_frame() {
            streamer.texture_id = Some(ctx.load_texture("vidstream", first_frame, streamer.texture_fiter))
        }

        Ok(streamer)
    }

    fn recieve_next_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.input_context.packets().next() {
            if stream.index() == self.video_stream_index {
                self.decoder.send_packet(&packet)?;
            }
        } else {
            self.decoder.send_eof()?;
            self.eof = true;
        }
        Ok(())
    }

    fn recieve_next_packet_until_frame(&mut self) -> Result<ColorImage> {
        if let Ok(color_image) = self.recieve_next_frame() {
            Ok(color_image)
        } else {
            self.recieve_next_packet()?;
            self.recieve_next_frame()
        }
    }

    fn recieve_next_frame(&mut self) -> Result<ColorImage> {
        let mut decoded_frame = Video::empty();
        match self.decoder.receive_frame(&mut decoded_frame) {
            Ok(()) => {
                let mut rgb_frame = Video::empty();
                self.scaler.run(&decoded_frame, &mut rgb_frame)?;

                let image = consume_frame(rgb_frame);
                self.frame_index += 1;

                Ok(image)
            }
            Err(e) => {
                return Err(e.into());
            }
        }
    }
}

fn main() {
    ffmpeg::init().unwrap();
    eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    )
}

fn main2() -> Result<()> {
    if let Ok(mut ictx) = input(&env::args().nth(1).expect("Cannot open file.")) {
        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = input.index();

        let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;

        let mut scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )?;

        let mut frame_index = 0;

        let mut receive_and_process_decoded_frames =
            |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
                let mut decoded = Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut rgb_frame = Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    save_file(&rgb_frame, frame_index).unwrap();
                    frame_index += 1;
                }
                Ok(())
            };

        for (stream, packet) in ictx.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;
                receive_and_process_decoded_frames(&mut decoder)?;
            }
        }

        decoder.send_eof()?;
        receive_and_process_decoded_frames(&mut decoder)?;
    }

    Ok(())
}
// use std::mem::transmute;
fn consume_frame(frame: Video) -> ColorImage {
    let size = [frame.width() as usize, frame.height() as usize];
    let data = frame.data(0);
    let stride = frame.stride(0);
    let pixel_size_bytes = 3;
    let byte_width: usize = pixel_size_bytes * frame.width() as usize;
    let height: usize = frame.height() as usize;
    let mut pixels = vec![];
    for line in 0..height {
        let begin = line * stride;
        let end = begin + byte_width;
        let data_line = &data[begin..end];
        pixels.extend(
            data_line
                .chunks_exact(pixel_size_bytes)
                .map(|p| Color32::from_rgb(p[0], p[1], p[2])),
        )
    }
    // dbg!(&size, &pixels.len());
    ColorImage { size, pixels }
}

fn save_file(frame: &Video, index: usize) -> std::result::Result<(), std::io::Error> {
    let mut file = File::create(format!("frame{}.ppm", index))?;
    file.write_all(format!("P6\n{} {}\n255\n", frame.width(), frame.height()).as_bytes())?;
    Ok(())
}
