extern crate ffmpeg_next as ffmpeg;

use anyhow::Result;
use chrono::Duration;
use derivative::Derivative;
use eframe::epaint::Shadow;
use eframe::NativeOptions;
use egui::{
    vec2, Align2, CentralPanel, Color32, ColorImage, FontId, Image, ImageData, Rect, Response,
    Rounding, Sense, Stroke, TextureFilter, TextureHandle, Ui, Widget, Grid,
};
use ffmpeg::ffi::AV_TIME_BASE;
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, stream, Pixel};
use ffmpeg::media::Type;
use ffmpeg::{rescale, Rational};
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use tempfile::{tempfile, NamedTempFile};
use std::fs::File;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::{env, thread};
use timer::{Guard, Timer};

#[derive(Derivative)]
#[derivative(Default)]
struct App {
    #[derivative(Default(
        value = "\"F:/Archive/veryold/Docs/Custom Office Templates/moveit/fe.mp4\".to_string()"
    ))]
    media_path: String,
    video_stream: Option<VideoStream>,
}

const b: &[u8] = include_bytes!("../cat.gif");
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
                // self.busy = false;
                match VideoStream::new(ctx, &self.media_path.replace("\"", "")) {
                    Ok(video_streamer) => self.video_stream = Some(video_streamer),
                    Err(e) => println!("failed to make stream: {e}"),
                }
                // if ui.button("load").clicked() {
                    // self.busy = false;
                    
                    // match VideoStream::new_from_bytes(ctx, b) {
                    //     Ok(video_streamer) => self.video_stream = Some(video_streamer),
                    //     Err(e) => println!("failed to make stream: {e}"),
                    // }
                // }
            }
            ctx.request_repaint();
            if let Some(streamer) = self.video_stream.as_mut() {
                ui.label(format!("frame rate: {}", streamer.framerate));
                ui.label(format!("size: {}x{}", streamer.width, streamer.height));
                ui.label(format!(
                    "{} / {}",
                    Duration::milliseconds(streamer.elapsed_ms),
                    Duration::milliseconds(streamer.duration_ms)
                    // streamer.total_duration
                ));
                ui.label(format!("{:?}", streamer.player_state.try_lock().as_deref()));

                ui.checkbox(&mut streamer.looping, "loop");
                if ui.button("start playing").clicked() {
                    streamer.play()
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
                    streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // ui.end_row();
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // ui.end_row();
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // streamer.ui(ui, [streamer.width as f32 * 0.5, streamer.height as f32 * 0.5]);
                    // ui.end_row();
                });
            }
        });
    }
}

struct VideoStream {
    stream_decoder: Arc<Mutex<StreamDecoder>>,
    player_state: Arc<Mutex<PlayerState>>,
    framerate: f64,
    texture_fiter: TextureFilter,
    texture_handle: Option<TextureHandle>,
    height: u32,
    width: u32,
    frame_timer: Timer,
    frame_thread: Option<Guard>,
    ctx_ref: egui::Context,
    looping: bool,
    total_duration: Duration,
    current_duration: Duration,
    duration_ms: i64,
    elapsed_ms: i64,
    temp_file: Option<NamedTempFile>
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum PlayerState {
    Stopped,
    EndOfFile,
    Seeking(f32),
    Paused,
    Playing,
}

struct StreamDecoder {
    decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    frame_index: usize,
    player_state: Arc<Mutex<PlayerState>>,
    input_context: Input,
    scaler: Context,
    current_ts: i64,
}

use ffmpeg::Rescale;
const AV_TIME_BASE_RATIONAL: Rational = Rational(1, AV_TIME_BASE);

fn timestamp_to_millisec(timestamp: i64, time_base: Rational) -> i64 {
    (timestamp as f64 * (time_base.numerator() as f64) / (time_base.denominator() as f64) * 1000.) as i64
}

impl VideoStream {
    fn set_state(&self, new_state: PlayerState) {
        let mut current_state = self.player_state.lock().unwrap();
        *current_state = new_state;
    }
    fn pause(&self) {
        self.set_state(PlayerState::Paused)
    }
    fn unpause(&self) {
        self.set_state(PlayerState::Playing)
    }
    fn stop(&self) {
        self.set_state(PlayerState::Stopped)
    }
    fn duration_frac(&self) -> f32 {
        self.elapsed_ms as f32 / self.duration_ms as f32
    }
    // fn current_duration(&self) -> Duration {
    //     Duration::milliseconds(
    //         (self.total_duration.num_milliseconds() as f32 * self.duration_frac()) as i64,
    //     )
    // }
    fn cleanup(self) {
        drop(self)
    }
    fn spawn_timer(&mut self) {
        if let Some(texture_handle) = self.texture_handle.as_ref() {
            let mut texture_handle = texture_handle.clone();
            let texture_filter = self.texture_fiter.clone();
            let ctx = self.ctx_ref.clone();
            let stream_decoder = Arc::clone(&self.stream_decoder);
            let player_state = Arc::clone(&self.player_state);
            let wait_duration = Duration::milliseconds((1e+3 / self.framerate) as i64);
            let max_ts = self.duration_ms;

            let timer_guard = self.frame_timer.schedule_repeating(wait_duration, move || {
                ctx.request_repaint();
                let mut stream_decoder = stream_decoder.lock().unwrap();
                let player_state = player_state.lock().unwrap().clone();

                if player_state == PlayerState::Playing {
                    match stream_decoder.recieve_next_packet_until_frame() {
                        Ok(frame) => {
                            texture_handle.set(frame, texture_filter);
                        }
                        _ => (),
                    }
                } else if let PlayerState::Seeking(seek_frac) = player_state {
                    let target: i64 =
                        (seek_frac * stream_decoder.input_context.duration() as f32) as i64;
                    let _ = stream_decoder.input_context.seek(target, ..target);
                    if seek_frac > 0. {
                        // this drop frame loop lets us refresh until current_ts is accurate
                        while (stream_decoder.current_ts as f64 / max_ts as f64) >= seek_frac as f64
                        {
                            // println!("loop 1, {}, {}", stream_decoder.current_ts, seek_frac);
                            stream_decoder.drop_frames();
                        }

                        // if seek_frac < 0.99 {
                        // this drop frame loop drops frames until we are at desired
                        while ((stream_decoder.current_ts as f64 / max_ts as f64)
                            <= seek_frac as f64)
                            && (1. - (stream_decoder.current_ts as f64 / max_ts as f64) > 0.007)
                        // magic number to prevent inifinite loop near end of stream
                        {
                            // println!(
                            //     "loop 2, {} {} {seek_frac}",
                            //     stream_decoder.current_ts,
                            //     (stream_decoder.current_ts as f64 / max_ts as f64)
                            // );
                            stream_decoder.drop_frames();
                        }
                        *stream_decoder.player_state.lock().as_deref_mut().unwrap() =
                            PlayerState::Playing;
                    }
                }
            });
            self.frame_thread = Some(timer_guard)
        }
    }

    fn play(&mut self) {
        // let wait_duration = Duration::milliseconds((1e+3 / self.framerate) as i64);
        if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
            stream_decoder.reset(true);
        }

        self.spawn_timer();
    }

    fn ui(&mut self, ui: &mut Ui, size: [f32; 2]) -> egui::Response {
        let mut reset_stream = false;
        if let Ok(player_state) = self.player_state.try_lock().as_deref_mut() {
            if let Ok(mut stream_decoder) = self.stream_decoder.try_lock() {
                self.current_duration = Duration::milliseconds(
                    (1e3 * stream_decoder.frame_index as f64 / self.framerate) as i64,
                );
                self.elapsed_ms = stream_decoder.current_ts; //timestamp_to_millisec(stream_decoder.current_ts, stream_decoder.decoder.time_base());
                match player_state {
                    PlayerState::Paused => (),
                    PlayerState::Playing => {}
                    PlayerState::Seeking(_) => (),
                    PlayerState::EndOfFile => {
                        if self.looping {
                            reset_stream = true;
                        } else {
                            self.frame_thread = None;
                            *player_state = PlayerState::Stopped;
                        }
                    }
                    PlayerState::Stopped => {
                        self.frame_thread = None;
                        stream_decoder.frame_index = 0;
                    }
                }
            }
        }

        if reset_stream {
            if let Ok(mut stream_decoder) = self.stream_decoder.try_lock() {
                stream_decoder.reset(true);
            }
        }

        if let Some(texture_id) = self.texture_handle.as_ref() {
            let image = Image::new(texture_id.id(), size).sense(Sense::click());
            let response = ui.add(image);
            let seekbar_rect = self.render_seekbar(ui, &response);
            self.render_pause(ui, &response, seekbar_rect);
            response
        } else {
            let (rect, response) = ui.allocate_at_least(size.into(), Sense::click());
            ui.painter()
                .rect_filled(rect, Rounding::none(), Color32::BLACK);
            response
        }
    }

    fn render_seekbar(&mut self, ui: &mut Ui, playback_response: &Response) -> Option<Rect> {
        let hovered = ui.rect_contains_pointer(playback_response.rect);
        let currently_seeking = matches!(
            self.player_state.try_lock().as_deref(),
            Ok(PlayerState::Seeking(_))
        );
        let stopped = matches!(
            self.player_state.try_lock().as_deref(),
            Ok(PlayerState::Stopped)
        );

        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            playback_response.id.with("seekbar_anim"),
            hovered || currently_seeking,
            0.2,
        );

        if seekbar_anim_frac > 0. {
            let seekbar_width_offset = 20.;
            let fullseekbar_width = playback_response.rect.width() - seekbar_width_offset;

            let seekbar_width =
                if let Ok(PlayerState::Seeking(h)) = self.player_state.try_lock().as_deref() {
                    fullseekbar_width * h
                } else {
                    fullseekbar_width * self.duration_frac()
                };

            let seekbar_offset = 20.;
            let seekbar_pos = playback_response.rect.left_bottom()
                + vec2(seekbar_width_offset / 2., -seekbar_offset);
            let seekbar_height = 3.;
            let mut fullseekbar_rect =
                Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

            let mut seekbar_rect =
                Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
            let seekbar_hoversense_rect = fullseekbar_rect.expand(10.);
            let seekbar_hovered = ui.rect_contains_pointer(seekbar_hoversense_rect);
            let seekbar_hover_anim_frac = ui.ctx().animate_bool_with_time(
                playback_response.id.with("seekbar_hover_anim"),
                seekbar_hovered || currently_seeking,
                0.2,
            );

            if seekbar_hover_anim_frac > 0. {
                let new_top = fullseekbar_rect.top() - (3. * seekbar_hover_anim_frac);
                fullseekbar_rect.set_top(new_top);
                seekbar_rect.set_top(new_top);
            }

            if seekbar_hovered || currently_seeking {
                if let Some(hover_pos) = ui.ctx().input().pointer.hover_pos() {
                    let seek_frac = ((hover_pos - playback_response.rect.left_top()).x
                        - seekbar_width_offset / 2.)
                        .max(0.)
                        .min(fullseekbar_width)
                        / fullseekbar_width;
                    if ui.ctx().input().pointer.primary_down() {
                        if stopped {
                            if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
                                stream_decoder.reset(true);
                            }
                            self.spawn_timer();
                        }
                        self.set_state(PlayerState::Seeking(seek_frac));
                        seekbar_rect.set_right(
                            hover_pos
                                .x
                                .min(fullseekbar_rect.right())
                                .max(fullseekbar_rect.left()),
                        );
                    } else if ui.ctx().input().pointer.any_released() {
                        self.set_state(PlayerState::Playing)
                    }
                }
            }

            let mut shadow = Shadow::small_light();
            shadow.color = shadow.color.linear_multiply(seekbar_anim_frac);
            let shadow_mesh = shadow.tessellate(fullseekbar_rect, Rounding::none());
            let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
            let seekbar_color = Color32::RED.linear_multiply(seekbar_anim_frac);

            ui.painter().add(shadow_mesh);
            ui.painter().rect_filled(
                fullseekbar_rect,
                Rounding::none(),
                fullseekbar_color.linear_multiply(0.5),
            );
            ui.painter()
                .rect_filled(seekbar_rect, Rounding::none(), seekbar_color);

            if seekbar_hover_anim_frac > 0. {
                ui.painter().circle_filled(
                    seekbar_rect.right_center(),
                    7. * seekbar_hover_anim_frac,
                    seekbar_color,
                );
            }
            Some(seekbar_hoversense_rect)
        } else {
            None
        }
    }

    fn render_pause(
        &mut self,
        ui: &mut Ui,
        playback_response: &Response,
        ignore_rect: Option<Rect>,
    ) {
        let pointer_in_ignore_rect = if let Some(ignore_rect) = ignore_rect {
            ui.rect_contains_pointer(ignore_rect)
        } else {
            false
        };

        if !pointer_in_ignore_rect && playback_response.clicked() {
            let mut reset_stream = false;
            let mut start_stream = false;
            if let Ok(current_state) = self.player_state.lock().as_deref_mut() {
                match current_state {
                    PlayerState::Stopped => start_stream = true,
                    PlayerState::EndOfFile => reset_stream = true,
                    PlayerState::Seeking(_) => (),
                    PlayerState::Paused => *current_state = PlayerState::Playing,
                    PlayerState::Playing => *current_state = PlayerState::Paused,
                }
            }

            if reset_stream {
                if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
                    stream_decoder.reset(true)
                }
            } else if start_stream {
                self.play();
            }
        }

        let is_paused = matches!(
            self.player_state
                .try_lock()
                .and_then(|s| Ok(*s == PlayerState::Paused)),
            Ok(true)
        );
        let pause_anim_frac = ui.ctx().animate_bool_with_time(
            playback_response.id.with("pause_anim"),
            is_paused,
            0.2,
        );

        let pause_symbol_opacity = 0.5;
        let pause_circle_radius = 50. - 10. * (1. - pause_anim_frac);
        let pause_icon_center = playback_response.rect.center();
        let pause_circle_color = Color32::BLACK
            .linear_multiply(pause_symbol_opacity)
            .linear_multiply(pause_anim_frac);
        let pause_text_color = Color32::WHITE
            .linear_multiply(pause_symbol_opacity)
            .linear_multiply(pause_anim_frac);
        let mut pause_text_font_id = FontId::default();
        pause_text_font_id.size = pause_circle_radius * 0.7;
        if pause_anim_frac > 0. {
            ui.painter().circle(
                pause_icon_center,
                pause_circle_radius,
                pause_circle_color,
                Stroke::none(),
            );
            ui.painter().text(
                pause_icon_center,
                Align2::CENTER_CENTER,
                "⏸",
                pause_text_font_id,
                pause_text_color,
            );
        }
    }

    fn new_from_bytes(ctx: &egui::Context, input_bytes: &[u8]) -> Result<Self> {
        let mut file = tempfile::Builder::new().tempfile()?;
        file.write_all(input_bytes)?;
        let path = file.path().to_string_lossy().to_string();
        let mut slf = Self::new(ctx, &path)?;
        slf.temp_file = Some(file);
        Ok(slf)
    }

    fn new(ctx: &egui::Context, input_path: &String) -> Result<Self> {
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

        let player_state = Arc::new(Mutex::new(PlayerState::Stopped));
        let total_duration =
            Duration::milliseconds((input_context.duration() as f64 * 1e-3) as i64);
        let duration_ms = timestamp_to_millisec(input_context.duration(), AV_TIME_BASE_RATIONAL); // in sec
        
        dbg!(video_stream.duration(), duration_ms);
        let stream_decoder = StreamDecoder {
            decoder: video_decoder,
            video_stream_index,
            frame_index: 0,
            current_ts: 0,
            input_context,
            player_state: Arc::clone(&player_state),
            scaler: frame_scaler,
        };

        let mut streamer = Self {
            stream_decoder: Arc::new(Mutex::new(stream_decoder)),
            texture_fiter: TextureFilter::Linear,
            framerate,
            frame_timer: Timer::new(),
            frame_thread: None,
            texture_handle: None,
            player_state,
            width,
            total_duration,
            duration_ms,
            elapsed_ms: 0,
            current_duration: Duration::seconds(0),
            looping: true,
            height,
            ctx_ref: ctx.clone(),
            temp_file: None,
        };

        loop {
            if let Ok(_texture_handle) = streamer.try_set_texture_handle() {
                break;
            }
        }

        Ok(streamer)
    }

    fn try_set_texture_handle(&mut self) -> Result<TextureHandle> {
        match self
            .stream_decoder
            .lock()
            .and_then(|mut s| Ok(s.recieve_next_packet_until_frame()))
        {
            Ok(Ok(first_frame)) => {
                let texture_handle =
                    self.ctx_ref
                        .load_texture("vidstream", first_frame, self.texture_fiter);
                let texture_handle_clone = texture_handle.clone();
                self.texture_handle = Some(texture_handle);
                Ok(texture_handle_clone)
            }
            Ok(Err(e)) => Err(e),
            _ => unreachable!(),
        }
    }
}

impl StreamDecoder {
    fn recieve_next_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.input_context.packets().next() {
            if stream.index() == self.video_stream_index {
                self.decoder.send_packet(&packet)?;
                if let Some(dts) = packet.dts() {
                    self.current_ts = timestamp_to_millisec(dts, stream.time_base());
                }
            }
        } else {
            self.decoder.send_eof()?;
            let mut state = self.player_state.lock().unwrap();
            *state = PlayerState::EndOfFile;
        }
        Ok(())
    }

    fn drop_frames(&mut self) {
        let mut decoded_frame = Video::empty();
        if self.decoder.receive_frame(&mut decoded_frame).is_err() {
            let _ = self.recieve_next_packet();
        } else {
            self.drop_frames();
        }
    }

    fn reset(&mut self, start_playing: bool) {
        let beginning: i64 = 0;
        let beginning_seek = beginning.rescale((1, 1), rescale::TIME_BASE);
        let _ = self.input_context.seek(beginning_seek, ..beginning_seek);
        self.decoder.flush();
        self.frame_index = 0;
        if start_playing {
            let mut state = self.player_state.lock().unwrap();
            *state = PlayerState::Playing;
        }
    }

    fn recieve_next_packet_until_frame(&mut self) -> Result<ColorImage> {
        if let Ok(color_image) = self.recieve_next_frame() {
            Ok(color_image)
        } else {
            self.recieve_next_packet()?;
            self.recieve_next_packet_until_frame()
        }
    }

    fn recieve_next_frame(&mut self) -> Result<ColorImage> {
        let mut decoded_frame = Video::empty();
        // self.decoder.decoder().video().unwrap().
        match self.decoder.receive_frame(&mut decoded_frame) {
            Ok(()) => {
                let mut rgb_frame = Video::empty();
                self.scaler.run(&decoded_frame, &mut rgb_frame)?;

                let image = video_frame_to_image(rgb_frame);
                self.frame_index += 1;

                Ok(image)
            }
            Err(e) => {
                return Err(e.into());
            }
        }
    }
}

fn video_frame_to_image(frame: Video) -> ColorImage {
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

fn main() {
    ffmpeg::init().unwrap();
    eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Box::new(App::default())),
    )
}
