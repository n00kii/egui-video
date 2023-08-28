#![warn(missing_docs)]
//! egui-video
//! video playback library for [`egui`]
//!
//! # Example
//!
//! This example can also be found in the `examples` directory.
//!
//! ```rust
//! use eframe::NativeOptions;
//! use egui::{CentralPanel, DragValue, Grid, Sense, Slider, TextEdit, Window};
//! use egui_video::{AudioDevice, Player};
//! fn main() {
//!     let _ = eframe::run_native(
//!         "app",
//!         NativeOptions::default(),
//!         Box::new(|_| Box::new(App::default())),
//!     );
//! }
//! struct App {
//!     audio_device: AudioDevice,
//!     player: Option<Player>,
//!
//!     media_path: String,
//!     stream_size_scale: f32,
//!     seek_frac: f32,
//! }
//!
//! impl Default for App {
//!     fn default() -> Self {
//!         Self {
//!             audio_device: egui_video::init_audio_device(&sdl2::init().unwrap().audio().unwrap())
//!                 .unwrap(),
//!             media_path: String::new(),
//!             stream_size_scale: 1.,
//!             seek_frac: 0.,
//!             player: None,
//!         }
//!     }
//! }
//!
//! impl eframe::App for App {
//!     fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
//!         ctx.request_repaint();
//!         CentralPanel::default().show(ctx, |ui| {
//!             ui.horizontal(|ui| {
//!                 ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
//!                     if ui.button("load").clicked() {
//!                         match Player::new(ctx, &self.media_path.replace("\"", ""))
//!                             .and_then(|p| p.with_audio(&mut self.audio_device))
//!                         {
//!                             Ok(player) => {
//!                                 self.player = Some(player);
//!                             }
//!                             Err(e) => println!("failed to make stream: {e}"),
//!                         }
//!                     }
//!                 });
//!                 ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
//!                     if ui.button("clear").clicked() {
//!                         self.player = None;
//!                     }
//!                 });
//!
//!                 let tedit_resp = ui.add_sized(
//!                     [ui.available_width(), ui.available_height()],
//!                     TextEdit::singleline(&mut self.media_path)
//!                         .hint_text("click to set path")
//!                         .interactive(false),
//!                 );
//!
//!                 if ui
//!                     .interact(
//!                         tedit_resp.rect,
//!                         tedit_resp.id.with("click_sense"),
//!                         Sense::click(),
//!                     )
//!                     .clicked()
//!                 {
//!                     if let Some(path_buf) = rfd::FileDialog::new()
//!                         .add_filter("videos", &["mp4", "gif", "webm"])
//!                         .pick_file()
//!                     {
//!                         self.media_path = path_buf.as_path().to_string_lossy().to_string();
//!                     }
//!                 }
//!             });
//!             ui.separator();
//!             if let Some(player) = self.player.as_mut() {
//!                 Window::new("info").show(ctx, |ui| {
//!                     Grid::new("info_grid").show(ui, |ui| {
//!                         ui.label("frame rate");
//!                         ui.label(player.framerate.to_string());
//!                         ui.end_row();
//!
//!                         ui.label("size");
//!                         ui.label(format!("{}x{}", player.width, player.height));
//!                         ui.end_row();
//!
//!                         ui.label("elapsed / duration");
//!                         ui.label(player.duration_text());
//!                         ui.end_row();
//!
//!                         ui.label("state");
//!                         ui.label(format!("{:?}", player.player_state.get()));
//!                         ui.end_row();
//!
//!                         ui.label("has audio?");
//!                         ui.label(player.audio_streamer.is_some().to_string());
//!                         ui.end_row();
//!                     });
//!                 });
//!                 Window::new("controls").show(ctx, |ui| {
//!                     ui.horizontal(|ui| {
//!                         if ui.button("seek to:").clicked() {
//!                             player.seek(self.seek_frac);
//!                         }
//!                         ui.add(
//!                             DragValue::new(&mut self.seek_frac)
//!                                 .speed(0.05)
//!                                 .clamp_range(0.0..=1.0),
//!                         );
//!                         ui.checkbox(&mut player.looping, "loop");
//!                     });
//!                     ui.horizontal(|ui| {
//!                         ui.label("size scale");
//!                         ui.add(Slider::new(&mut self.stream_size_scale, 0.0..=2.));
//!                     });
//!                     ui.separator();
//!                     ui.horizontal(|ui| {
//!                         if ui.button("play").clicked() {
//!                             player.start()
//!                         }
//!                         if ui.button("unpause").clicked() {
//!                             player.resume();
//!                         }
//!                         if ui.button("pause").clicked() {
//!                             player.pause();
//!                         }
//!                         if ui.button("stop").clicked() {
//!                             player.stop();
//!                         }
//!                     });
//!                     ui.horizontal(|ui| {
//!                         ui.label("volume");
//!                         let mut volume = player.audio_volume.get();
//!                         if ui
//!                             .add(Slider::new(&mut volume, 0.0..=player.max_audio_volume))
//!                             .changed()
//!                         {
//!                             player.audio_volume.set(volume);
//!                         };
//!                     });
//!                 });
//!
//!                 player.ui(
//!                     ui,
//!                     [
//!                         player.width as f32 * self.stream_size_scale,
//!                         player.height as f32 * self.stream_size_scale,
//!                     ],
//!                 );
//!             }
//!         });
//!     }
//! }
//! ```

extern crate ffmpeg_the_third as ffmpeg;
use anyhow::Result;
use atomic::Atomic;
use chrono::{DateTime, Duration, Utc};
use egui::epaint::Shadow;
use egui::{
    vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Rounding, Sense, Spinner,
    TextureHandle, TextureOptions, Ui, Vec2,
};
use ffmpeg::ffi::AV_TIME_BASE;
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Audio;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{rescale, Rational, Rescale};
use ffmpeg::{software, ChannelLayout};
use parking_lot::Mutex;
use ringbuf::SharedRb;
use sdl2::audio::{self, AudioCallback, AudioFormat, AudioSpecDesired};
use std::sync::{Arc, Weak};
use std::time::UNIX_EPOCH;
use timer::{Guard, Timer};

#[cfg(feature = "from_bytes")]
use tempfile::NamedTempFile;

#[cfg(feature = "from_bytes")]
use std::io::Write;

fn format_duration(dur: Duration) -> String {
    let dt = DateTime::<Utc>::from(UNIX_EPOCH) + dur;
    if dt.format("%H").to_string().parse::<i64>().unwrap() > 0 {
        dt.format("%H:%M:%S").to_string()
    } else {
        dt.format("%M:%S").to_string()
    }
}

/// The playback device. Needs to be initialized (and kept alive!) for use by a [`Player`].
pub type AudioDevice = audio::AudioDevice<AudioDeviceCallback>;

type ApplyVideoFrameFn = Box<dyn FnMut(ColorImage) + Send>;

type RingbufProducer<T> = ringbuf::Producer<T, Arc<SharedRb<T, Vec<std::mem::MaybeUninit<T>>>>>;
type RingbufConsumer<T> = ringbuf::Consumer<T, Arc<SharedRb<T, Vec<std::mem::MaybeUninit<T>>>>>;

type AudioSampleProducer = RingbufProducer<f32>;
type AudioSampleConsumer = RingbufConsumer<f32>;

/// The [`Player`] processes and controls streams of video/audio. This is what you use to show a video file.
/// Initialize once, and use the [`Player::ui`] or [`Player::ui_at()`] functions to show the playback.
pub struct Player {
    /// The video streamer of the player.
    pub video_streamer: Arc<Mutex<VideoStreamer>>,
    /// The audio streamer of the player. Won't exist unless [`Player::with_audio`] is called and there exists
    /// a valid audio stream in the file.
    pub audio_streamer: Option<Arc<Mutex<AudioStreamer>>>,
    /// The state of the player.
    pub player_state: Shared<PlayerState>,
    /// The framerate of the video stream.
    pub framerate: f64,
    texture_options: TextureOptions,
    /// The player's texture handle.
    pub texture_handle: TextureHandle,
    /// The height of the video stream.
    pub height: u32,
    /// The width of the video stream.
    pub width: u32,
    frame_timer: Timer,
    audio_timer: Timer,
    audio_thread: Option<Guard>,
    frame_thread: Option<Guard>,
    ctx_ref: egui::Context,
    /// Should the stream loop if it finishes?
    pub looping: bool,
    /// The volume of the audio stream.
    pub audio_volume: Shared<f32>,
    /// The maximum volume of the audio stream.
    pub max_audio_volume: f32,
    duration_ms: i64,
    last_seek_ms: Option<i64>,
    preseek_player_state: Option<PlayerState>,
    #[cfg(feature = "from_bytes")]
    temp_file: Option<NamedTempFile>,
    video_elapsed_ms: Shared<i64>,
    audio_elapsed_ms: Shared<i64>,
    video_elapsed_ms_override: Option<i64>,
    input_path: String,
}

#[derive(PartialEq, Clone, Copy, Debug)]
/// The possible states of a [`Player`].
pub enum PlayerState {
    /// No playback.
    Stopped,
    /// Streams have reached the end of the file.
    EndOfFile,
    /// Stream is seeking. Inner bool represents whether or not the seek is completed.
    Seeking(bool),
    /// Playback is paused.
    Paused,
    /// Playback is ongoing.
    Playing,
    /// Playback is scheduled to restart.
    Restarting,
}

/// Streams video.
pub struct VideoStreamer {
    video_decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    player_state: Shared<PlayerState>,
    duration_ms: i64,
    input_context: Input,
    video_elapsed_ms: Shared<i64>,
    _audio_elapsed_ms: Shared<i64>,
    //scaler: software::scaling::Context,
    apply_video_frame_fn: Option<ApplyVideoFrameFn>,
}

/// Streams audio.
pub struct AudioStreamer {
    _video_elapsed_ms: Shared<i64>,
    audio_elapsed_ms: Shared<i64>,
    audio_stream_index: usize,
    duration_ms: i64,
    audio_decoder: ffmpeg::decoder::Audio,
    resampler: software::resampling::Context,
    audio_sample_producer: AudioSampleProducer,
    input_context: Input,
    player_state: Shared<PlayerState>,
}

#[derive(Clone)]
/// Just `Arc<Mutex<T>>` with a local cache.
pub struct Shared<T: Copy> {
    raw_value: Arc<Atomic<T>>,
}

impl<T: Copy> Shared<T> {
    /// Set the value.
    pub fn set(&self, value: T) {
        self.raw_value.store(value, atomic::Ordering::Relaxed)
    }
    /// Get the value.
    pub fn get(&self) -> T {
        self.raw_value.load(atomic::Ordering::Relaxed)
    }
    /// Make a new cache.
    pub fn new(value: T) -> Self {
        Self {
            raw_value: Arc::new(Atomic::new(value)),
        }
    }
}

const AV_TIME_BASE_RATIONAL: Rational = Rational(1, AV_TIME_BASE);
const MILLISEC_TIME_BASE: Rational = Rational(1, 1000);

fn timestamp_to_millisec(timestamp: i64, time_base: Rational) -> i64 {
    timestamp.rescale(time_base, MILLISEC_TIME_BASE)
}

fn millisec_to_timestamp(millisec: i64, time_base: Rational) -> i64 {
    millisec.rescale(MILLISEC_TIME_BASE, time_base)
}

#[inline(always)]
fn millisec_approx_eq(a: i64, b: i64) -> bool {
    a.abs_diff(b) < 50
}

impl Player {
    /// A formatted string for displaying the duration of the video stream.
    pub fn duration_text(&mut self) -> String {
        format!(
            "{} / {}",
            format_duration(Duration::milliseconds(self.elapsed_ms())),
            format_duration(Duration::milliseconds(self.duration_ms))
        )
    }
    fn reset(&mut self) {
        self.last_seek_ms = None;
        self.video_elapsed_ms_override = None;
        self.video_elapsed_ms.set(0);
        self.audio_elapsed_ms.set(0);
        self.video_streamer.lock().reset();
        if let Some(audio_decoder) = self.audio_streamer.as_mut() {
            audio_decoder.lock().reset();
        }
    }
    fn elapsed_ms(&self) -> i64 {
        self.video_elapsed_ms_override
            .as_ref()
            .map(|i| *i)
            .unwrap_or(self.video_elapsed_ms.get())
    }
    fn set_state(&mut self, new_state: PlayerState) {
        self.player_state.set(new_state)
    }
    /// Pause the stream.
    pub fn pause(&mut self) {
        self.set_state(PlayerState::Paused)
    }
    /// Resume the stream from a paused state.
    pub fn resume(&mut self) {
        self.set_state(PlayerState::Playing)
    }
    /// Stop the stream.
    pub fn stop(&mut self) {
        self.set_state(PlayerState::Stopped)
    }
    /// Directly stop the stream. Use if you need to immmediately end the streams, and/or you
    /// aren't able to call the player's [`Player::ui`]/[`Player::ui_at`] functions later on.
    pub fn stop_direct(&mut self) {
        self.frame_thread = None;
        self.audio_thread = None;
        self.reset()
    }
    fn duration_frac(&mut self) -> f32 {
        self.elapsed_ms() as f32 / self.duration_ms as f32
    }
    /// Seek to a location in the stream.
    pub fn seek(&mut self, seek_frac: f32) {
        let current_state = self.player_state.get();
        if !matches!(current_state, PlayerState::Seeking(true)) {
            match current_state {
                PlayerState::Stopped | PlayerState::EndOfFile => {
                    self.preseek_player_state = Some(PlayerState::Paused);
                    self.start();
                }
                PlayerState::Paused | PlayerState::Playing => {
                    self.preseek_player_state = Some(current_state);
                }
                _ => (),
            }

            let video_streamer = self.video_streamer.clone();

            if let Some(audio_streamer) = self.audio_streamer.as_mut() {
                audio_streamer.lock().seek(seek_frac);
            };

            self.last_seek_ms = Some((seek_frac as f64 * self.duration_ms as f64) as i64);
            self.set_state(PlayerState::Seeking(true));

            std::thread::spawn(move || {
                video_streamer.lock().seek(seek_frac);
            });
        }
    }
    fn spawn_timers(&mut self) {
        let mut texture_handle = self.texture_handle.clone();
        let texture_options = self.texture_options.clone();
        let ctx = self.ctx_ref.clone();
        let wait_duration = Duration::milliseconds((1000. / self.framerate) as i64);

        fn play<T: Streamer>(streamer: &Weak<Mutex<T>>) {
            if let Some(streamer) = streamer.upgrade() {
                if let Some(mut streamer) = streamer.try_lock() {
                    if streamer.player_state().get() == PlayerState::Playing {
                        match streamer.recieve_next_packet_until_frame() {
                            Ok(frame) => streamer.apply_frame(frame),
                            Err(e) => {
                                if is_ffmpeg_eof_error(&e) && streamer.is_primary_streamer() {
                                    streamer.player_state().set(PlayerState::EndOfFile)
                                }
                            }
                        }
                    }
                }
            }
        }

        self.video_streamer.lock().apply_video_frame_fn = Some(Box::new(move |frame| {
            texture_handle.set(frame, texture_options)
        }));

        let video_streamer_ref = Arc::downgrade(&self.video_streamer);

        let frame_timer_guard = self.frame_timer.schedule_repeating(wait_duration, move || {
            play(&video_streamer_ref);
            ctx.request_repaint();
        });

        self.frame_thread = Some(frame_timer_guard);

        if let Some(audio_decoder) = self.audio_streamer.as_ref() {
            let audio_decoder_ref = Arc::downgrade(&audio_decoder);
            let audio_timer_guard = self
                .audio_timer
                .schedule_repeating(Duration::zero(), move || play(&audio_decoder_ref));
            self.audio_thread = Some(audio_timer_guard);
        }
    }
    /// Start the stream.
    pub fn start(&mut self) {
        self.stop_direct();
        self.spawn_timers();
        self.resume();
    }
    fn process_state(&mut self) {
        let mut reset_stream = false;

        match self.player_state.get() {
            PlayerState::EndOfFile => {
                if self.looping {
                    reset_stream = true;
                } else {
                    self.player_state.set(PlayerState::Stopped);
                }
            }
            PlayerState::Stopped => {
                self.stop_direct();
            }
            PlayerState::Seeking(seek_in_progress) => {
                if self.last_seek_ms.is_some() {
                    // let video_elapsed_ms = self.video_elapsed_ms.get();
                    let last_seek_ms = *self.last_seek_ms.as_ref().unwrap();
                    // if (millisec_approx_eq(video_elapsed_ms, last_seek_ms) || video_elapsed_ms == 0)
                    if !seek_in_progress {
                        if let Some(previeous_player_state) = self.preseek_player_state {
                            self.set_state(previeous_player_state)
                        }
                        self.video_elapsed_ms_override = None;
                        self.last_seek_ms = None;
                    } else {
                        self.video_elapsed_ms_override = Some(last_seek_ms);
                    }
                } else {
                    self.video_elapsed_ms_override = None;
                }
            }
            PlayerState::Restarting => reset_stream = true,
            _ => (),
        }

        if reset_stream {
            self.reset();
            self.resume();
        }
    }

    /// Draw the player's ui and process state changes.
    pub fn ui(&mut self, ui: &mut Ui, size: [f32; 2]) -> egui::Response {
        let image = Image::new(self.texture_handle.id(), size).sense(Sense::click());
        let response = ui.add(image);
        self.render_ui(ui, &response);
        self.process_state();
        response
    }

    /// Draw the player's ui with a specific rect, and process state changes.
    pub fn ui_at(&mut self, ui: &mut Ui, rect: Rect) -> egui::Response {
        let image = Image::new(self.texture_handle.id(), rect.size()).sense(Sense::click());
        let response = ui.put(rect, image);
        self.render_ui(ui, &response);
        self.process_state();
        response
    }

    fn render_ui(&mut self, ui: &mut Ui, playback_response: &Response) -> Option<Rect> {
        let hovered = ui.rect_contains_pointer(playback_response.rect);
        let currently_seeking = matches!(self.player_state.get(), PlayerState::Seeking(_));
        let is_stopped = matches!(self.player_state.get(), PlayerState::Stopped);
        let is_paused = matches!(self.player_state.get(), PlayerState::Paused);
        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            playback_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            0.2,
        );

        if seekbar_anim_frac > 0. {
            let seekbar_width_offset = 20.;
            let fullseekbar_width = playback_response.rect.width() - seekbar_width_offset;

            let seekbar_width = fullseekbar_width * self.duration_frac();

            let seekbar_offset = 20.;
            let seekbar_pos = playback_response.rect.left_bottom()
                + vec2(seekbar_width_offset / 2., -seekbar_offset);
            let seekbar_height = 3.;
            let mut fullseekbar_rect =
                Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

            let mut seekbar_rect =
                Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
            let seekbar_interact_rect = fullseekbar_rect.expand(10.);
            ui.interact(seekbar_interact_rect, playback_response.id, Sense::drag());

            let seekbar_response = ui.interact(
                seekbar_interact_rect,
                playback_response.id.with("seekbar"),
                Sense::click_and_drag(),
            );

            let seekbar_hovered = seekbar_response.hovered();
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

            let seek_indicator_anim = ui.ctx().animate_bool_with_time(
                playback_response.id.with("seek_indicator_anim"),
                currently_seeking,
                0.1,
            );

            if currently_seeking {
                let mut seek_indicator_shadow = Shadow::big_dark();
                seek_indicator_shadow.color = seek_indicator_shadow
                    .color
                    .linear_multiply(seek_indicator_anim);
                let spinner_size = 20. * seek_indicator_anim;
                ui.painter().add(
                    seek_indicator_shadow.tessellate(playback_response.rect, Rounding::none()),
                );
                ui.put(
                    Rect::from_center_size(
                        playback_response.rect.center(),
                        Vec2::splat(spinner_size),
                    ),
                    Spinner::new().size(spinner_size),
                );
            }

            if seekbar_hovered || currently_seeking {
                if let Some(hover_pos) = seekbar_response.hover_pos() {
                    if seekbar_response.clicked() || seekbar_response.dragged() {
                        let seek_frac = ((hover_pos - playback_response.rect.left_top()).x
                            - seekbar_width_offset / 2.)
                            .max(0.)
                            .min(fullseekbar_width)
                            / fullseekbar_width;
                        seekbar_rect.set_right(
                            hover_pos
                                .x
                                .min(fullseekbar_rect.right())
                                .max(fullseekbar_rect.left()),
                        );
                        if is_stopped {
                            self.start()
                        }
                        self.seek(seek_frac);
                    }
                }
            }
            let text_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

            let pause_icon = if is_paused {
                "▶"
            } else if is_stopped {
                "◼"
            } else if currently_seeking {
                "↔"
            } else {
                "⏸"
            };
            let audio_volume_frac = self.audio_volume.get() / self.max_audio_volume;
            let sound_icon = if audio_volume_frac > 0.7 {
                "🔊"
            } else if audio_volume_frac > 0.4 {
                "🔉"
            } else if audio_volume_frac > 0. {
                "🔈"
            } else {
                "🔇"
            };
            let mut icon_font_id = FontId::default();
            icon_font_id.size = 16.;

            let text_y_offset = -7.;
            let sound_icon_offset = vec2(-5., text_y_offset);
            let sound_icon_pos = fullseekbar_rect.right_top() + sound_icon_offset;

            let pause_icon_offset = vec2(3., text_y_offset);
            let pause_icon_pos = fullseekbar_rect.left_top() + pause_icon_offset;

            let duration_text_offset = vec2(25., text_y_offset);
            let duration_text_pos = fullseekbar_rect.left_top() + duration_text_offset;
            let mut duration_text_font_id = FontId::default();
            duration_text_font_id.size = 14.;

            let mut shadow = Shadow::big_light();
            shadow.color = shadow.color.linear_multiply(seekbar_anim_frac);

            let mut shadow_rect = playback_response.rect;
            shadow_rect.set_top(shadow_rect.bottom() - seekbar_offset - 10.);
            let shadow_mesh = shadow.tessellate(shadow_rect, Rounding::none());

            let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
            let seekbar_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

            ui.painter().add(shadow_mesh);

            ui.painter().rect_filled(
                fullseekbar_rect,
                Rounding::none(),
                fullseekbar_color.linear_multiply(0.5),
            );
            ui.painter()
                .rect_filled(seekbar_rect, Rounding::none(), seekbar_color);
            ui.painter().text(
                pause_icon_pos,
                Align2::LEFT_BOTTOM,
                pause_icon,
                icon_font_id.clone(),
                text_color,
            );

            ui.painter().text(
                duration_text_pos,
                Align2::LEFT_BOTTOM,
                self.duration_text(),
                duration_text_font_id,
                text_color,
            );

            if seekbar_hover_anim_frac > 0. {
                ui.painter().circle_filled(
                    seekbar_rect.right_center(),
                    7. * seekbar_hover_anim_frac,
                    seekbar_color,
                );
            }

            if playback_response.clicked() {
                let mut reset_stream = false;
                let mut start_stream = false;

                match self.player_state.get() {
                    PlayerState::Stopped => start_stream = true,
                    PlayerState::EndOfFile => reset_stream = true,
                    PlayerState::Paused => self.player_state.set(PlayerState::Playing),
                    PlayerState::Playing => self.player_state.set(PlayerState::Paused),
                    _ => (),
                }

                if reset_stream {
                    self.reset();
                    self.resume();
                } else if start_stream {
                    self.start();
                }
            }

            if self.audio_streamer.is_some() {
                let sound_icon_rect = ui.painter().text(
                    sound_icon_pos,
                    Align2::RIGHT_BOTTOM,
                    sound_icon,
                    icon_font_id.clone(),
                    text_color,
                );

                if ui
                    .interact(
                        sound_icon_rect,
                        playback_response.id.with("sound_icon_sense"),
                        Sense::click(),
                    )
                    .clicked()
                {
                    if self.audio_volume.get() != 0. {
                        self.audio_volume.set(0.)
                    } else {
                        self.audio_volume.set(self.max_audio_volume / 2.)
                    }
                }

                let sound_slider_outer_height = 75.;
                let sound_slider_margin = 5.;
                let sound_slider_opacity = 100;
                let mut sound_slider_rect = sound_icon_rect;
                sound_slider_rect.set_bottom(sound_icon_rect.top() - sound_slider_margin);
                sound_slider_rect.set_top(sound_slider_rect.top() - sound_slider_outer_height);

                let sound_slider_interact_rect = sound_slider_rect.expand(sound_slider_margin);
                let sound_hovered = ui.rect_contains_pointer(sound_icon_rect);
                let sound_slider_hovered = ui.rect_contains_pointer(sound_slider_interact_rect);
                let sound_anim_id = playback_response.id.with("sound_anim");
                let mut sound_anim_frac: f32 = ui
                    .ctx()
                    .memory_mut(|m| *m.data.get_temp_mut_or_default(sound_anim_id));
                sound_anim_frac = ui.ctx().animate_bool_with_time(
                    sound_anim_id,
                    sound_hovered || (sound_slider_hovered && sound_anim_frac > 0.),
                    0.2,
                );
                ui.ctx()
                    .memory_mut(|m| m.data.insert_temp(sound_anim_id, sound_anim_frac));
                let sound_slider_bg_color = Color32::from_black_alpha(sound_slider_opacity)
                    .linear_multiply(sound_anim_frac);
                let sound_bar_color = Color32::from_white_alpha(sound_slider_opacity)
                    .linear_multiply(sound_anim_frac);
                let mut sound_bar_rect = sound_slider_rect;
                sound_bar_rect.set_top(
                    sound_bar_rect.bottom()
                        - (self.audio_volume.get() / self.max_audio_volume)
                            * sound_bar_rect.height(),
                );

                ui.painter().rect_filled(
                    sound_slider_rect,
                    Rounding::same(5.),
                    sound_slider_bg_color,
                );

                ui.painter()
                    .rect_filled(sound_bar_rect, Rounding::same(5.), sound_bar_color);
                let sound_slider_resp = ui.interact(
                    sound_slider_rect,
                    playback_response.id.with("sound_slider_sense"),
                    Sense::click_and_drag(),
                );
                if sound_anim_frac > 0. && sound_slider_resp.clicked()
                    || sound_slider_resp.dragged()
                {
                    if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                        let sound_frac = 1.
                            - ((hover_pos - sound_slider_rect.left_top()).y
                                / sound_slider_rect.height())
                            .max(0.)
                            .min(1.);
                        self.audio_volume.set(sound_frac * self.max_audio_volume);
                    }
                }
            }

            Some(seekbar_interact_rect)
        } else {
            None
        }
    }

    #[cfg(feature = "from_bytes")]
    /// Create a new [`Player`] from input bytes.
    pub fn new_from_bytes(ctx: &egui::Context, input_bytes: &[u8]) -> Result<Self> {
        let mut file = tempfile::Builder::new().tempfile()?;
        file.write_all(input_bytes)?;
        let path = file.path().to_string_lossy().to_string();
        let mut slf = Self::new(ctx, &path)?;
        slf.temp_file = Some(file);
        Ok(slf)
    }

    /// Initializes the audio stream (if there is one), required for making a [`Player`] output audio.
    /// Will stop and reset the player's state.
    pub fn set_audio(&mut self, audio_device: &mut AudioDevice) -> Result<()> {
        let audio_input_context = input(&self.input_path)?;
        let audio_stream = audio_input_context.streams().best(Type::Audio);

        let audio_streamer = if let Some(audio_stream) = audio_stream.as_ref() {
            let audio_stream_index = audio_stream.index();
            let audio_context =
                ffmpeg::codec::context::Context::from_parameters(audio_stream.parameters())?;
            let audio_decoder = audio_context.decoder().audio()?;
            let audio_sample_buffer =
                SharedRb::<f32, Vec<_>>::new(audio_device.spec().size as usize);
            let (audio_sample_producer, audio_sample_consumer) = audio_sample_buffer.split();
            let audio_resampler = ffmpeg::software::resampling::context::Context::get(
                audio_decoder.format(),
                audio_decoder.channel_layout(),
                audio_decoder.rate(),
                audio_device.spec().format.to_sample(),
                ChannelLayout::STEREO,
                audio_device.spec().freq as u32,
            )?;

            audio_device.lock().sample_streams.push(AudioSampleStream {
                sample_consumer: audio_sample_consumer,
                audio_volume: self.audio_volume.clone(),
            });

            audio_device.resume();

            self.stop_direct();

            Some(AudioStreamer {
                duration_ms: self.duration_ms,
                player_state: self.player_state.clone(),
                _video_elapsed_ms: self.video_elapsed_ms.clone(),
                audio_elapsed_ms: self.audio_elapsed_ms.clone(),
                audio_sample_producer,
                input_context: audio_input_context,
                audio_decoder,
                audio_stream_index,
                resampler: audio_resampler,
            })
        } else {
            None
        };
        self.audio_streamer = audio_streamer.map(|s| Arc::new(Mutex::new(s)));
        Ok(())
    }

    /// Enables using [`Player::set_audio`] with the builder pattern.
    pub fn with_audio(mut self, audio_device: &mut AudioDevice) -> Result<Self> {
        self.set_audio(audio_device)?;
        Ok(self)
    }

    /// Create a new [`Player`].
    pub fn new(ctx: &egui::Context, input_path: &String) -> Result<Self> {
        let input_context = input(&input_path)?;
        let video_stream = input_context
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = video_stream.index();
        let max_audio_volume = 1.;

        let audio_volume = Shared::new(max_audio_volume / 2.);

        let video_elapsed_ms = Shared::new(0);
        let audio_elapsed_ms = Shared::new(0);
        let player_state = Shared::new(PlayerState::Stopped);

        let video_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let video_decoder = video_context.decoder().video()?;
        let framerate = (video_stream.avg_frame_rate().numerator() as f64)
            / video_stream.avg_frame_rate().denominator() as f64;

        let (width, height) = (video_decoder.width(), video_decoder.height());
        let duration_ms = timestamp_to_millisec(input_context.duration(), AV_TIME_BASE_RATIONAL); // in sec

        let stream_decoder = VideoStreamer {
            apply_video_frame_fn: None,
            duration_ms,
            video_decoder,
            video_stream_index,
            _audio_elapsed_ms: audio_elapsed_ms.clone(),
            video_elapsed_ms: video_elapsed_ms.clone(),
            input_context,
            player_state: player_state.clone(),
            //scaler: frame_scaler,
        };
        let texture_options = TextureOptions::LINEAR;
        let texture_handle = ctx.load_texture("vidstream", ColorImage::example(), texture_options);
        let mut streamer = Self {
            input_path: input_path.clone(),
            audio_streamer: None,
            video_streamer: Arc::new(Mutex::new(stream_decoder)),
            texture_options,
            framerate,
            frame_timer: Timer::new(),
            audio_timer: Timer::new(),
            preseek_player_state: None,
            frame_thread: None,
            audio_thread: None,
            texture_handle,
            player_state,
            video_elapsed_ms,
            audio_elapsed_ms,
            width,
            last_seek_ms: None,
            duration_ms,
            audio_volume,
            max_audio_volume,
            video_elapsed_ms_override: None,
            looping: true,
            height,
            ctx_ref: ctx.clone(),
            #[cfg(feature = "from_bytes")]
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
        match self.video_streamer.lock().recieve_next_packet_until_frame() {
            Ok(first_frame) => {
                let texture_handle =
                    self.ctx_ref
                        .load_texture("vidstream", first_frame, self.texture_options);
                let texture_handle_clone = texture_handle.clone();
                self.texture_handle = texture_handle;
                Ok(texture_handle_clone)
            }
            Err(e) => Err(e),
        }
    }
}

/// Streams data.
pub trait Streamer: Send {
    /// The associated type of frame used for the stream.
    type Frame;
    /// The associated type after the frame is processed.
    type ProcessedFrame;
    /// Seek to a location within the stream.
    fn seek(&mut self, seek_frac: f32) {
        let target_ms = (seek_frac as f64 * self.duration_ms() as f64) as i64;
        let seek_completed = millisec_approx_eq(target_ms, self.elapsed_ms().get());

        // stop seeking near target so we dont waste cpu cycles
        if !seek_completed {
            let elapsed_ms = self.elapsed_ms().clone();
            let currently_behind_target = || elapsed_ms.get() < target_ms;

            let seeking_backwards = target_ms < self.elapsed_ms().get();
            let target_ts = millisec_to_timestamp(target_ms, rescale::TIME_BASE);
            // let player_state = self.player_state().clone();
            // let still_seeking = || matches!(player_state.get(), PlayerState::Seeking(_));

            if let Err(_) = self.input_context().seek(target_ts, ..target_ts) {
                // dbg!(e); TODO: propogate error
            } else if seek_frac < 0.03 {
                // prevent seek inaccuracy errors near start of stream
                self.player_state().set(PlayerState::Restarting);
                return;
            } else if seek_frac >= 1.0 {
                // disable this safeguard for now (fixed?)
                // prevent inifinite loop near end of stream
                self.player_state().set(PlayerState::EndOfFile);
                return;
            } else {
                // this drop frame loop lets us refresh until current_ts is accurate
                if seeking_backwards {
                    while !currently_behind_target() {
                        if let Err(e) = self.drop_frames() {
                            if is_ffmpeg_eof_error(&e) {
                                break;
                            }
                        }
                    }
                }

                // // this drop frame loop drops frames until we are at desired
                while currently_behind_target() {
                    if let Err(e) = self.drop_frames() {
                        if is_ffmpeg_eof_error(&e) {
                            break;
                        }
                    }
                }

                // frame preview
                if self.is_primary_streamer() {
                    match self.recieve_next_packet_until_frame() {
                        Ok(frame) => self.apply_frame(frame),
                        _ => (),
                    }
                }
            }
        }
        if self.is_primary_streamer() {
            self.player_state().set(PlayerState::Seeking(false));
        }
    }

    /// The primary streamer will control most of the state/syncing.
    fn is_primary_streamer(&self) -> bool;

    /// The stream index.
    fn stream_index(&self) -> usize;
    /// The elapsed time, in milliseconds.
    fn elapsed_ms(&mut self) -> &mut Shared<i64>;
    /// The total duration of the stream, in milliseconds.
    fn duration_ms(&mut self) -> i64;
    /// The streamer's decoder.
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened;
    /// The streamer's input context.
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input;
    /// The streamer's state.
    fn player_state(&self) -> &Shared<PlayerState>;
    /// Output a frame from the decoder.
    fn decode_frame(&mut self) -> Result<Self::Frame>;
    /// Ignore the remainder of this packet.
    fn drop_frames(&mut self) -> Result<()> {
        if self.decode_frame().is_err() {
            self.recieve_next_packet()
        } else {
            self.drop_frames()
        }
    }
    /// Recieve the next packet of the stream.
    fn recieve_next_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.input_context().packets().next() {
            let time_base = stream.time_base();
            if stream.index() == self.stream_index() {
                self.decoder().send_packet(&packet)?;
                if let Some(dts) = packet.dts() {
                    self.elapsed_ms().set(timestamp_to_millisec(dts, time_base));
                }
            }
        } else {
            self.decoder().send_eof()?;
            // self.player_state().set(PlayerState::EndOfFile);
        }
        Ok(())
    }
    /// Reset the stream to its initial state.
    fn reset(&mut self) {
        let beginning: i64 = 0;
        let beginning_seek = beginning.rescale((1, 1), rescale::TIME_BASE);
        let _ = self.input_context().seek(beginning_seek, ..beginning_seek);
        self.decoder().flush();
    }
    /// Keep recieving packets until a frame can be decoded.
    fn recieve_next_packet_until_frame(&mut self) -> Result<Self::ProcessedFrame> {
        match self.recieve_next_frame() {
            Ok(frame_result) => Ok(frame_result),
            Err(e) => {
                if is_ffmpeg_eof_error(&e) {
                    Err(e)
                } else {
                    self.recieve_next_packet()?;
                    self.recieve_next_packet_until_frame()
                }
            }
        }
    }
    /// Process a decoded frame.
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame>;
    /// Apply a processed frame
    fn apply_frame(&mut self, _frame: Self::ProcessedFrame) {}
    /// Decode and process a frame.
    fn recieve_next_frame(&mut self) -> Result<Self::ProcessedFrame> {
        match self.decode_frame() {
            Ok(decoded_frame) => self.process_frame(decoded_frame),
            Err(e) => {
                return Err(e.into());
            }
        }
    }
}

impl Streamer for VideoStreamer {
    type Frame = Video;
    type ProcessedFrame = ColorImage;
    fn is_primary_streamer(&self) -> bool {
        true
    }
    fn stream_index(&self) -> usize {
        self.video_stream_index
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.video_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&mut self) -> &mut Shared<i64> {
        &mut self.video_elapsed_ms
    }
    fn duration_ms(&mut self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        let mut decoded_frame = Video::empty();
        self.video_decoder.receive_frame(&mut decoded_frame)?;
        Ok(decoded_frame)
    }
    fn apply_frame(&mut self, frame: Self::ProcessedFrame) {
        if let Some(apply_video_frame_fn) = self.apply_video_frame_fn.as_mut() {
            apply_video_frame_fn(frame)
        }
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        let mut rgb_frame = Video::empty();
        let mut scaler = Context::get(
            frame.format(),
            frame.width(),
            frame.height(),
            Pixel::RGB24,
            frame.width(),
            frame.height(),
            Flags::BILINEAR,
        )?;
        scaler.run(&frame, &mut rgb_frame)?;

        let image = video_frame_to_image(rgb_frame);
        Ok(image)
    }
}

impl Streamer for AudioStreamer {
    type Frame = Audio;
    type ProcessedFrame = ();
    fn is_primary_streamer(&self) -> bool {
        false
    }
    fn stream_index(&self) -> usize {
        self.audio_stream_index
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.audio_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&mut self) -> &mut Shared<i64> {
        &mut self.audio_elapsed_ms
    }
    fn duration_ms(&mut self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        let mut decoded_frame = Audio::empty();
        self.audio_decoder.receive_frame(&mut decoded_frame)?;
        Ok(decoded_frame)
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        self.resampler.run(&frame, &mut resampled_frame)?;
        let audio_samples = if resampled_frame.is_packed() {
            packed(&resampled_frame)
        } else {
            resampled_frame.plane(0)
        };
        while self.audio_sample_producer.free_len() < audio_samples.len() {
            // std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.audio_sample_producer.push_slice(audio_samples);
        Ok(())
    }
}

type FfmpegAudioFormat = ffmpeg::format::Sample;
type FfmpegAudioFormatType = ffmpeg::format::sample::Type;
trait AsFfmpegSample {
    fn to_sample(&self) -> ffmpeg::format::Sample;
}

impl AsFfmpegSample for AudioFormat {
    fn to_sample(&self) -> FfmpegAudioFormat {
        match self {
            AudioFormat::U8 => FfmpegAudioFormat::U8(FfmpegAudioFormatType::Packed),
            AudioFormat::S8 => panic!("unsupported audio format"),
            AudioFormat::U16LSB => panic!("unsupported audio format"),
            AudioFormat::U16MSB => panic!("unsupported audio format"),
            AudioFormat::S16LSB => FfmpegAudioFormat::I16(FfmpegAudioFormatType::Packed),
            AudioFormat::S16MSB => FfmpegAudioFormat::I16(FfmpegAudioFormatType::Packed),
            AudioFormat::S32LSB => FfmpegAudioFormat::I32(FfmpegAudioFormatType::Packed),
            AudioFormat::S32MSB => FfmpegAudioFormat::I32(FfmpegAudioFormatType::Packed),
            AudioFormat::F32LSB => FfmpegAudioFormat::F32(FfmpegAudioFormatType::Packed),
            AudioFormat::F32MSB => FfmpegAudioFormat::F32(FfmpegAudioFormatType::Packed),
        }
    }
}

/// Create a new [`AudioDeviceCallback`]. Required for using audio.
pub fn init_audio_device(audio_sys: &sdl2::AudioSubsystem) -> Result<AudioDevice, String> {
    AudioDeviceCallback::init(audio_sys)
}

/// Pipes audio samples to SDL2.
pub struct AudioDeviceCallback {
    sample_streams: Vec<AudioSampleStream>,
}

struct AudioSampleStream {
    sample_consumer: AudioSampleConsumer,
    audio_volume: Shared<f32>,
}

impl AudioCallback for AudioDeviceCallback {
    type Channel = f32;
    fn callback(&mut self, output: &mut [Self::Channel]) {
        for x in output.iter_mut() {
            *x = self
                .sample_streams
                .iter_mut()
                .map(|s| s.sample_consumer.pop().unwrap_or(0.) * s.audio_volume.get())
                .sum()
        }
    }
}

impl AudioDeviceCallback {
    fn init(audio_sys: &sdl2::AudioSubsystem) -> Result<AudioDevice, String> {
        let audio_spec = AudioSpecDesired {
            freq: Some(44_100),
            channels: Some(2),
            samples: None,
        };
        let device = audio_sys.open_playback(None, &audio_spec, |_spec| AudioDeviceCallback {
            sample_streams: vec![],
        })?;
        Ok(device)
    }
}

#[inline]
// Thanks https://github.com/zmwangx/rust-ffmpeg/issues/72 <3
// Interpret the audio frame's data as packed (alternating channels, 12121212, as opposed to planar 11112222)
fn packed<T: ffmpeg::frame::audio::Sample>(frame: &ffmpeg::frame::Audio) -> &[T] {
    if !frame.is_packed() {
        panic!("data is not packed");
    }

    if !<T as ffmpeg::frame::audio::Sample>::is_valid(frame.format(), frame.channels()) {
        panic!("unsupported type");
    }

    unsafe {
        std::slice::from_raw_parts(
            (*frame.as_ptr()).data[0] as *const T,
            frame.samples() * frame.channels() as usize,
        )
    }
}

fn is_ffmpeg_eof_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<ffmpeg::Error>(),
        Some(ffmpeg::Error::Eof)
    )
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
    ColorImage { size, pixels }
}
