#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```
extern crate ffmpeg_the_third as ffmpeg;
use anyhow::Result;
use atomic::Atomic;
use chrono::{DateTime, Duration, Utc};
use egui::emath::RectTransform;
use egui::epaint::Shadow;
use egui::load::SizedTexture;

use egui::{
    vec2, Align2, Color32, ColorImage, FontId, Image, Pos2, Rect, Response, Rounding, Sense,
    Spinner, TextureHandle, TextureOptions, Ui, Vec2,
};
use ffmpeg::error::EAGAIN;
use ffmpeg::ffi::{AVERROR, AV_TIME_BASE};
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Audio;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{rescale, Packet, Rational, Rescale};
use ffmpeg::{software, ChannelLayout};
use parking_lot::Mutex;
use ringbuf::SharedRb;
use sdl2::audio::{self, AudioCallback, AudioFormat, AudioSpecDesired};
use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::UNIX_EPOCH;
use subtitle::Subtitle;
use timer::{Guard, Timer};

mod subtitle;

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
type SubtitleQueue = Arc<Mutex<VecDeque<Subtitle>>>;
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
    /// The subtitle streamer of the player. Won't exist unless [`Player::with_subtitles`] is called and there exists
    /// a valid subtitle stream in the file.
    pub subtitle_streamer: Option<Arc<Mutex<SubtitleStreamer>>>,
    /// The state of the player.
    pub player_state: Shared<PlayerState>,
    /// The framerate of the video stream.
    pub framerate: f64,
    texture_options: TextureOptions,
    /// The player's texture handle.
    pub texture_handle: TextureHandle,
    /// The size of the video stream.
    pub size: Vec2,
    video_timer: Timer,
    audio_timer: Timer,
    subtitle_timer: Timer,
    audio_thread: Option<Guard>,
    video_thread: Option<Guard>,
    subtitle_thread: Option<Guard>,
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
    subtitle_elapsed_ms: Shared<i64>,
    video_elapsed_ms_override: Option<i64>,
    subtitles_queue: SubtitleQueue, // text, end_display_time_ms
    current_subtitles: Vec<Subtitle>,
    input_path: String,
}

#[derive(PartialEq, Clone, Copy, Debug)]
/// The possible states of a [`Player`].
pub enum PlayerState {
    /// No playback.
    Stopped,
    /// Streams have reached the end of the file.
    EndOfFile,
    /// Stream is seeking. Inner bool represents whether or not the seek is currently in progress.
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
    video_elapsed_ms: Shared<i64>,
    audio_elapsed_ms: Shared<i64>,
    audio_stream_index: usize,
    duration_ms: i64,
    audio_decoder: ffmpeg::decoder::Audio,
    resampler: software::resampling::Context,
    audio_sample_producer: AudioSampleProducer,
    input_context: Input,
    player_state: Shared<PlayerState>,
    audio_stream_ids: Vec<usize>,
}

impl AudioStreamer {
    fn next_stream(&mut self) {
        for s in self.audio_stream_ids.iter() {
            if s == &self.audio_stream_index {
                self.audio_stream_index = self.audio_stream_ids[(self
                    .audio_stream_ids
                    .iter()
                    .position(|&s| s == self.audio_stream_index)
                    .unwrap()
                    + 1)
                    % self.audio_stream_ids.len()];
                break;
            }
        }
    }
}

/// Streams subtitles.
pub struct SubtitleStreamer {
    video_elapsed_ms: Shared<i64>,
    _audio_elapsed_ms: Shared<i64>,
    subtitle_elapsed_ms: Shared<i64>,
    subtitle_stream_index: usize,
    duration_ms: i64,
    subtitle_decoder: ffmpeg::decoder::Subtitle,
    next_packet: Option<Packet>,
    subtitles_queue: SubtitleQueue,
    // resampler: software::resampling::Context,
    // audio_sample_producer: AudioSampleProducer,
    input_context: Input,
    player_state: Shared<PlayerState>,
}

#[derive(Clone)]
/// Simple concurrecy of primitive values.
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
        self.video_thread = None;
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
            let mut audio_streamer = self.audio_streamer.clone();
            let mut subtitle_streamer = self.subtitle_streamer.clone();
            let subtitle_queue = self.subtitles_queue.clone();

            self.last_seek_ms = Some((seek_frac as f64 * self.duration_ms as f64) as i64);
            self.set_state(PlayerState::Seeking(true));

            if let Some(audio_streamer) = audio_streamer.take() {
                std::thread::spawn(move || {
                    audio_streamer.lock().seek(seek_frac);
                });
            };
            if let Some(subtitle_streamer) = subtitle_streamer.take() {
                self.current_subtitles.clear();
                std::thread::spawn(move || {
                    subtitle_queue.lock().clear();
                    subtitle_streamer.lock().seek(seek_frac);
                });
            };
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
                    if (streamer.player_state().get() == PlayerState::Playing)
                        && streamer.primary_elapsed_ms().get() >= streamer.elapsed_ms().get()
                    {
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

        let video_timer_guard = self.video_timer.schedule_repeating(wait_duration, move || {
            play(&video_streamer_ref);
            ctx.request_repaint();
        });

        self.video_thread = Some(video_timer_guard);

        if let Some(audio_decoder) = self.audio_streamer.as_ref() {
            let audio_decoder_ref = Arc::downgrade(&audio_decoder);
            let audio_timer_guard = self
                .audio_timer
                .schedule_repeating(Duration::zero(), move || play(&audio_decoder_ref));
            self.audio_thread = Some(audio_timer_guard);
        }

        if let Some(subtitle_decoder) = self.subtitle_streamer.as_ref() {
            let subtitle_decoder_ref = Arc::downgrade(&subtitle_decoder);
            let subtitle_timer_guard = self
                .subtitle_timer
                .schedule_repeating(wait_duration, move || play(&subtitle_decoder_ref));
            self.subtitle_thread = Some(subtitle_timer_guard);
        }
    }
    /// Start the stream.
    pub fn start(&mut self) {
        self.stop_direct();
        self.spawn_timers();
        self.resume();
    }

    /// Process player state updates. This function must be called for proper function
    /// of the player. This function is already included in  [`Player::ui`] or
    /// [`Player::ui_at`].
    pub fn process_state(&mut self) {
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
            PlayerState::Playing => {
                for subtitle in self.current_subtitles.iter_mut() {
                    subtitle.remaining_duration_ms -=
                        self.ctx_ref.input(|i| (i.stable_dt * 1000.) as i64);
                }
                self.current_subtitles
                    .retain(|s| s.remaining_duration_ms > 0);
                if let Some(mut queue) = self.subtitles_queue.try_lock() {
                    if queue.len() > 1 {
                        self.current_subtitles.push(queue.pop_front().unwrap());
                    }
                }
            }
            PlayerState::Seeking(seek_in_progress) => {
                if self.last_seek_ms.is_some() {
                    let last_seek_ms = *self.last_seek_ms.as_ref().unwrap();
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

    /// Create the [`egui::Image`] for the video frame.
    pub fn generate_frame_image(&self, size: Vec2) -> Image {
        Image::new(SizedTexture::new(self.texture_handle.id(), size)).sense(Sense::click())
    }

    /// Draw the video frame with a specific rect (without controls). Make sure to call [`Player::process_state`].
    pub fn render_frame(&self, ui: &mut Ui, size: Vec2) -> Response {
        ui.add(self.generate_frame_image(size))
    }

    /// Draw the video frame (without controls). Make sure to call [`Player::process_state`].
    pub fn render_frame_at(&self, ui: &mut Ui, rect: Rect) -> Response {
        ui.put(rect, self.generate_frame_image(rect.size()))
    }

    /// Draw the video frame and player controls and process state changes.
    pub fn ui(&mut self, ui: &mut Ui, size: Vec2) -> egui::Response {
        let frame_response = self.render_frame(ui, size);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        self.process_state();
        frame_response
    }

    /// Draw the video frame and player controls with a specific rect, and process state changes.
    pub fn ui_at(&mut self, ui: &mut Ui, rect: Rect) -> egui::Response {
        let frame_response = self.render_frame_at(ui, rect);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        self.process_state();
        frame_response
    }

    /// Draw the subtitles, if any. Only works when a subtitle streamer has been already created with
    /// [`Player::add_subtitles`] or [`Player::with_subtitles`] and a valid subtitle stream exists.
    pub fn render_subtitles(&mut self, ui: &mut Ui, frame_response: &Response) {
        let original_rect_center_bottom = Pos2::new(self.size.x / 2., self.size.y);
        let mut last_bottom = self.size.y;
        for subtitle in self.current_subtitles.iter() {
            let transform = RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, self.size),
                frame_response.rect,
            );
            let text_rect = ui.painter().text(
                subtitle
                    .position
                    .map(|p| transform.transform_pos(p))
                    .unwrap_or_else(|| {
                        //TODO incorporate left/right margin
                        let mut center_bottom = original_rect_center_bottom;
                        center_bottom.y = center_bottom.y.min(last_bottom) - subtitle.margin.bottom;
                        transform.transform_pos(center_bottom)
                    }),
                subtitle.alignment,
                &subtitle.text,
                FontId::proportional(transform.transform_pos(Pos2::new(subtitle.font_size, 0.)).x),
                subtitle.primary_fill,
            );
            last_bottom = transform.inverse().transform_pos(text_rect.center_top()).y;
        }
    }

    /// Draw the player controls. Make sure to call [`Player::process_state()`]. Unless you are explicitly
    /// drawing something in between the video frames and controls, it is probably better to use
    /// [`Player::ui`] or [`Player::ui_at`].
    pub fn render_controls(&mut self, ui: &mut Ui, frame_response: &Response) {
        let hovered = ui.rect_contains_pointer(frame_response.rect);
        let currently_seeking = matches!(self.player_state.get(), PlayerState::Seeking(_));
        let is_stopped = matches!(self.player_state.get(), PlayerState::Stopped);
        let is_paused = matches!(self.player_state.get(), PlayerState::Paused);
        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            0.2,
        );

        if seekbar_anim_frac <= 0. {
            return;
        }

        let seekbar_width_offset = 20.;
        let fullseekbar_width = frame_response.rect.width() - seekbar_width_offset;

        let seekbar_width = fullseekbar_width * self.duration_frac();

        let seekbar_offset = 20.;
        let seekbar_pos =
            frame_response.rect.left_bottom() + vec2(seekbar_width_offset / 2., -seekbar_offset);
        let seekbar_height = 3.;
        let mut fullseekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

        let mut seekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
        let seekbar_interact_rect = fullseekbar_rect.expand(10.);
        ui.interact(seekbar_interact_rect, frame_response.id, Sense::drag());

        let seekbar_response = ui.interact(
            seekbar_interact_rect,
            frame_response.id.with("seekbar"),
            Sense::click_and_drag(),
        );

        let seekbar_hovered = seekbar_response.hovered();
        let seekbar_hover_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_hover_anim"),
            seekbar_hovered || currently_seeking,
            0.2,
        );

        if seekbar_hover_anim_frac > 0. {
            let new_top = fullseekbar_rect.top() - (3. * seekbar_hover_anim_frac);
            fullseekbar_rect.set_top(new_top);
            seekbar_rect.set_top(new_top);
        }

        let seek_indicator_anim = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seek_indicator_anim"),
            currently_seeking,
            0.1,
        );

        if currently_seeking {
            let mut seek_indicator_shadow = Shadow::big_dark();
            seek_indicator_shadow.color = seek_indicator_shadow
                .color
                .linear_multiply(seek_indicator_anim);
            let spinner_size = 20. * seek_indicator_anim;
            ui.painter()
                .add(seek_indicator_shadow.tessellate(frame_response.rect, Rounding::ZERO));
            ui.put(
                Rect::from_center_size(frame_response.rect.center(), Vec2::splat(spinner_size)),
                Spinner::new().size(spinner_size),
            );
        }

        if seekbar_hovered || currently_seeking {
            if let Some(hover_pos) = seekbar_response.hover_pos() {
                if seekbar_response.clicked() || seekbar_response.dragged() {
                    let seek_frac = ((hover_pos - frame_response.rect.left_top()).x
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

        let audio_index_icon = "🔁";

        let text_y_offset = -7.;
        let sound_icon_offset = vec2(-5., text_y_offset);
        let sound_icon_pos = fullseekbar_rect.right_top() + sound_icon_offset;

        let audio_index_icon_offset = vec2(-20., text_y_offset);
        let audio_index_icon_pos = fullseekbar_rect.right_top() + audio_index_icon_offset;

        let pause_icon_offset = vec2(3., text_y_offset);
        let pause_icon_pos = fullseekbar_rect.left_top() + pause_icon_offset;

        let duration_text_offset = vec2(25., text_y_offset);
        let duration_text_pos = fullseekbar_rect.left_top() + duration_text_offset;
        let mut duration_text_font_id = FontId::default();
        duration_text_font_id.size = 14.;

        let mut shadow = Shadow::big_light();
        shadow.color = shadow.color.linear_multiply(seekbar_anim_frac);

        let mut shadow_rect = frame_response.rect;
        shadow_rect.set_top(shadow_rect.bottom() - seekbar_offset - 10.);
        let shadow_mesh = shadow.tessellate(shadow_rect, Rounding::ZERO);

        let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
        let seekbar_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        ui.painter().add(shadow_mesh);

        ui.painter().rect_filled(
            fullseekbar_rect,
            Rounding::ZERO,
            fullseekbar_color.linear_multiply(0.5),
        );
        ui.painter()
            .rect_filled(seekbar_rect, Rounding::ZERO, seekbar_color);
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

        if frame_response.clicked() {
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

            let audio_index_icon_rect = ui.painter().text(
                audio_index_icon_pos,
                Align2::RIGHT_BOTTOM,
                audio_index_icon,
                icon_font_id.clone(),
                text_color,
            );

            if ui
                .interact(
                    audio_index_icon_rect,
                    frame_response.id.with("audio_stream_icon_sense"),
                    Sense::click(),
                )
                .clicked()
            {
                if let Some(audio_streamer) = self.audio_streamer.as_mut() {
                    audio_streamer.lock().next_stream();
                }
            }

            if ui
                .interact(
                    sound_icon_rect,
                    frame_response.id.with("sound_icon_sense"),
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
            let sound_anim_id = frame_response.id.with("sound_anim");
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
            let sound_slider_bg_color =
                Color32::from_black_alpha(sound_slider_opacity).linear_multiply(sound_anim_frac);
            let sound_bar_color =
                Color32::from_white_alpha(sound_slider_opacity).linear_multiply(sound_anim_frac);
            let mut sound_bar_rect = sound_slider_rect;
            sound_bar_rect.set_top(
                sound_bar_rect.bottom()
                    - (self.audio_volume.get() / self.max_audio_volume) * sound_bar_rect.height(),
            );

            ui.painter()
                .rect_filled(sound_slider_rect, Rounding::same(5.), sound_slider_bg_color);

            ui.painter()
                .rect_filled(sound_bar_rect, Rounding::same(5.), sound_bar_color);
            let sound_slider_resp = ui.interact(
                sound_slider_rect,
                frame_response.id.with("sound_slider_sense"),
                Sense::click_and_drag(),
            );
            if sound_anim_frac > 0. && sound_slider_resp.clicked() || sound_slider_resp.dragged() {
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
    pub fn add_audio(&mut self, audio_device: &mut AudioDevice) -> Result<()> {
        let audio_input_context = input(&self.input_path)?;
        let audio_streams: Vec<ffmpeg::Stream> = audio_input_context
            .streams()
            .filter(|s| s.parameters().medium() == Type::Audio)
            .collect();
        let audio_stream_ids: Vec<_> = audio_streams.iter().map(|s| s.index()).collect();

        let audio_streamer = if let Some(audio_stream) = audio_streams.first() {
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
                video_elapsed_ms: self.video_elapsed_ms.clone(),
                audio_elapsed_ms: self.audio_elapsed_ms.clone(),
                audio_sample_producer,
                input_context: audio_input_context,
                audio_decoder,
                audio_stream_index,
                resampler: audio_resampler,
                audio_stream_ids,
            })
        } else {
            None
        };
        self.audio_streamer = audio_streamer.map(|s| Arc::new(Mutex::new(s)));
        Ok(())
    }

    /// Switches to the next audio stream.
    pub fn next_audio_stream(&mut self) -> Result<()> {
        if let Some(audio_streamer) = self.audio_streamer.take() {
            std::thread::spawn(move || {
                audio_streamer.lock().next_stream();
            });
        };
        Ok(())
    }

    /// Initializes the subtitle stream (if there is one), required for making a [`Player`] display subtitles.
    /// Will stop and reset the player's state.
    pub fn add_subtitles(&mut self) -> Result<()> {
        let subtitle_input_context = input(&self.input_path)?;
        let subtitle_stream = subtitle_input_context.streams().best(Type::Subtitle);

        let subtitle_streamer = if let Some(subtitle_stream) = subtitle_stream.as_ref() {
            let subtitle_stream_index = subtitle_stream.index();
            let subtitle_context =
                ffmpeg::codec::context::Context::from_parameters(subtitle_stream.parameters())?;
            let subtitle_decoder = subtitle_context.decoder().subtitle()?;

            self.stop_direct();

            Some(SubtitleStreamer {
                next_packet: None,
                duration_ms: self.duration_ms,
                player_state: self.player_state.clone(),
                video_elapsed_ms: self.video_elapsed_ms.clone(),
                _audio_elapsed_ms: self.audio_elapsed_ms.clone(),
                subtitle_elapsed_ms: self.subtitle_elapsed_ms.clone(),
                input_context: subtitle_input_context,
                subtitles_queue: self.subtitles_queue.clone(),
                subtitle_decoder,
                subtitle_stream_index,
            })
        } else {
            dbg!("bruh");
            None
        };
        self.subtitle_streamer = subtitle_streamer.map(|s| Arc::new(Mutex::new(s)));
        Ok(())
    }

    /// Enables using [`Player::add_audio`] with the builder pattern.
    pub fn with_audio(mut self, audio_device: &mut AudioDevice) -> Result<Self> {
        self.add_audio(audio_device)?;
        Ok(self)
    }

    /// Enables using [`Player::add_subtitles`] with the builder pattern.
    pub fn with_subtitles(mut self) -> Result<Self> {
        self.add_subtitles()?;
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
        let size = Vec2::new(width as f32, height as f32);
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
            subtitle_streamer: None,
            video_streamer: Arc::new(Mutex::new(stream_decoder)),
            texture_options,
            framerate,
            video_timer: Timer::new(),
            audio_timer: Timer::new(),
            subtitle_timer: Timer::new(),
            subtitle_elapsed_ms: Shared::new(0),
            preseek_player_state: None,
            video_thread: None,
            subtitle_thread: None,
            audio_thread: None,
            texture_handle,
            player_state,
            video_elapsed_ms,
            audio_elapsed_ms,
            size,
            last_seek_ms: None,
            duration_ms,
            audio_volume,
            max_audio_volume,
            video_elapsed_ms_override: None,
            looping: true,
            ctx_ref: ctx.clone(),
            subtitles_queue: Arc::new(Mutex::new(VecDeque::new())),
            current_subtitles: Vec::new(),
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

            if let Err(_) = self.input_context().seek(target_ts, ..target_ts) {
                // dbg!(e); TODO: propogate error
            } else {
                self.decoder().flush();
                let mut previous_elapsed_ms = self.elapsed_ms().get();

                // this drop frame loop lets us refresh until current_ts is accurate
                if seeking_backwards {
                    while !currently_behind_target() {
                        let next_elapsed_ms = self.elapsed_ms().get();
                        if next_elapsed_ms > previous_elapsed_ms {
                            break;
                        }
                        previous_elapsed_ms = next_elapsed_ms;
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
    /// The elapsed time of this streamer, in milliseconds.
    fn elapsed_ms(&self) -> &Shared<i64>;
    /// The elapsed time of the primary streamer, in milliseconds.
    fn primary_elapsed_ms(&self) -> &Shared<i64>;
    /// The total duration of the stream, in milliseconds.
    fn duration_ms(&self) -> i64;
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
                // dbg!(&e, is_ffmpeg_incomplete_error(&e));
                if is_ffmpeg_incomplete_error(&e) {
                    self.recieve_next_packet()?;
                    self.recieve_next_packet_until_frame()
                } else {
                    Err(e)
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
                return Err(e);
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
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
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
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.audio_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
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

impl Streamer for SubtitleStreamer {
    type Frame = (ffmpeg::codec::subtitle::Subtitle, i64);
    type ProcessedFrame = Subtitle;
    fn is_primary_streamer(&self) -> bool {
        false
    }
    fn stream_index(&self) -> usize {
        self.subtitle_stream_index
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.subtitle_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.subtitle_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn recieve_next_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.input_context().packets().next() {
            let time_base = stream.time_base();
            if stream.index() == self.stream_index() {
                if let Some(dts) = packet.dts() {
                    self.elapsed_ms().set(timestamp_to_millisec(dts, time_base));
                }
                self.next_packet = Some(packet);
            }
        } else {
            self.decoder().send_eof()?;
        }
        Ok(())
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        if let Some(packet) = self.next_packet.take() {
            let mut decoded_frame = ffmpeg::Subtitle::new();
            self.subtitle_decoder.decode(&packet, &mut decoded_frame)?;
            Ok((decoded_frame, packet.duration()))
        } else {
            Err(ffmpeg::Error::from(AVERROR(EAGAIN)).into())
        }
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        // TODO: manage the case when frame rects len > 1
        let (frame, duration) = frame;
        if let Some(rect) = frame.rects().next() {
            Subtitle::from_ffmpeg_rect(rect).map(|s| s.with_duration_ms(duration))
        } else {
            anyhow::bail!("no subtitle")
        }
    }
    fn apply_frame(&mut self, frame: Self::ProcessedFrame) {
        let mut queue = self.subtitles_queue.lock();
        queue.push_back(frame)
    }
}

type FfmpegAudioFormat = ffmpeg::format::Sample;
type FfmpegAudioFormatType = ffmpeg::format::sample::Type;
trait AsFfmpegSample {
    fn to_sample(&self) -> FfmpegAudioFormat;
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

/// Create a new [`AudioDeviceCallback`] from an existing [`sdl2::AudioSubsystem`]. An [`AudioDevice`] is required for using audio.
pub fn init_audio_device(audio_sys: &sdl2::AudioSubsystem) -> Result<AudioDevice, String> {
    AudioDeviceCallback::init(audio_sys)
}

/// Create a new [`AudioDeviceCallback`]. Creates an [`sdl2::AudioSubsystem`]. An [`AudioDevice`] is required for using audio.
pub fn init_audio_device_default() -> Result<AudioDevice, String> {
    AudioDeviceCallback::init(&sdl2::init()?.audio()?)
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

fn is_ffmpeg_incomplete_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<ffmpeg::Error>(),
        Some(ffmpeg::Error::Other { errno } ) if *errno == EAGAIN
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
