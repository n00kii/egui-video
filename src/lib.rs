extern crate ffmpeg_next as ffmpeg;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use egui::epaint::Shadow;
use egui::{
    vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Rounding, Sense,
    TextureHandle, TextureId, TextureOptions, Ui,
};
use ffmpeg::ffi::AV_TIME_BASE;
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Audio;
use ffmpeg::media::Type;
use ffmpeg::software;
use ffmpeg::util::frame::video::Video;
use ffmpeg::{rescale, Rational, Rescale};
use rand::{thread_rng, Rng};
use sdl2::audio::{AudioCallback, AudioDevice, AudioFormat, AudioSpecDesired};

use parking_lot::Mutex;
use std::io::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use tempfile::NamedTempFile;
use timer::{Guard, Timer};

use ringbuf::{Consumer, HeapRb, SharedRb};

fn arc_atomic_i64() -> Arc<AtomicI64> {
    Arc::new(AtomicI64::new(0))
}

fn format_duration(dur: Duration) -> String {
    let dt = DateTime::<Utc>::from(UNIX_EPOCH) + dur;
    if dt.format("%H").to_string().parse::<i64>().unwrap() > 0 {
        dt.format("%H:%M:%S").to_string()
    } else {
        dt.format("%M:%S").to_string()
    }
}

const _TEST_BYTES: &[u8] = include_bytes!("../cat.gif");
type AudioSampleProducer =
    ringbuf::Producer<f32, Arc<ringbuf::SharedRb<f32, Vec<std::mem::MaybeUninit<f32>>>>>;
type AudioSampleConsumer =
    ringbuf::Consumer<f32, Arc<ringbuf::SharedRb<f32, Vec<std::mem::MaybeUninit<f32>>>>>;

pub struct AudioDecoder {
    video_elapsed_ms: Cache<i64>,
    audio_elapsed_ms: Cache<i64>,
    audio_stream_index: usize,
    audio_decoder: ffmpeg::decoder::Audio,
    resampler: software::resampling::Context,
    audio_sample_producer: AudioSampleProducer,
    audio_input_context: Input,
    player_state: Cache<PlayerState>,
}

pub struct VideoStream {
    pub stream_decoder: Arc<Mutex<StreamDecoder>>,
    pub audio_decoder: Option<Arc<Mutex<AudioDecoder>>>,
    pub player_state: Cache<PlayerState>,
    pub framerate: f64,
    texture_options: TextureOptions,
    texture_handle: TextureHandle,
    pub height: u32,
    pub width: u32,
    frame_timer: Timer,
    audio_timer: Timer,
    audio_thread: Option<Guard>,
    frame_thread: Option<Guard>,
    ctx_ref: egui::Context,
    pub looping: bool,
    audio_device: AudioDevice<AudioPlayCallback>,
    duration_ms: i64,
    last_seek_ms: Option<i64>,
    preseek_player_state: Option<PlayerState>,
    temp_file: Option<NamedTempFile>,
    video_elapsed_ms: Cache<i64>,
    audio_elapsed_ms: Cache<i64>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PlayerState {
    Stopped,
    EndOfFile,
    Seeking(f32),
    Paused,
    Playing,
}

pub struct StreamDecoder {
    video_decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    player_state: Cache<PlayerState>,
    input_context: Input,
    video_elapsed_ms: Cache<i64>,
    audio_elapsed_ms: Cache<i64>,
    scaler: software::scaling::Context,
}

#[derive(Clone)]
pub struct Cache<T: Copy> {
    cached_value: T,
    override_value: Option<T>,
    raw_value: Arc<Mutex<T>>,
}

impl<T: Copy> Cache<T> {
    fn set(&mut self, value: T) {
        self.cached_value = value;
        *self.raw_value.lock() = value
    }
    pub fn get(&mut self) -> T {
        self.override_value.unwrap_or(self.get_true())
    }
    pub fn get_true(&mut self) -> T {
        self.try_update_cache().unwrap_or(self.cached_value)
    }
    fn update_cache(&mut self) {
        self.cached_value = *self.raw_value.lock();
    }
    fn get_updated(&mut self) -> T {
        self.update_cache();
        self.cached_value
    }
    fn try_update_cache(&mut self) -> Option<T> {
        if let Some(new_value) = self.raw_value.try_lock() {
            self.cached_value = *new_value;
            Some(self.cached_value)
        } else {
            None
        }
    }
    fn new(value: T) -> Self {
        Self {
            override_value: None,
            cached_value: value,
            raw_value: Arc::new(Mutex::new(value)),
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

impl VideoStream {
    pub fn duration_text(&mut self) -> String {
        format!(
            "{} / {}",
            format_duration(Duration::milliseconds(self.video_elapsed_ms.get())),
            format_duration(Duration::milliseconds(self.duration_ms))
        )
    }
    fn reset(&mut self, start_playing: bool) {
        self.stream_decoder.lock().reset(start_playing);
        if let Some(audio_decoder) = self.audio_decoder.as_mut() {
            audio_decoder.lock().reset(start_playing);
        }
    }
    fn set_state(&mut self, new_state: PlayerState) {
        self.player_state.set(new_state)
    }
    pub fn pause(&mut self) {
        self.set_state(PlayerState::Paused)
    }
    pub fn unpause(&mut self) {
        self.set_state(PlayerState::Playing)
    }
    pub fn stop(&mut self) {
        self.set_state(PlayerState::Stopped)
    }
    fn duration_frac(&mut self) -> f32 {
        self.video_elapsed_ms.get() as f32 / self.duration_ms as f32
    }
    pub fn cleanup(self) {
        drop(self)
    }

    fn spawn_timer(&mut self) {
        // if let Some(texture_handle) = self.texture_handle.as_ref() {
        let mut texture_handle = self.texture_handle.clone();
        let texture_options = self.texture_options.clone();
        let ctx = self.ctx_ref.clone();
        let stream_decoder = Arc::clone(&self.stream_decoder);
        let mut player_state = self.player_state.clone();
        let wait_duration = Duration::milliseconds((1000. / self.framerate) as i64);
        let duration_ms = self.duration_ms;
        let frame_timer_guard = self.frame_timer.schedule_repeating(wait_duration, move || {
            ctx.request_repaint();
            let mut stream_decoder = stream_decoder.lock();
            let player_state = player_state.get_updated();

            if player_state == PlayerState::Playing {
                match stream_decoder.recieve_next_packet_until_video_frame() {
                    Ok(frame) => {
                        texture_handle.set(frame, texture_options);
                    }
                    _ => (),
                }
            } else if let PlayerState::Seeking(seek_frac) = player_state {
                let target_ms = (seek_frac as f64 * duration_ms as f64) as i64;
                let seeking_forward = target_ms > stream_decoder.video_elapsed_ms.get();
                
                let target_ts = millisec_to_timestamp(target_ms, rescale::TIME_BASE);//target_ms.rescale((1, 1000), rescale::TIME_BASE);

                if let Err(e) = stream_decoder.input_context.seek(target_ts, ..target_ts) {
                    dbg!(e);
                } else {
                    if seek_frac >= 0.99 {
                        // prevent inifinite loop near end of stream
                        stream_decoder.player_state.set(PlayerState::EndOfFile)

                    } else if seek_frac > 0. {
                        // this drop frame loop lets us refresh until current_ts is accurate
                        if !seeking_forward {
                            while (stream_decoder.video_elapsed_ms.get() as f64 / duration_ms as f64)
                                > seek_frac as f64
                            {
                                stream_decoder.drop_frames();
                            }
                        }

                        // this drop frame loop drops frames until we are at desired
                        while (stream_decoder.video_elapsed_ms.get() as f64 / duration_ms as f64)
                            < seek_frac as f64
                        {
                            stream_decoder.drop_frames();
                        }

                        // frame preview
                        match stream_decoder.recieve_next_packet_until_video_frame() {
                            Ok(frame) => {
                                texture_handle.set(frame, texture_options);
                            }
                            _ => (),
                        }
                    }
                };
            }
        });
        if let Some(audio_decoder) = self.audio_decoder.as_ref() {
            self.audio_device.resume();
            let audio_decoder = Arc::clone(&audio_decoder);
            let mut player_state = self.player_state.clone();
            let audio_timer_guard = self.audio_timer.schedule_repeating(Duration::zero(), move || {

                let player_state = player_state.get_updated();
                let mut audio_decoder = audio_decoder.lock();
                if player_state == PlayerState::Playing {
                   let _ = audio_decoder.recieve_next_packet_until_audio_frame();
                } else if let PlayerState::Seeking(seek_frac) = player_state {
                    let target_ms = (seek_frac as f64 * duration_ms as f64) as i64;
                    let seeking_forward = target_ms > audio_decoder.audio_elapsed_ms.get();
                    let target_ts = millisec_to_timestamp(target_ms, rescale::TIME_BASE);//target_ms.rescale((1, 1000), rescale::TIME_BASE);
    
                    if let Err(e) = audio_decoder
                        .audio_input_context
                        .seek(target_ts, ..target_ts)
                    {
                        dbg!(e);
                    } else {
                        if seek_frac >= 0.99 {
                            // prevent inifinite loop near end of stream
                            audio_decoder.player_state.set(PlayerState::EndOfFile);
                        } else if seek_frac > 0. {
                            // // this drop frame loop lets us refresh until current_ts is accurate
                            if !seeking_forward {
                                while (audio_decoder.audio_elapsed_ms.get() as f64 / duration_ms as f64)
                                    > seek_frac as f64
                                {
                                    audio_decoder.drop_frames();
                                }
                            }
    
                            // this drop frame loop drops frames until we are at desired
                            while (audio_decoder.audio_elapsed_ms.get() as f64 / duration_ms as f64)
                                < seek_frac as f64
                            {
                                audio_decoder.drop_frames();
                            }

                        }
                    };
                }
            });
            self.frame_thread = Some(frame_timer_guard);
            self.audio_thread = Some(audio_timer_guard);
        }
    }

    pub fn start(&mut self) {
        self.frame_thread = None;
        self.audio_thread = None;
        self.reset(true);
        self.spawn_timer();
    }

    pub fn process_state(&mut self) {
        let mut reset_stream = false;
        let video_elapsed_ms = self.video_elapsed_ms.get();
        if self.last_seek_ms.is_some() {
            let last_seek_ms = *self.last_seek_ms.as_ref().unwrap();
            if self.video_elapsed_ms.get_true() > last_seek_ms || video_elapsed_ms == 0 {
                self.video_elapsed_ms.override_value = None;
                self.last_seek_ms = None;
            } else {
                self.video_elapsed_ms.override_value = Some(last_seek_ms);
            }
        } else {
            self.video_elapsed_ms.override_value = None;
        }

            match self.player_state.get() {
                PlayerState::EndOfFile => {
                    if self.looping {
                        reset_stream = true;
                    } else {
                        self.player_state.set(PlayerState::Stopped);
                    }
                }
                PlayerState::Stopped => {
                    self.frame_thread = None;
                    self.audio_thread = None;
                }
                _ => (),
            }

        if reset_stream {
            self.reset(true);
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, size: [f32; 2]) -> egui::Response {
        let image = Image::new(self.texture_handle.id(), size).sense(Sense::click());
        let response = ui.add(image);
        self.render_ui(ui, &response);
        response
    }

    pub fn ui_at(&mut self, ui: &mut Ui, rect: Rect) -> egui::Response {
        let image = Image::new(self.texture_handle.id(), rect.size()).sense(Sense::click());
        let response = ui.put(rect, image);
        self.render_ui(ui, &response);
        response
    }

    pub fn texture_id(&self) -> TextureId {
        self.texture_handle.id()
    }

    fn render_ui(&mut self, ui: &mut Ui, playback_response: &Response) -> Option<Rect> {
        let hovered = ui.rect_contains_pointer(playback_response.rect);
        let currently_seeking = matches!(self.player_state.get(), (PlayerState::Seeking(_)));
        let is_stopped = matches!(self.player_state.get(), (PlayerState::Stopped));
        let is_paused = matches!(self.player_state.get(), (PlayerState::Paused));

        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            playback_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            0.2,
        );

        if playback_response.clicked() {
            let mut reset_stream = false;
            let mut start_stream = false;
            // if let Ok(current_state) =  {
            match self.player_state.get_updated() {
                PlayerState::Stopped => start_stream = true,
                PlayerState::EndOfFile => reset_stream = true,
                PlayerState::Paused => self.player_state.set(PlayerState::Playing),
                PlayerState::Playing => self.player_state.set(PlayerState::Paused),
                _ => (),
            }

            if reset_stream {
                self.reset(true);
            } else if start_stream {
                self.start();
            }
        }

        if seekbar_anim_frac > 0. {
            let seekbar_width_offset = 20.;
            let fullseekbar_width = playback_response.rect.width() - seekbar_width_offset;

            let seekbar_width = if let (PlayerState::Seeking(h)) = self.player_state.get() {
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
            let seekbar_interact_rect = fullseekbar_rect.expand(10.);
            ui.interact(seekbar_interact_rect, playback_response.id, Sense::drag());

            let seekbar_hovered = ui.rect_contains_pointer(seekbar_interact_rect);
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
                        if is_stopped {
                            // if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
                            self.reset(true);
                            self.spawn_timer();
                        }
                        if !currently_seeking {
                            self.preseek_player_state = Some(self.player_state.get_updated());
                        }
                        self.set_state(PlayerState::Seeking(seek_frac));
                        self.last_seek_ms =
                            Some((seek_frac as f64 * self.duration_ms as f64) as i64);
                        seekbar_rect.set_right(
                            hover_pos
                                .x
                                .min(fullseekbar_rect.right())
                                .max(fullseekbar_rect.left()),
                        );
                    } else if ui.ctx().input().pointer.any_released() {
                        if let Some(previous_state) = self.preseek_player_state.take() {
                            self.set_state(previous_state)
                        } else {
                            self.set_state(PlayerState::Playing)
                        }
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
            let mut icon_font_id = FontId::default();
            icon_font_id.size = 16.;

            let text_y_offset = -7.;
            let sound_icon = "🔊";
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

            ui.painter().text(
                sound_icon_pos,
                Align2::RIGHT_BOTTOM,
                sound_icon,
                icon_font_id.clone(),
                text_color,
            );
            if seekbar_hover_anim_frac > 0. {
                ui.painter().circle_filled(
                    seekbar_rect.right_center(),
                    7. * seekbar_hover_anim_frac,
                    seekbar_color,
                );
            }

            Some(seekbar_interact_rect)
        } else {
            None
        }
    }

    pub fn new_from_bytes(ctx: &egui::Context, input_bytes: &[u8]) -> Result<Self> {
        let mut file = tempfile::Builder::new().tempfile()?;
        file.write_all(input_bytes)?;
        let path = file.path().to_string_lossy().to_string();
        let mut slf = Self::new(ctx, &path)?;
        slf.temp_file = Some(file);
        Ok(slf)
    }

    pub fn new(ctx: &egui::Context, input_path: &String) -> Result<Self> {
        let input_context = input(&input_path)?;
        let video_stream = input_context
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = video_stream.index();

        let audio_stream = input_context.streams().best(Type::Audio);
        let mut audio_device = AudioPlayCallback::init().unwrap();
        // let audio_elapsed_ms_arc = arc_atomic_i64();
        // let video_elapsed_ms_arc = arc_atomic_i64();
        let video_elapsed_ms = Cache::new(0);
        let audio_elapsed_ms = Cache::new(0);
        let player_state = Cache::new(PlayerState::Stopped);

        let audio_decoder = if let Some(audio_stream) = audio_stream.as_ref() {
            let audio_input_context = input(&input_path)?;
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
                audio_decoder.channel_layout(),
                audio_device.spec().freq as u32,
            )?;

            audio_device.lock().sample_consumer = Some(audio_sample_consumer);

            Some(AudioDecoder {
                player_state: player_state.clone(),
                video_elapsed_ms: video_elapsed_ms.clone(),
                audio_elapsed_ms: audio_elapsed_ms.clone(),
                // audio_elapsed_ms_arc: Arc::clone(&audio_elapsed_ms_arc),
                // video_elapsed_ms_arc: Arc::clone(&video_elapsed_ms_arc),
                // audio_elapsed_ms: 0,
                audio_sample_producer,
                audio_input_context,
                audio_decoder,
                audio_stream_index,
                resampler: audio_resampler,
            })
        } else {
            None
        };

        let video_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let video_decoder = video_context.decoder().video()?;
        let framerate = (video_stream.avg_frame_rate().numerator() as f64)
            / video_stream.avg_frame_rate().denominator() as f64;

        let (width, height) = (video_decoder.width(), video_decoder.height());
        let frame_scaler = software::scaling::Context::get(
            video_decoder.format(),
            video_decoder.width(),
            video_decoder.height(),
            Pixel::RGB24,
            video_decoder.width(),
            video_decoder.height(),
            software::scaling::flag::Flags::BILINEAR,
        )?;

        let duration_ms = timestamp_to_millisec(input_context.duration(), AV_TIME_BASE_RATIONAL); // in sec
        let stream_decoder = StreamDecoder {
            video_decoder,
            video_stream_index,
            audio_elapsed_ms: audio_elapsed_ms.clone(),
            video_elapsed_ms: video_elapsed_ms.clone(),
            // audio_elapsed_ms_arc: Arc::clone(&audio_elapsed_ms_arc),
            // video_elapsed_ms_arc: Arc::clone(&video_elapsed_ms_arc),
            // video_elapsed_ms: 0,
            input_context,
            player_state: player_state.clone(),
            scaler: frame_scaler,
        };
        let texture_options = TextureOptions::LINEAR;
        let texture_handle = ctx.load_texture("vidstream", ColorImage::example(), texture_options);

        let mut streamer = Self {
            audio_device,
            audio_decoder: audio_decoder.map(|ad| Arc::new(Mutex::new(ad))),
            stream_decoder: Arc::new(Mutex::new(stream_decoder)),
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
            // video_elapsed_ms_arc,
            // audio_elapsed_ms_arc,
            width,
            last_seek_ms: None,
            duration_ms,
            // elapsed_ms: 0,
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
            .recieve_next_packet_until_video_frame()
            // .and_then(|mut s| Ok(s.))
        {
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

trait Streamer {
    fn decode_frame<F>(&mut self, frame_buffer: F) -> Result<()>;
    fn empty_frame<F>() -> F;
    fn drop_frames<F>(&mut self) {
        if self.decode_frame(Self::empty_frame::<F>()).is_err() {
            let _ = self.recieve_next_packet();
        } else {
            self.drop_frames::<F>();
        }
    }
    fn recieve_next_packet(&mut self) -> Result<()>;
    fn reset(&mut self);
    fn recieve_next_packet_until_frame(&mut self);
    fn recieve_next_frame(&mut self);
}

impl StreamDecoder {
    fn recieve_next_video_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.input_context.packets().next() {
            if stream.index() == self.video_stream_index {
                self.video_decoder.send_packet(&packet)?;
                if let Some(dts) = packet.dts() {
                    self.video_elapsed_ms.set(timestamp_to_millisec(dts, stream.time_base()));
                    // self.video_elapsed_ms = timestamp_to_millisec(dts, stream.time_base());
                    // self.video_elapsed_ms_arc
                    //     .store(self.video_elapsed_ms, Ordering::Relaxed)
                }
            }
        } else {
            self.video_decoder.send_eof()?;
            self.player_state.set(PlayerState::EndOfFile);
        }
        Ok(())
    }

    fn drop_frames(&mut self) {
        let mut decoded_frame = Video::empty();
        if self
            .video_decoder
            .receive_frame(&mut decoded_frame)
            .is_err()
        {
            let _ = self.recieve_next_video_packet();
        } else {
            self.drop_frames();
        }
    }

    fn reset(&mut self, start_playing: bool) {
        let beginning: i64 = 0;
        let beginning_seek = beginning.rescale((1, 1), rescale::TIME_BASE);
        let _ = self.input_context.seek(beginning_seek, ..beginning_seek);
        self.video_decoder.flush();
        if start_playing {
            self.player_state.set(PlayerState::Playing);
        }
    }

    pub fn recieve_next_packet_until_video_frame(&mut self) -> Result<ColorImage> {
        if let Ok(color_image) = self.recieve_next_video_frame() {
            Ok(color_image)
        } else {
            self.recieve_next_video_packet()?;
            self.recieve_next_packet_until_video_frame()
        }
    }

    fn recieve_next_video_frame(&mut self) -> Result<ColorImage> {
        let mut decoded_frame = Video::empty();
        match self.video_decoder.receive_frame(&mut decoded_frame) {
            Ok(()) => {
                let mut rgb_frame = Video::empty();
                self.scaler.run(&decoded_frame, &mut rgb_frame)?;

                // let audio_elapsed_ms = self.audio_elapsed_ms_arc.load(Ordering::Relaxed);
                // if self.video_elapsed_ms > audio_elapsed_ms {
                //     std::thread::sleep(std::time::Duration::from_millis(10));
                // }

                let image = video_frame_to_image(rgb_frame);
                Ok(image)
            }
            Err(e) => {
                return Err(e.into());
            }
        }
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

struct AudioPlayCallback {
    sample_consumer: Option<AudioSampleConsumer>,
    volume: f32,
}

impl AudioCallback for AudioPlayCallback {
    type Channel = f32;
    fn callback(&mut self, output: &mut [Self::Channel]) {
        if let Some(sample_consumer) = self.sample_consumer.as_mut() {
            for x in output.iter_mut() {
                match sample_consumer.pop() {
                    Some(sample) => *x = sample * self.volume,
                    None => *x = 0.,
                }
            }
        }
    }
}

impl AudioPlayCallback {
    fn init() -> Result<AudioDevice<Self>, String> {
        let sdl_ctx = sdl2::init()?;
        let audio_sys = sdl_ctx.audio()?;

        let audio_spec = AudioSpecDesired {
            freq: Some(44_100),
            channels: Some(2),
            samples: None,
        };

        let device = audio_sys.open_playback(None, &audio_spec, |spec| {
            dbg!(&spec);
            AudioPlayCallback {
                sample_consumer: None,
                volume: 1.,
            }
        })?;

        Ok(device)
    }
}

impl AudioDecoder {
    fn recieve_next_audio_frame(&mut self) -> Result<()> {
        let mut decoded_frame = ffmpeg::frame::Audio::empty();
        match self.audio_decoder.receive_frame(&mut decoded_frame) {
            Ok(()) => {
                let mut resampled_frame = ffmpeg::frame::Audio::empty();
                self.resampler.run(&decoded_frame, &mut resampled_frame)?;
                let audio_samples = if resampled_frame.is_packed() {
                    packed(&resampled_frame)
                } else {
                    resampled_frame.plane(0)
                };

                while self.audio_sample_producer.free_len() < audio_samples.len() {
                    // std::thread::sleep(std::time::Duration::from_millis(10));
                }

                // let video_elapsed_ms = self.video_elapsed_ms_arc.load(Ordering::Relaxed);

                // if video_elapsed_ms < self.audio_elapsed_ms {
                //     std::thread::sleep(std::time::Duration::from_millis(10));
                // }

                self.audio_sample_producer.push_slice(audio_samples);
            }
            Err(e) => {
                return Err(e.into());
            }
        }
        Ok(())
    }
    pub fn recieve_next_packet_until_audio_frame(&mut self) -> Result<()> {
        if let Ok(()) = self.recieve_next_audio_frame() {
            Ok(())
        } else {
            self.recieve_next_audio_packet()?;
            self.recieve_next_packet_until_audio_frame()
        }
    }
    fn drop_frames(&mut self) {
        let mut decoded_frame = Audio::empty();
        if self
            .audio_decoder
            .receive_frame(&mut decoded_frame)
            .is_err()
        {
            let _ = self.recieve_next_audio_packet();
        } else {
            self.drop_frames();
        }
    }
    fn recieve_next_audio_packet(&mut self) -> Result<()> {
        if let Some((stream, packet)) = self.audio_input_context.packets().next() {
            if stream.index() == self.audio_stream_index {
                self.audio_decoder.send_packet(&packet)?;
                if let Some(dts) = packet.dts() {
                    self.audio_elapsed_ms.set(timestamp_to_millisec(dts, stream.time_base()));
                    // self.audio_elapsed_ms = timestamp_to_millisec(dts, stream.time_base());
                    // self.audio_elapsed_ms_arc
                    //     .store(self.audio_elapsed_ms, Ordering::Relaxed)
                }
            }
        } else {
            self.audio_decoder.send_eof()?;
        }
        Ok(())
    }
    fn reset(&mut self, start_playing: bool) {
        let beginning: i64 = 0;
        let beginning_seek = beginning.rescale((1, 1), rescale::TIME_BASE);
        let _ = self
            .audio_input_context
            .seek(beginning_seek, ..beginning_seek);
        self.audio_decoder.flush();
        if start_playing {
            self.player_state.set(PlayerState::Playing);
        }
    }
}

#[inline]
// Interpret the audio frame's data as packed (alternating channels, 12121212, as opposed to planar 11112222)
pub fn packed<T: ffmpeg::frame::audio::Sample>(frame: &ffmpeg::frame::Audio) -> &[T] {
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

pub fn init() {
    ffmpeg::init().unwrap();
}
