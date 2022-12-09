extern crate ffmpeg_next as ffmpeg;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use egui::epaint::Shadow;
use egui::{
    vec2, Align2, Color32, ColorImage, FontId, Image, Rect, Response, Rounding,
    Sense, TextureFilter, TextureHandle, TextureId, Ui, TextureOptions,
};
use ffmpeg::ffi::AV_TIME_BASE;
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{rescale, Rational};

use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use tempfile::NamedTempFile;
use timer::{Guard, Timer};


fn format_duration(dur: Duration) -> String {
    let dt = DateTime::<Utc>::from(UNIX_EPOCH) + dur;
    if dt.format("%H").to_string().parse::<i64>().unwrap() > 0 {
        dt.format("%H:%M:%S").to_string()
    } else {
        dt.format("%M:%S").to_string()
    }
}

const _TEST_BYTES: &[u8] = include_bytes!("../cat.gif");

pub struct VideoStream {
    pub stream_decoder: Arc<Mutex<StreamDecoder>>,
    pub player_state: Arc<Mutex<PlayerState>>,
    pub framerate: f64,
    texture_options: TextureOptions,
    texture_handle: TextureHandle,
    pub height: u32,
    pub width: u32,
    frame_timer: Timer,
    frame_thread: Option<Guard>,
    ctx_ref: egui::Context,
    pub looping: bool,
    duration_ms: i64,
    elapsed_ms: i64,
    last_seek_ms: Option<i64>,
    preseek_player_state: Option<PlayerState>,
    temp_file: Option<NamedTempFile>,
    elapsed_ms_arc: Arc<Mutex<i64>>,
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
    decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    frame_index: usize,
    player_state: Arc<Mutex<PlayerState>>,
    input_context: Input,
    elapsed_ms: i64,
    scaler: Context,
    elapsed_ms_arc: Arc<Mutex<i64>>,
}

use ffmpeg::Rescale;
const AV_TIME_BASE_RATIONAL: Rational = Rational(1, AV_TIME_BASE);

fn timestamp_to_millisec(timestamp: i64, time_base: Rational) -> i64 {
    (timestamp as f64 * (time_base.numerator() as f64) / (time_base.denominator() as f64) * 1000.)
        as i64
}
fn millisec_to_timestamp(millisec: i64, time_base: Rational) -> i64 {
    (millisec as f64 * (time_base.denominator() as f64) / (time_base.numerator() as f64) / 1000.)
        as i64
}

impl VideoStream {
    pub fn duration_text(&self) -> String {
        format!(
            "{} / {}",
            format_duration(Duration::milliseconds(self.elapsed_ms)),
            format_duration(Duration::milliseconds(self.duration_ms))
        )
    }

    fn set_state(&self, new_state: PlayerState) {
        let mut current_state = self.player_state.lock().unwrap();
        *current_state = new_state;
    }
    pub fn pause(&self) {
        self.set_state(PlayerState::Paused)
    }
    pub fn unpause(&self) {
        self.set_state(PlayerState::Playing)
    }
    pub fn stop(&self) {
        self.set_state(PlayerState::Stopped)
    }
    fn duration_frac(&self) -> f32 {
        self.elapsed_ms as f32 / self.duration_ms as f32
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
        let player_state = Arc::clone(&self.player_state);
        let wait_duration = Duration::milliseconds((1000. / self.framerate) as i64);
        let duration_ms = self.duration_ms;
        let timer_guard = self.frame_timer.schedule_repeating(wait_duration, move || {
            ctx.request_repaint();
            let mut stream_decoder = stream_decoder.lock().unwrap();
            let player_state = player_state.lock().unwrap().clone();

            if player_state == PlayerState::Playing {
                match stream_decoder.recieve_next_packet_until_frame() {
                    Ok(frame) => {
                        texture_handle.set(frame, texture_options);
                    }
                    _ => (),
                }
            } else if let PlayerState::Seeking(seek_frac) = player_state {
                let target_ms = (seek_frac as f64 * duration_ms as f64) as i64;
                let target: i64 =
                    millisec_to_timestamp(target_ms, stream_decoder.decoder.time_base());
                (seek_frac * stream_decoder.input_context.duration() as f32) as i64;
                let _ = stream_decoder.input_context.seek(target, ..target);
                if seek_frac >= 0.99 {
                    // prevent inifinite loop near end of stream
                    *stream_decoder.player_state.lock().as_deref_mut().unwrap() =
                        PlayerState::EndOfFile;
                } else if seek_frac > 0. {
                    // this drop frame loop lets us refresh until current_ts is accurate
                    while (stream_decoder.elapsed_ms as f64 / duration_ms as f64)
                        >= seek_frac as f64
                    {
                        stream_decoder.drop_frames();
                    }

                    // this drop frame loop drops frames until we are at desired
                    while (stream_decoder.elapsed_ms as f64 / duration_ms as f64)
                        <= seek_frac as f64
                    {
                        stream_decoder.drop_frames();
                    }

                    // frame preview
                    match stream_decoder.recieve_next_packet_until_frame() {
                        Ok(frame) => {
                            texture_handle.set(frame, texture_options);
                        }
                        _ => (),
                    }
                }
            }
        });
        self.frame_thread = Some(timer_guard)
        // }
    }

    pub fn start(&mut self) {
        self.frame_thread = None;
        if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
            stream_decoder.reset(true);
        }
        self.spawn_timer();
    }

    pub fn process_state(&mut self) {
        let mut reset_stream = false;
        if let Ok(elapsed_ms_arc) = self.elapsed_ms_arc.try_lock() {
            if self.last_seek_ms.is_some() {
                let last_seek_ms = *self.last_seek_ms.as_ref().unwrap();
                if *elapsed_ms_arc > last_seek_ms || *elapsed_ms_arc == 0 {
                    self.elapsed_ms = *elapsed_ms_arc;
                    self.last_seek_ms = None
                } else {
                    self.elapsed_ms = last_seek_ms;
                }
            } else {
                self.elapsed_ms = *elapsed_ms_arc
            }
        }

        if let Ok(player_state) = self.player_state.try_lock().as_deref_mut() {
            if let Ok(mut stream_decoder) = self.stream_decoder.try_lock() {
                match player_state {
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
                    _ => (),
                }
            }
        }

        if reset_stream {
            if let Ok(mut stream_decoder) = self.stream_decoder.try_lock() {
                stream_decoder.reset(true);
            }
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
        // self.texture_handle.as_ref().and_then(|h| Some(h.id()))
        self.texture_handle.id()
    }

    fn render_ui(&mut self, ui: &mut Ui, playback_response: &Response) -> Option<Rect> {
        let hovered = ui.rect_contains_pointer(playback_response.rect);
        let currently_seeking = matches!(
            self.player_state.try_lock().as_deref(),
            Ok(PlayerState::Seeking(_))
        );
        let is_stopped = matches!(
            self.player_state.try_lock().as_deref(),
            Ok(PlayerState::Stopped)
        );
        let is_paused = matches!(
            self.player_state.try_lock().as_deref(),
            Ok(PlayerState::Paused)
        );

        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            playback_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            0.2,
        );

        if playback_response.clicked() {
            let mut reset_stream = false;
            let mut start_stream = false;
            if let Ok(current_state) = self.player_state.lock().as_deref_mut() {
                match current_state {
                    PlayerState::Stopped => start_stream = true,
                    PlayerState::EndOfFile => reset_stream = true,
                    PlayerState::Paused => *current_state = PlayerState::Playing,
                    PlayerState::Playing => *current_state = PlayerState::Paused,
                    _ => (),
                }
            }

            if reset_stream {
                if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
                    stream_decoder.reset(true)
                }
            } else if start_stream {
                self.start();
            }
        }

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
                            if let Ok(mut stream_decoder) = self.stream_decoder.lock() {
                                stream_decoder.reset(true);
                            }
                            self.spawn_timer();
                        }
                        if !currently_seeking {
                            self.preseek_player_state =
                                Some(self.player_state.lock().unwrap().clone());
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
            let seekbar_color = Color32::RED.linear_multiply(seekbar_anim_frac);

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
        let duration_ms = timestamp_to_millisec(input_context.duration(), AV_TIME_BASE_RATIONAL); // in sec
        let elapsed_ms_arc = Arc::new(Mutex::new(0));
        let stream_decoder = StreamDecoder {
            decoder: video_decoder,
            video_stream_index,
            elapsed_ms_arc: Arc::clone(&elapsed_ms_arc),
            frame_index: 0,
            elapsed_ms: 0,
            input_context,
            player_state: Arc::clone(&player_state),
            scaler: frame_scaler,
        };
        let texture_options = TextureOptions::LINEAR;
        let texture_handle = ctx.load_texture("vidstream", ColorImage::example(), texture_options);

        let mut streamer = Self {
            stream_decoder: Arc::new(Mutex::new(stream_decoder)),
            texture_options,
            framerate,
            frame_timer: Timer::new(),
            preseek_player_state: None,
            frame_thread: None,
            texture_handle,
            player_state,
            elapsed_ms_arc,
            width,
            last_seek_ms: None,
            duration_ms,
            elapsed_ms: 0,
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
                        .load_texture("vidstream", first_frame, self.texture_options);
                let texture_handle_clone = texture_handle.clone();
                self.texture_handle = texture_handle;
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
                    self.elapsed_ms = timestamp_to_millisec(dts, stream.time_base());
                    if let Ok(mut elapsed_ms_arc) = self.elapsed_ms_arc.lock() {
                        *elapsed_ms_arc = self.elapsed_ms;
                    }
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

    pub fn recieve_next_packet_until_frame(&mut self) -> Result<ColorImage> {
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

    ColorImage { size, pixels }
}

pub fn init() {
    ffmpeg::init().unwrap();
}
