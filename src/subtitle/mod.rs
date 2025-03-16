use anyhow::Result;
use egui::{Align2, Color32, Margin, Pos2, TextureHandle};
use std::fmt;

use self::ass::parse_ass_subtitle;

mod ass;

#[derive(Default)]
pub struct SubtitleBitmap {
    pub data: Vec<Color32>,
    pub x: usize,
    pub y: usize,
    pub w: u32,
    pub h: u32,
    pub tex_handle: Option<TextureHandle>,
}

impl fmt::Debug for SubtitleBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SubtitleBitmap")
        .field(&self.data)
        .field(&self.x)
        .field(&self.y)
        .field(&self.w)
        .field(&self.h)
        .finish()
    }
}

#[derive(Debug)]
pub struct Subtitle {
    pub text: String,
    pub fade: FadeEffect,
    pub alignment: Align2,
    pub primary_fill: Color32,
    pub position: Option<Pos2>,
    pub font_size: f32,
    pub margin: Margin,
    pub remaining_duration_ms: i64,
    pub presentation_time_ms: Option<i64>,
    pub showing: bool,
    pub bitmap: SubtitleBitmap,
}

// todo, among others
// struct Transition<'a> {
//     offset_start_ms: i64,
//     offset_end_ms: i64,
//     accel: f64,
//     field: SubtitleField<'a>,
// }

enum SubtitleField<'a> {
    Fade(FadeEffect),
    Alignment(Align2),
    PrimaryFill(Color32),
    Position(Pos2),
    #[allow(unused)]
    Undefined(&'a str),
}

#[derive(Debug, Default)]
pub struct FadeEffect {
    _fade_in_ms: i64,
    _fade_out_ms: i64,
}

impl Default for Subtitle {
    fn default() -> Self {
        Self {
            text: String::new(),
            fade: FadeEffect {
                _fade_in_ms: 0,
                _fade_out_ms: 0,
            },
            remaining_duration_ms: 0,
            font_size: 30.,
            margin: Margin::same(85),
            alignment: Align2::CENTER_CENTER,
            primary_fill: Color32::WHITE,
            position: None,
            presentation_time_ms: None,
            showing: false,
            bitmap: SubtitleBitmap::default(),
        }
    }
}

impl Subtitle {
    fn from_text(text: &str) -> Self {
        Subtitle::default().with_text(text)
    }
    pub(crate) fn with_text(mut self, text: &str) -> Self {
        self.text = String::from(text);
        self
    }
    pub(crate) fn with_duration_ms(mut self, duration_ms: i64) -> Self {
        self.remaining_duration_ms = duration_ms;
        self
    }
    pub(crate) fn with_presentation_time_ms(mut self, pts: i64) -> Self {
        self.presentation_time_ms = Some(pts);
        self
    }
    fn from_bitmap(bitmap: &ffmpeg::subtitle::Bitmap<'_>) -> Self {
        let mut subtitle = Subtitle::default();
        subtitle.bitmap.x = bitmap.x();
        subtitle.bitmap.y = bitmap.y();
        subtitle.bitmap.w = bitmap.width();
        subtitle.bitmap.h = bitmap.height();
        unsafe {
            let data: [*mut u8; 4] = (*bitmap.as_ptr()).data;
            let linesize: [i32; 4] = (*bitmap.as_ptr()).linesize;
            subtitle.bitmap.data.resize((bitmap.width() * bitmap.height()) as usize, Color32::BLACK);
            let mut i: usize = 0;
            for y in 0..bitmap.height() as isize {
                // pixel buffer
                let linedata = data[0].wrapping_offset(y * linesize[0] as isize);
                for x in 0..bitmap.width() as isize {
                    let color_id_x = *linedata.wrapping_offset(x);
                    let color = *(data[1] as *mut u32).wrapping_offset(color_id_x as isize);
                    let r = (color >> 16 & 0xFF) as u8;
                    let g = (color >> 8 & 0xFF) as u8;
                    let b = (color >> 0 & 0xFF) as u8;
                    let a = (color >> 24 & 0xFF) as u8;
                    subtitle.bitmap.data[i] = Color32::from_rgba_unmultiplied(r, g, b, a);
                    i += 1;
                }
            }
        }
        subtitle
    }
    pub(crate) fn from_ffmpeg_rect(rect: ffmpeg::subtitle::Rect) -> Result<Self> {
        match rect {
            ffmpeg::subtitle::Rect::Ass(ass) => parse_ass_subtitle(ass.get()),
            ffmpeg::subtitle::Rect::Bitmap(bitmap) => Ok(Subtitle::from_bitmap(&bitmap)),
            ffmpeg::subtitle::Rect::None(_none) => anyhow::bail!("no subtitle"),
            ffmpeg::subtitle::Rect::Text(text) => Ok(Subtitle::from_text(text.get())),
        }
    }
}

impl FadeEffect {
    fn _is_zero(&self) -> bool {
        self._fade_in_ms == 0 && self._fade_out_ms == 0
    }
}
