use anyhow::Result;
use egui::{text::LayoutJob, Align2, Color32, FontId, Pos2, Margin};

use self::ass::parse_ass_subtitle;

mod ass;

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
    Undefined(&'a str),
}

#[derive(Debug, Default)]
pub struct FadeEffect {
    fade_in_ms: i64,
    fade_out_ms: i64,
}

impl Default for Subtitle {
    fn default() -> Self {
        Self {
            text: String::new(),
            fade: FadeEffect {
                fade_in_ms: 0,
                fade_out_ms: 0,
            },
            remaining_duration_ms: 0,
            font_size: 30.,
            margin: Margin::same(85.),
            alignment: Align2::CENTER_CENTER,
            primary_fill: Color32::WHITE,
            position: None,
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
    pub(crate) fn from_ffmpeg_rect<'a>(rect: ffmpeg::subtitle::Rect<'a>) -> Result<Self> {
        match rect {
            ffmpeg::subtitle::Rect::Ass(ass) => parse_ass_subtitle(ass.get()),
            ffmpeg::subtitle::Rect::Bitmap(_bitmap) => {
                Ok(Subtitle::from_text("[ unsupported bitmap subtitle ]"))
            }
            ffmpeg::subtitle::Rect::None(_none) => anyhow::bail!("no subtitle"),
            ffmpeg::subtitle::Rect::Text(text) => Ok(Subtitle::from_text(text.get())),
        }
    }
    pub(crate) fn to_layout_job(&self) -> LayoutJob {
        let font_id = FontId::default();
        LayoutJob::simple(String::from(&self.text), font_id, self.primary_fill, 1000.)
    }
}

impl FadeEffect {
    fn is_zero(&self) -> bool {
        self.fade_in_ms == 0 && self.fade_out_ms == 0
    }
}
