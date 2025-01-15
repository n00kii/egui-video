use anyhow::{anyhow, bail, Context, Result};
#[cfg(feature = "eframe")]
use eframe::egui::{Align2, Color32, Pos2};
#[cfg(not(feature = "eframe"))]
use egui::{Align2, Color32, Pos2};
use nom::branch::alt;
use nom::bytes::complete::{is_not, tag, take_till, take_until, take_while_m_n};
use nom::character::complete::{char, digit0, digit1};
use nom::combinator::{map, map_res, opt, rest};
use nom::error::context;
use nom::multi::{many0, separated_list0};
use nom::number::complete::double;
use nom::sequence::{delimited, pair, preceded, tuple};
use nom::{AsChar, IResult};

use super::{FadeEffect, Subtitle, SubtitleField};

fn num_list(i: &str) -> IResult<&str, Vec<f64>> {
    delimited(char('('), separated_list0(char(','), double), char(')'))(i)
}

fn tuple_int_2(v: Vec<f64>) -> Result<(i64, i64)> {
    tuple_float_2(v).map(|v| (v.0 as i64, v.1 as i64))
}

fn tuple_float_2(v: Vec<f64>) -> Result<(f64, f64)> {
    const FAIL_TEXT: &str = "invalid number of items";
    Ok((
        *v.first().context(FAIL_TEXT)?,
        *v.get(1).context(FAIL_TEXT)?,
    ))
}

fn fad(i: &str) -> IResult<&str, SubtitleField> {
    preceded(
        tag(r"\fad"),
        map(map_res(num_list, tuple_int_2), |f| {
            let fade_effect = SubtitleField::Fade(FadeEffect {
                _fade_in_ms: f.0,
                _fade_out_ms: f.1,
            });
            fade_effect
        }),
    )(i)
}

fn t(i: &str) -> IResult<&str, SubtitleField> {
    preceded(
        tag(r"\t"),
        delimited(
            char('('),
            map(take_until(")"), |_| {
                SubtitleField::Undefined("transition not implemented")
            }),
            char(')'),
        ),
    )(i)
}

fn an(i: &str) -> IResult<&str, SubtitleField> {
    preceded(
        tag(r"\an"),
        map_res(digit1, |s: &str| match s.parse::<i64>() {
            Ok(1) => Ok(SubtitleField::Alignment(Align2::LEFT_BOTTOM)),
            Ok(2) => Ok(SubtitleField::Alignment(Align2::CENTER_BOTTOM)),
            Ok(3) => Ok(SubtitleField::Alignment(Align2::RIGHT_BOTTOM)),

            Ok(4) => Ok(SubtitleField::Alignment(Align2::LEFT_CENTER)),
            Ok(5) => Ok(SubtitleField::Alignment(Align2::CENTER_CENTER)),
            Ok(6) => Ok(SubtitleField::Alignment(Align2::RIGHT_CENTER)),

            Ok(7) => Ok(SubtitleField::Alignment(Align2::LEFT_TOP)),
            Ok(8) => Ok(SubtitleField::Alignment(Align2::CENTER_TOP)),
            Ok(9) => Ok(SubtitleField::Alignment(Align2::RIGHT_TOP)),
            _ => bail!("invalid alignment"),
        }),
    )(i)
}

fn pos(i: &str) -> IResult<&str, SubtitleField> {
    preceded(
        tag(r"\pos"),
        map(map_res(num_list, tuple_float_2), |p| {
            SubtitleField::Position(Pos2::new(p.0 as f32, p.1 as f32))
        }),
    )(i)
}

// color parsing credit: example on https://github.com/rust-bakery/nom/tree/main
fn from_hex(i: &str) -> Result<u8> {
    Ok(u8::from_str_radix(i, 16)?)
}
fn hex_primary(i: &str) -> IResult<&str, u8> {
    map_res(take_while_m_n(2, 2, |c: char| c.is_hex_digit()), from_hex)(i)
}
fn hex_to_color32(i: &str) -> IResult<&str, Color32> {
    let (i, (blue, green, red)) = tuple((hex_primary, hex_primary, hex_primary))(i)?;
    Ok((i, Color32::from_rgb(red, green, blue)))
}
fn c(i: &str) -> IResult<&str, SubtitleField> {
    delimited(
        alt((tag(r"\c&H"), tag(r"\1c&H"))),
        map(hex_to_color32, SubtitleField::PrimaryFill),
        tag("&"),
    )(i)
}
fn undefined(i: &str) -> IResult<&str, SubtitleField> {
    map(
        preceded(char('\\'), take_till(|c| "}\\".contains(c))),
        SubtitleField::Undefined,
    )(i)
}
fn parse_style(i: &str) -> IResult<&str, Subtitle> {
    let (i, subtitle_style_components) = delimited(
        char('{'),
        many0(alt((t, fad, an, pos, c, undefined))),
        tuple((take_until("}"), char('}'))),
    )(i)?;

    let mut subtitle = Subtitle::default();

    for component in subtitle_style_components {
        match component {
            SubtitleField::Fade(fade) => subtitle.fade = fade,
            SubtitleField::Alignment(alignment) => subtitle.alignment = alignment,
            SubtitleField::PrimaryFill(primary_fill) => subtitle.primary_fill = primary_fill,
            SubtitleField::Position(position) => subtitle.position = Some(position),
            SubtitleField::Undefined(_) => (),
        }
    }
    Ok((i, subtitle))
}

fn text_field(i: &str) -> IResult<&str, Subtitle> {
    let (i, (subtitle, subtitle_text)) = preceded(opt_comma, pair(opt(parse_style), rest))(i)?;
    let mut subtitle = subtitle.unwrap_or_default();
    subtitle.text = subtitle_text.replace(r"\N", "\n");
    Ok((i, subtitle))
}

fn not_comma(i: &str) -> IResult<&str, &str> {
    is_not(",")(i)
}
fn comma(i: &str) -> IResult<&str, char> {
    char(',')(i)
}
fn opt_comma(i: &str) -> IResult<&str, Option<char>> {
    opt(comma)(i)
}

fn string_field(i: &str) -> IResult<&str, Option<String>> {
    preceded(opt_comma, map(opt(not_comma), |s| s.map(String::from)))(i)
}

fn num_field(i: &str) -> IResult<&str, i32> {
    preceded(opt_comma, map_res(digit0, str::parse))(i)
}

pub(crate) fn parse_ass_subtitle(i: &str) -> Result<Subtitle> {
    let (_i, (_layer, _start, _style, _name, _margin_l, _margin_r, _margin_v, _effect, subtitle)) =
        tuple((
            context("layer", num_field),
            context("start", num_field),
            context("style", string_field),
            context("name", string_field),
            context("margin_l", num_field),
            context("margin_r", num_field),
            context("margin_v", num_field),
            context("effect", string_field),
            context("style override + text", text_field),
        ))(i)
        .map_err(|e| anyhow!(format!("subtitle parse failed: {e}")))?;

    Ok(subtitle)
}
