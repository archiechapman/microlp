//! CPLEX LP format reader: objective (`Minimize`/`Maximize`), `Subject To`,
//! `Bounds`, `General`/`Integer`, `Binary`, `End`. Comments start with `\`.
//! Constraints are `[name:] expr op rhs` or `[name:] lo op expr op hi`
//! (ranged); bounds are `x op v`, `v op x`, `lo <= x <= hi`, `x free`, and
//! `x = v`. Variables default to `[0, +inf)`; a `General` var keeps its
//! bounds, a `Binary` var gets `[0, 1]`. Semi-continuous and SOS sections are
//! errors.

use crate::model::{Direction, Domain, Model, Var};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Id(String),
    Op(Cmp),
    Plus,
    Minus,
    Colon,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cmp {
    Le,
    Ge,
    Eq,
}

#[derive(Clone, Copy, PartialEq)]
enum Sec {
    Obj,
    Cons,
    Bounds,
    Gen,
    Bin,
    End,
}

fn section_of(line: &str) -> Option<(Sec, Option<Direction>, usize)> {
    let lower = line.trim_start().to_ascii_lowercase();
    let first: String = lower.split_whitespace().next().unwrap_or("").to_string();
    let two: String = lower
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    let skip = |n: usize| line.len() - line.trim_start().len() + n;
    match first.as_str() {
        "minimize" | "minimise" | "minimum" | "min" => {
            Some((Sec::Obj, Some(Direction::Minimize), skip(first.len())))
        }
        "maximize" | "maximise" | "maximum" | "max" => {
            Some((Sec::Obj, Some(Direction::Maximize), skip(first.len())))
        }
        "st" | "s.t." | "st." => Some((Sec::Cons, None, skip(first.len()))),
        "bounds" | "bound" => Some((Sec::Bounds, None, skip(first.len()))),
        "general" | "generals" | "gen" | "integer" | "integers" => {
            Some((Sec::Gen, None, skip(first.len())))
        }
        "binary" | "binaries" | "bin" => Some((Sec::Bin, None, skip(first.len()))),
        "end" => Some((Sec::End, None, skip(first.len()))),
        _ if two == "subject to" || two == "such that" => {
            let n = line
                .to_ascii_lowercase()
                .find(" to")
                .or(line.to_ascii_lowercase().find(" that"));
            let n = n.map(|i| {
                if line[i..].to_ascii_lowercase().starts_with(" to") {
                    i + 3
                } else {
                    i + 5
                }
            });
            Some((Sec::Cons, None, n.unwrap_or(line.len())))
        }
        _ => None,
    }
}

fn tokenize(s: &str, out: &mut Vec<Tok>) -> Result<(), String> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            ':' => {
                out.push(Tok::Colon);
                i += 1;
            }
            '<' | '>' | '=' => {
                let mut j = i + 1;
                while j < b.len() && matches!(b[j] as char, '<' | '>' | '=') {
                    j += 1;
                }
                let op = match &s[i..j] {
                    "<" | "<=" | "=<" => Cmp::Le,
                    ">" | ">=" | "=>" => Cmp::Ge,
                    "=" | "==" => Cmp::Eq,
                    o => return Err(format!("unknown operator {o}")),
                };
                out.push(Tok::Op(op));
                i = j;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let mut j = i;
                while j < b.len() && ((b[j] as char).is_ascii_digit() || b[j] == b'.') {
                    j += 1;
                }
                if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
                    let mut k = j + 1;
                    if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                        k += 1;
                    }
                    if k < b.len() && (b[k] as char).is_ascii_digit() {
                        while k < b.len() && (b[k] as char).is_ascii_digit() {
                            k += 1;
                        }
                        j = k;
                    }
                }
                let v: f64 = s[i..j]
                    .parse()
                    .map_err(|_| format!("bad number {}", &s[i..j]))?;
                out.push(Tok::Num(v));
                i = j;
            }
            _ => {
                let mut j = i;
                while j < b.len() {
                    let d = b[j] as char;
                    if d.is_whitespace() || matches!(d, '+' | '-' | ':' | '<' | '>' | '=') {
                        break;
                    }
                    j += 1;
                }
                let id = &s[i..j];
                let lower = id.to_ascii_lowercase();
                if lower == "inf" || lower == "infinity" {
                    out.push(Tok::Num(f64::INFINITY));
                } else {
                    out.push(Tok::Id(id.to_string()));
                }
                i = j;
            }
        }
    }
    Ok(())
}

struct Builder {
    model: Model,
    index: HashMap<String, usize>,
}

impl Builder {
    fn var(&mut self, name: &str) -> usize {
        if let Some(&j) = self.index.get(name) {
            return j;
        }
        let j = self.model.vars.len();
        self.model.vars.push(Var {
            name: name.to_string(),
            lo: 0.0,
            hi: f64::INFINITY,
            cost: 0.0,
            domain: Domain::Real,
        });
        self.index.insert(name.to_string(), j);
        j
    }
}

/// A linear expression `Σ a·x + constant` read from `toks[*pos..]`, stopping
/// at an operator or the end.
fn expr(
    toks: &[Tok],
    pos: &mut usize,
    b: &mut Builder,
) -> Result<(Vec<(usize, f64)>, f64), String> {
    let mut terms = Vec::new();
    let mut constant = 0.0;
    let mut sign = 1.0;
    let mut coef: Option<f64> = None;
    while *pos < toks.len() {
        match &toks[*pos] {
            Tok::Plus => {
                if let Some(c) = coef.take() {
                    constant += sign * c;
                    sign = 1.0;
                }
            }
            Tok::Minus => {
                if let Some(c) = coef.take() {
                    constant += sign * c;
                    sign = -1.0;
                } else {
                    sign = -sign;
                }
            }
            Tok::Num(v) => {
                if let Some(c) = coef {
                    // "2 3 x" is not valid; treat as product for robustness.
                    coef = Some(c * v);
                } else {
                    coef = Some(*v);
                }
            }
            Tok::Id(name) => {
                // A label "name:" starts the next constraint.
                if matches!(toks.get(*pos + 1), Some(Tok::Colon)) {
                    break;
                }
                let j = b.var(name);
                terms.push((j, sign * coef.take().unwrap_or(1.0)));
                sign = 1.0;
            }
            Tok::Op(_) | Tok::Colon => break,
        }
        *pos += 1;
    }
    if let Some(c) = coef {
        constant += sign * c;
    }
    Ok((terms, constant))
}

fn signed_num(toks: &[Tok], pos: &mut usize) -> Result<f64, String> {
    let mut sign = 1.0;
    loop {
        match toks.get(*pos) {
            Some(Tok::Minus) => sign = -sign,
            Some(Tok::Plus) => {}
            Some(Tok::Num(v)) => {
                *pos += 1;
                return Ok(sign * v);
            }
            t => return Err(format!("expected a number, found {t:?}")),
        }
        *pos += 1;
    }
}

fn op_at(toks: &[Tok], pos: usize) -> Option<Cmp> {
    match toks.get(pos) {
        Some(Tok::Op(c)) => Some(*c),
        _ => None,
    }
}

fn sides(op: Cmp, v: f64) -> (f64, f64) {
    match op {
        Cmp::Le => (f64::NEG_INFINITY, v),
        Cmp::Ge => (v, f64::INFINITY),
        Cmp::Eq => (v, v),
    }
}

fn flip(op: Cmp) -> Cmp {
    match op {
        Cmp::Le => Cmp::Ge,
        Cmp::Ge => Cmp::Le,
        Cmp::Eq => Cmp::Eq,
    }
}

/// Parses CPLEX LP text.
pub fn parse(text: &str) -> Result<Model, String> {
    let mut b = Builder {
        model: Model::new(Direction::Minimize),
        index: HashMap::new(),
    };
    let mut sections: Vec<(Sec, Vec<Tok>)> = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = match raw.find('\\') {
            Some(i) => &raw[..i],
            None => raw,
        };
        if line.trim().is_empty() {
            continue;
        }
        let mut rest = line;
        if let Some((sec, dir, skip)) = section_of(line) {
            if let Some(d) = dir {
                b.model.direction = d;
            }
            if sec == Sec::End {
                break;
            }
            sections.push((sec, Vec::new()));
            rest = &line[skip.min(line.len())..];
        } else {
            let low = line.trim_start().to_ascii_lowercase();
            if low.starts_with("semi") || low.starts_with("sos") {
                return Err(format!("line {}: unsupported section", ln + 1));
            }
        }
        let Some(cur) = sections.last_mut() else {
            return Err(format!("line {}: text before the objective", ln + 1));
        };
        tokenize(rest, &mut cur.1).map_err(|e| format!("line {}: {e}", ln + 1))?;
    }
    let mut binaries = Vec::new();
    for (sec, toks) in &sections {
        let mut pos = 0;
        match sec {
            Sec::Obj => {
                if toks.len() >= 2 && matches!(toks[1], Tok::Colon) {
                    pos = 2;
                }
                let (terms, constant) = expr(toks, &mut pos, &mut b)?;
                for (j, a) in terms {
                    b.model.vars[j].cost += a;
                }
                b.model.obj_offset += constant;
            }
            Sec::Cons => {
                while pos < toks.len() {
                    let mut name = None;
                    if let (Some(Tok::Id(n)), Some(Tok::Colon)) = (toks.get(pos), toks.get(pos + 1))
                    {
                        name = Some(n.clone());
                        pos += 2;
                    }
                    let start = pos;
                    let (terms, constant) = expr(toks, &mut pos, &mut b)?;
                    let op = op_at(toks, pos).ok_or("constraint without an operator")?;
                    pos += 1;
                    let (lo, hi, terms) = if terms.is_empty() {
                        // Ranged: lo op expr op hi.
                        let left = constant;
                        let (t2, c2) = expr(toks, &mut pos, &mut b)?;
                        let op2 =
                            op_at(toks, pos).ok_or("ranged constraint needs two operators")?;
                        pos += 1;
                        let right = signed_num(toks, &mut pos)?;
                        let (l1, h1) = sides(flip(op), left - c2);
                        let (l2, h2) = sides(op2, right - c2);
                        let _ = start;
                        (l1.max(l2), h1.min(h2), t2)
                    } else {
                        let v = signed_num(toks, &mut pos)? - constant;
                        let (l, h) = sides(op, v);
                        (l, h, terms)
                    };
                    let i = b.model.add_row(&terms, lo, hi);
                    if let Some(n) = name {
                        b.model.rows[i].name = n;
                    }
                }
            }
            Sec::Bounds => {
                while pos < toks.len() {
                    // v op x [op v] | x op v | x free
                    let lead = match toks.get(pos) {
                        Some(Tok::Num(_)) | Some(Tok::Minus) | Some(Tok::Plus) => {
                            let v = signed_num(toks, &mut pos)?;
                            let op = op_at(toks, pos).ok_or("bound without an operator")?;
                            pos += 1;
                            Some((v, op))
                        }
                        _ => None,
                    };
                    let Some(Tok::Id(name)) = toks.get(pos) else {
                        return Err(format!("bad bound near token {pos}"));
                    };
                    let j = b.var(name);
                    pos += 1;
                    let var = &mut b.model.vars[j];
                    if let Some((v, op)) = lead {
                        let (l, h) = sides(flip(op), v);
                        if op == Cmp::Eq {
                            var.lo = v;
                            var.hi = v;
                        } else {
                            if l > f64::NEG_INFINITY
                                || v == f64::NEG_INFINITY && flip(op) == Cmp::Ge
                            {
                                var.lo = if v == f64::NEG_INFINITY { v } else { l };
                            }
                            if h < f64::INFINITY {
                                var.hi = h;
                            }
                        }
                    }
                    match toks.get(pos) {
                        Some(Tok::Op(op)) => {
                            let op = *op;
                            pos += 1;
                            let v = signed_num(toks, &mut pos)?;
                            let var = &mut b.model.vars[j];
                            match op {
                                Cmp::Le => var.hi = v,
                                Cmp::Ge => var.lo = v,
                                Cmp::Eq => {
                                    var.lo = v;
                                    var.hi = v;
                                }
                            }
                        }
                        Some(Tok::Id(w)) if w.eq_ignore_ascii_case("free") => {
                            pos += 1;
                            let var = &mut b.model.vars[j];
                            var.lo = f64::NEG_INFINITY;
                            var.hi = f64::INFINITY;
                        }
                        _ => {}
                    }
                }
            }
            Sec::Gen | Sec::Bin => {
                for t in toks {
                    let Tok::Id(name) = t else {
                        return Err(format!("unexpected token {t:?} in an integer section"));
                    };
                    let j = b.var(name);
                    b.model.vars[j].domain = Domain::Integer;
                    if *sec == Sec::Bin {
                        binaries.push(j);
                    }
                }
            }
            Sec::End => {}
        }
    }
    for j in binaries {
        let v = &mut b.model.vars[j];
        v.lo = v.lo.max(0.0);
        v.hi = v.hi.min(1.0);
    }
    Ok(b.model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_small_model() {
        let text = "\\ comment\nMaximize\n obj: 3 x + 2y - z + 4\nSubject To\n c1: x + y <= 4\n \
                    c2: -2 <= x - z <= 5\n c3: 2 x + 3 y\n >= 1\nBounds\n -inf <= z <= 10\n y <= 3\n \
                    w free\nGeneral\n y\nBinary\n b\nEnd\n";
        let m = parse(text).unwrap();
        assert_eq!(m.direction, Direction::Maximize);
        assert_eq!(m.obj_offset, 4.0);
        assert_eq!(m.vars.len(), 5);
        assert_eq!(m.vars[0].cost, 3.0);
        assert_eq!(m.vars[1].cost, 2.0);
        assert_eq!((m.vars[2].lo, m.vars[2].hi), (f64::NEG_INFINITY, 10.0));
        assert_eq!(m.vars[1].domain, Domain::Integer);
        assert_eq!(
            (m.vars[3].lo, m.vars[3].hi),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!((m.vars[4].lo, m.vars[4].hi), (0.0, 1.0));
        assert_eq!((m.rows[1].lo, m.rows[1].hi), (-2.0, 5.0));
        assert_eq!((m.rows[2].lo, m.rows[2].hi), (1.0, f64::INFINITY));
        assert_eq!(m.rows[0].name, "c1");
    }
}
