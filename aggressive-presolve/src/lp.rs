//! CPLEX LP format reader.
//!
//! Supported: `\` comments; `Minimize`/`Maximize` (and their usual
//! spellings) with an optional objective name and a constant; `Subject To`
//! (`st`, `s.t.`, `such that`) with named or unnamed rows, operators
//! `<= =< < >= => > =`, constants on the left folded into the right-hand
//! side, and double-sided ranges `l <= expr <= u`; `Bounds` with `x >= l`,
//! `x <= u`, `l <= x <= u`, `x = v`, `x free`, `±inf`/`±infinity`;
//! `General`/`Generals`/`Gen`/`Integer`/`Integers`; `Binary`/`Binaries`/
//! `Bin`; `End`. A variable defaults to `[0, +inf)`, a binary to `[0, 1]`.
//!
//! Errors (never guesses): quadratic terms (`[ ... ]`), semi-continuous,
//! SOS, indicator and lazy-constraint sections, and variables on a
//! right-hand side.

use crate::model::{Direction, Model, VarKind};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Section {
    Objective,
    Constraints,
    Bounds,
    General,
    Binary,
    End,
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "!\"#$%&()/,.;?@_`'{}|~".contains(c)
}

fn tokenize(line: &str, out: &mut Vec<Tok>) -> Result<(), String> {
    let b: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '+' {
            out.push(Tok::Plus);
            i += 1;
        } else if c == '-' {
            out.push(Tok::Minus);
            i += 1;
        } else if c == ':' {
            out.push(Tok::Colon);
            i += 1;
        } else if c == '<' || c == '>' || c == '=' {
            let next = b.get(i + 1).copied();
            let (op, len) = match (c, next) {
                ('<', Some('=')) | ('=', Some('<')) => (Cmp::Le, 2),
                ('>', Some('=')) | ('=', Some('>')) => (Cmp::Ge, 2),
                ('<', _) => (Cmp::Le, 1),
                ('>', _) => (Cmp::Ge, 1),
                _ => (Cmp::Eq, 1),
            };
            out.push(Tok::Op(op));
            i += len;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let mut k = i + 1;
                if k < b.len() && (b[k] == '+' || b[k] == '-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    while k < b.len() && b[k].is_ascii_digit() {
                        k += 1;
                    }
                    i = k;
                }
            }
            let s: String = b[start..i].iter().collect();
            let v = s.parse::<f64>().map_err(|_| format!("bad number {:?}", s))?;
            out.push(Tok::Num(v));
        } else if c == '[' {
            return Err("quadratic terms are not supported".into());
        } else if is_name_char(c) {
            let start = i;
            while i < b.len() && is_name_char(b[i]) {
                i += 1;
            }
            let s: String = b[start..i].iter().collect();
            match s.to_ascii_lowercase().as_str() {
                "inf" | "infinity" => out.push(Tok::Num(f64::INFINITY)),
                _ => out.push(Tok::Name(s)),
            }
        } else {
            return Err(format!("unexpected character {:?}", c));
        }
    }
    Ok(())
}

/// A section header at the start of a line, and the rest of the line.
fn header(line: &str) -> Option<(Section, &str)> {
    let t = line.trim_start();
    let lower = t.to_ascii_lowercase();
    let keys: &[(&str, Section)] = &[
        ("minimize", Section::Objective),
        ("minimise", Section::Objective),
        ("minimum", Section::Objective),
        ("min", Section::Objective),
        ("maximize", Section::Objective),
        ("maximise", Section::Objective),
        ("maximum", Section::Objective),
        ("max", Section::Objective),
        ("subject to", Section::Constraints),
        ("such that", Section::Constraints),
        ("s.t.", Section::Constraints),
        ("st", Section::Constraints),
        ("bounds", Section::Bounds),
        ("bound", Section::Bounds),
        ("generals", Section::General),
        ("general", Section::General),
        ("gen", Section::General),
        ("integers", Section::General),
        ("integer", Section::General),
        ("binaries", Section::Binary),
        ("binary", Section::Binary),
        ("bin", Section::Binary),
        ("end", Section::End),
    ];
    for (k, s) in keys {
        if let Some(rest) = lower.strip_prefix(k) {
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                return Some((*s, &t[k.len()..]));
            }
        }
    }
    None
}

fn unsupported(line: &str) -> Option<&'static str> {
    let l = line.trim_start().to_ascii_lowercase();
    for k in ["semi-continuous", "semis", "semi", "sos", "lazy constraints", "user cuts"] {
        if l == k || l.starts_with(&format!("{} ", k)) {
            return Some("unsupported LP section");
        }
    }
    None
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
        let j = self
            .model
            .add_var(name, 0.0, f64::INFINITY, 0.0, VarKind::Continuous);
        self.index.insert(name.to_string(), j);
        j
    }
}

/// Reads a linear expression from `t[*i..]`: terms and constants, stopping at
/// an operator, a `name :` label or the end. Returns the terms and the
/// constant.
fn expr(b: &mut Builder, t: &[Tok], i: &mut usize) -> Result<(Vec<(usize, f64)>, f64), String> {
    let mut terms = Vec::new();
    let mut constant = 0.0;
    loop {
        let start = *i;
        let mut sign = 1.0;
        let mut any_sign = false;
        while *i < t.len() {
            match t[*i] {
                Tok::Plus => {}
                Tok::Minus => sign = -sign,
                _ => break,
            }
            any_sign = true;
            *i += 1;
        }
        let coef = if let Some(Tok::Num(v)) = t.get(*i) {
            *i += 1;
            Some(*v)
        } else {
            None
        };
        match t.get(*i) {
            Some(Tok::Name(n)) if !matches!(t.get(*i + 1), Some(Tok::Colon)) => {
                let j = b.var(n);
                terms.push((j, sign * coef.unwrap_or(1.0)));
                *i += 1;
            }
            _ => match coef {
                Some(v) => constant += sign * v,
                None => {
                    if any_sign {
                        return Err("dangling sign".into());
                    }
                    *i = start;
                    return Ok((terms, constant));
                }
            },
        }
    }
}

/// Parses CPLEX LP text.
pub fn parse(text: &str) -> Result<Model, String> {
    let mut b = Builder {
        model: Model::new(Direction::Minimize),
        index: HashMap::new(),
    };
    let mut obj = Vec::new();
    let mut cons = Vec::new();
    let mut bounds: Vec<Vec<Tok>> = Vec::new();
    let mut ints = Vec::new();
    let mut bins = Vec::new();
    let mut section: Option<Section> = None;
    let mut seen_objective = false;
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('\\').next().unwrap_or("");
        if line.trim().is_empty() {
            continue;
        }
        if unsupported(line).is_some() {
            return Err(format!("line {}: unsupported section {:?}", ln + 1, line.trim()));
        }
        let mut content = line;
        if let Some((s, rest)) = header(line) {
            let is_header = match s {
                // `min`/`max` only start the objective once.
                Section::Objective => !seen_objective,
                _ => true,
            };
            if is_header {
                if s == Section::Objective {
                    seen_objective = true;
                    let l = line.trim_start().to_ascii_lowercase();
                    if l.starts_with("max") {
                        b.model.direction = Direction::Maximize;
                    }
                }
                section = Some(s);
                content = rest;
                if s == Section::End {
                    break;
                }
            }
        }
        let err = |e: String| format!("line {}: {}", ln + 1, e);
        match section {
            None => return Err(err("text before the objective".into())),
            Some(Section::Objective) => tokenize(content, &mut obj).map_err(err)?,
            Some(Section::Constraints) => tokenize(content, &mut cons).map_err(err)?,
            Some(Section::Bounds) => {
                let mut v = Vec::new();
                tokenize(content, &mut v).map_err(err)?;
                if !v.is_empty() {
                    bounds.push(v);
                }
            }
            Some(Section::General) => tokenize(content, &mut ints).map_err(err)?,
            Some(Section::Binary) => tokenize(content, &mut bins).map_err(err)?,
            Some(Section::End) => break,
        }
    }

    // Objective.
    let mut i = 0;
    if let (Some(Tok::Name(_)), Some(Tok::Colon)) = (obj.first(), obj.get(1)) {
        i = 2;
    }
    let (terms, constant) = expr(&mut b, &obj, &mut i)?;
    if i != obj.len() {
        return Err(format!("objective: unexpected {:?}", obj.get(i)));
    }
    for (j, a) in terms {
        b.model.vars[j].obj += a;
    }
    b.model.obj_offset = constant;

    // Constraints.
    let mut i = 0;
    let mut rows: Vec<(String, Vec<(usize, f64)>, f64, f64)> = Vec::new();
    while i < cons.len() {
        let mut name = String::new();
        if let (Some(Tok::Name(n)), Some(Tok::Colon)) = (cons.get(i), cons.get(i + 1)) {
            name = n.clone();
            i += 2;
        }
        let rhs_num = |t: &[Tok], i: &mut usize| -> Result<f64, String> {
            let mut sign = 1.0;
            while let Some(tok @ (Tok::Plus | Tok::Minus)) = t.get(*i) {
                if *tok == Tok::Minus {
                    sign = -sign;
                }
                *i += 1;
            }
            match t.get(*i) {
                Some(Tok::Num(v)) => {
                    *i += 1;
                    Ok(sign * v)
                }
                other => Err(format!("row {:?}: expected a number, got {:?}", name, other)),
            }
        };
        // Ranged: number op expr op number.
        let save = i;
        let lead = rhs_num(&cons, &mut i).ok();
        let ranged_first = match (lead, cons.get(i)) {
            (Some(v), Some(Tok::Op(op))) => Some((v, *op)),
            _ => None,
        };
        if ranged_first.is_none() {
            i = save;
        } else {
            i += 1;
        }
        let (terms, constant) = expr(&mut b, &cons, &mut i)?;
        let op = match cons.get(i) {
            Some(Tok::Op(op)) => *op,
            other => return Err(format!("row {:?}: expected an operator, got {:?}", name, other)),
        };
        i += 1;
        let rhs = rhs_num(&cons, &mut i)? - constant;
        if let Some(Tok::Name(_)) = cons.get(i) {
            if !matches!(cons.get(i + 1), Some(Tok::Colon)) {
                return Err(format!("row {:?}: variables on the right-hand side", name));
            }
        }
        let (mut lo, mut hi) = match op {
            Cmp::Le => (f64::NEG_INFINITY, rhs),
            Cmp::Ge => (rhs, f64::INFINITY),
            Cmp::Eq => (rhs, rhs),
        };
        if let Some((v, op1)) = ranged_first {
            let v = v - constant;
            match (op1, op) {
                (Cmp::Le, Cmp::Le) => lo = v,
                (Cmp::Ge, Cmp::Ge) => hi = v,
                _ => return Err(format!("row {:?}: inconsistent range operators", name)),
            }
        }
        rows.push((name, terms, lo, hi));
    }

    // Bounds.
    for line in &bounds {
        bound_line(&mut b, line)?;
    }
    for t in &ints {
        match t {
            Tok::Name(n) => {
                let j = b.var(n);
                b.model.vars[j].kind = VarKind::Integer;
            }
            other => return Err(format!("General: unexpected {:?}", other)),
        }
    }
    for t in &bins {
        match t {
            Tok::Name(n) => {
                let j = b.var(n);
                let v = &mut b.model.vars[j];
                v.kind = VarKind::Integer;
                v.lo = v.lo.max(0.0);
                v.hi = v.hi.min(1.0);
            }
            other => return Err(format!("Binary: unexpected {:?}", other)),
        }
    }
    for (name, terms, lo, hi) in rows {
        b.model.add_range_row(name, terms, lo, hi);
    }
    b.model.validate()?;
    Ok(b.model)
}

fn bound_line(b: &mut Builder, t: &[Tok]) -> Result<(), String> {
    let num_at = |i: usize| -> Option<(f64, usize)> {
        let mut sign = 1.0;
        let mut k = i;
        while let Some(tok @ (Tok::Plus | Tok::Minus)) = t.get(k) {
            if *tok == Tok::Minus {
                sign = -sign;
            }
            k += 1;
        }
        match t.get(k) {
            Some(Tok::Num(v)) => Some((sign * v, k + 1)),
            _ => None,
        }
    };
    let bad = || format!("bad bound {:?}", t);
    let apply = |b: &mut Builder, j: usize, op: Cmp, v: f64, var_on_left: bool| {
        let op = if var_on_left {
            op
        } else {
            match op {
                Cmp::Le => Cmp::Ge,
                Cmp::Ge => Cmp::Le,
                Cmp::Eq => Cmp::Eq,
            }
        };
        let var = &mut b.model.vars[j];
        match op {
            Cmp::Le => var.hi = v,
            Cmp::Ge => var.lo = v,
            Cmp::Eq => {
                var.lo = v;
                var.hi = v;
            }
        }
    };
    if let Some((v, k)) = num_at(0) {
        // l op x [op u]
        let Some(Tok::Op(op1)) = t.get(k) else {
            return Err(bad());
        };
        let Some(Tok::Name(n)) = t.get(k + 1) else {
            return Err(bad());
        };
        let j = b.var(n);
        apply(b, j, *op1, v, false);
        if let Some(Tok::Op(op2)) = t.get(k + 2) {
            let (u, end) = num_at(k + 3).ok_or_else(bad)?;
            if end != t.len() {
                return Err(bad());
            }
            apply(b, j, *op2, u, true);
        } else if k + 2 != t.len() {
            return Err(bad());
        }
        return Ok(());
    }
    let Some(Tok::Name(n)) = t.first() else {
        return Err(bad());
    };
    let j = b.var(n);
    match t.get(1) {
        Some(Tok::Name(f)) if f.eq_ignore_ascii_case("free") && t.len() == 2 => {
            b.model.vars[j].lo = f64::NEG_INFINITY;
            b.model.vars[j].hi = f64::INFINITY;
        }
        Some(Tok::Op(op)) => {
            let (v, end) = num_at(2).ok_or_else(bad)?;
            if end != t.len() {
                return Err(bad());
            }
            apply(b, j, *op, v, true);
        }
        _ => return Err(bad()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_small_lp() {
        let text = r"\ a comment
Maximize
 obj: 3 x + 2 y - z + 4
Subject To
 c1: x + y + z <= 10
 c2: x - y >= -2.5
 -x + 2 z = 1
 r: 1 <= x + y <= 8
Bounds
 0 <= x <= 4
 y <= 1e1
 z free
 -inf <= w <= 3
General
 x
Binary
 b
End
";
        let m = parse(text).unwrap();
        assert_eq!(m.direction, Direction::Maximize);
        assert_eq!(m.obj_offset, 4.0);
        assert_eq!(m.vars.len(), 5);
        assert_eq!(m.vars[0].kind, VarKind::Integer);
        assert_eq!((m.vars[0].lo, m.vars[0].hi), (0.0, 4.0));
        assert_eq!(m.vars[2].lo, f64::NEG_INFINITY);
        assert_eq!((m.vars[3].lo, m.vars[3].hi), (f64::NEG_INFINITY, 3.0));
        assert_eq!((m.vars[4].lo, m.vars[4].hi, m.vars[4].kind), (0.0, 1.0, VarKind::Integer));
        assert_eq!(m.rows.len(), 4);
        assert_eq!(m.rows[1].lo, -2.5);
        assert_eq!(m.rows[2].terms, vec![(0, -1.0), (2, 2.0)]);
        assert_eq!((m.rows[2].lo, m.rows[2].hi), (1.0, 1.0));
        assert_eq!((m.rows[3].lo, m.rows[3].hi), (1.0, 8.0));
    }

    #[test]
    fn multi_line_rows_and_constants() {
        let text = "Minimize\n x + y\n + 2 z\nst\n c1: x + y\n  + z + 3 >= 5\n c2: 2 x\n = 4\nend\n";
        let m = parse(text).unwrap();
        assert_eq!(m.vars[2].obj, 2.0);
        assert_eq!(m.rows[0].lo, 2.0);
        assert_eq!(m.rows[1].terms, vec![(0, 2.0)]);
    }
}
