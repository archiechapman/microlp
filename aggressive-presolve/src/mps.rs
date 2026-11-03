//! MPS reader, fixed and free.
//!
//! Sections: NAME, OBJSENSE (own line or inline; MIN/MAX/MINIMIZE/MAXIMIZE),
//! ROWS (N/L/G/E; the first N row is the objective, entries on later N rows
//! are dropped), COLUMNS with `'MARKER' 'INTORG'`/`'INTEND'`, RHS (an entry
//! on the objective row is the *negated* objective constant), RANGES, BOUNDS
//! (UP/LO/FX/FR/MI/PL/BV/LI/UI), ENDATA. Only the first RHS, RANGES and
//! BOUNDS vector is honoured. Conventions:
//!
//! * an integer column with no bound entry is binary when
//!   [`MpsOptions::int_default_binary`] (the MIPLIB 3 / glpk convention, the
//!   default), else `[0, +inf)`;
//! * a negative `UP` (or `UI`) on a column without a `LO` entry sets the lower
//!   bound to `-inf`, with a warning (the classic MPSX rule);
//! * RANGES: E row `R > 0` gives `[b, b+|R|]`, `R < 0` gives `[b-|R|, b]`;
//!   L row `[b-|R|, b]`; G row `[b, b+|R|]`.
//!
//! SOS, quadratic and other extension sections and `SC` bounds are errors.

use crate::model::{Direction, Domain, Model, Row, Var};
use std::collections::HashMap;

/// How data lines are split into fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpsFormat {
    /// Whitespace-separated fields (names without spaces).
    Free,
    /// Fixed columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61 (names may hold
    /// spaces).
    Fixed,
    /// [`MpsFormat::Free`]; reads every fixed file whose names have no spaces.
    Auto,
}

#[derive(Clone, Debug)]
pub struct MpsOptions {
    pub format: MpsFormat,
    pub int_default_binary: bool,
}

impl Default for MpsOptions {
    fn default() -> Self {
        MpsOptions {
            format: MpsFormat::Auto,
            int_default_binary: true,
        }
    }
}

/// A model read from MPS, with the reader's warnings.
#[derive(Clone, Debug)]
pub struct MpsRead {
    pub model: Model,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    ObjSense,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowKind {
    N,
    L,
    G,
    E,
}

struct ColInfo {
    integer: bool,
    lo: f64,
    hi: f64,
    had_bound: bool,
    had_lo: bool,
}

fn number(s: &str, line: usize) -> Result<f64, String> {
    let t = s.replace(['D', 'd'], "E");
    t.parse::<f64>()
        .map_err(|_| format!("line {line}: '{s}' is not a number"))
        .and_then(|v| {
            if v.is_nan() {
                Err(format!("line {line}: NaN"))
            } else {
                Ok(v)
            }
        })
}

fn fixed_fields(line: &str) -> Vec<String> {
    const SPANS: [(usize, usize); 6] = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)];
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    for (i, &(a, b)) in SPANS.iter().enumerate() {
        if a >= chars.len() {
            break;
        }
        let end = if i == SPANS.len() - 1 {
            chars.len()
        } else {
            b.min(chars.len())
        };
        let f: String = chars[a..end].iter().collect();
        out.push(f.trim().to_string());
    }
    // Drop trailing empties; keep interior empties only for field 1.
    while out.last().is_some_and(|f| f.is_empty()) {
        out.pop();
    }
    out
}

/// Parses MPS text.
pub fn parse(text: &str, opts: &MpsOptions) -> Result<MpsRead, String> {
    let mut warnings = Vec::new();
    let mut model = Model::new(Direction::Minimize);
    let mut section = Section::None;
    let mut obj_row: Option<String> = None;
    let mut row_index: HashMap<String, (RowKind, usize)> = HashMap::new();
    let mut row_kinds: Vec<RowKind> = Vec::new();
    let mut row_terms: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut ranges: Vec<Option<f64>> = Vec::new();
    let mut col_index: HashMap<String, usize> = HashMap::new();
    let mut cols: Vec<ColInfo> = Vec::new();
    let mut costs: Vec<f64> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();
    let mut row_names: Vec<String> = Vec::new();
    let mut in_int = false;
    let mut rhs_set: Option<String> = None;
    let mut range_set: Option<String> = None;
    let mut bound_set: Option<String> = None;
    let mut ended = false;
    let mut dropped_n = 0usize;

    for (ln0, raw) in text.lines().enumerate() {
        let ln = ln0 + 1;
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() || line.starts_with('*') {
            continue;
        }
        let is_header = !line.starts_with([' ', '\t']);
        if is_header {
            let mut it = line.split_whitespace();
            let head = it.next().unwrap().to_ascii_uppercase();
            let rest: Vec<&str> = it.collect();
            section = match head.as_str() {
                "NAME" => {
                    model.name = rest.join(" ");
                    Section::None
                }
                "OBJSENSE" => {
                    if let Some(s) = rest.first() {
                        model.direction = sense(s, ln)?;
                        Section::None
                    } else {
                        Section::ObjSense
                    }
                }
                "OBJSENSE_MAX" | "MAX" | "MAXIMIZE" if section == Section::ObjSense => {
                    model.direction = Direction::Maximize;
                    Section::None
                }
                "MIN" | "MINIMIZE" if section == Section::ObjSense => {
                    model.direction = Direction::Minimize;
                    Section::None
                }
                "ROWS" => Section::Rows,
                "COLUMNS" => Section::Columns,
                "RHS" => Section::Rhs,
                "RANGES" => Section::Ranges,
                "BOUNDS" => Section::Bounds,
                "ENDATA" => {
                    ended = true;
                    break;
                }
                "OBJNAME" => {
                    // Free MPS: the objective row's name.
                    if let Some(s) = rest.first() {
                        obj_row = Some(s.to_string());
                    }
                    Section::None
                }
                other => return Err(format!("line {ln}: unsupported section {other}")),
            };
            continue;
        }
        let fields: Vec<String> = match opts.format {
            MpsFormat::Fixed => fixed_fields(line),
            _ => line.split_whitespace().map(str::to_string).collect(),
        };
        let f: Vec<&str> = fields.iter().map(String::as_str).collect();
        match section {
            Section::None => return Err(format!("line {ln}: data outside a section")),
            Section::ObjSense => {
                model.direction = sense(line.trim(), ln)?;
            }
            Section::Rows => {
                let (kind, name) = match (opts.format, f.as_slice()) {
                    (MpsFormat::Fixed, [k, n, ..]) => (*k, *n),
                    (_, [k, n]) => (*k, *n),
                    _ => return Err(format!("line {ln}: bad ROWS line")),
                };
                let kind = match kind.to_ascii_uppercase().as_str() {
                    "N" => RowKind::N,
                    "L" => RowKind::L,
                    "G" => RowKind::G,
                    "E" => RowKind::E,
                    k => return Err(format!("line {ln}: unknown row type {k}")),
                };
                if row_index.contains_key(name) {
                    return Err(format!("line {ln}: duplicate row {name}"));
                }
                if kind == RowKind::N {
                    match &obj_row {
                        None => obj_row = Some(name.to_string()),
                        Some(o) if o == name => {}
                        Some(_) => dropped_n += 1,
                    }
                    row_index.insert(name.to_string(), (RowKind::N, usize::MAX));
                } else {
                    row_index.insert(name.to_string(), (kind, row_kinds.len()));
                    row_kinds.push(kind);
                    row_terms.push(Vec::new());
                    rhs.push(0.0);
                    ranges.push(None);
                    row_names.push(name.to_string());
                }
            }
            Section::Columns => {
                let ws: Vec<&str> = line.split_whitespace().collect();
                if ws.len() >= 3 && ws[1].trim_matches('\'').eq_ignore_ascii_case("MARKER") {
                    let m = ws[2].trim_matches('\'').to_ascii_uppercase();
                    match m.as_str() {
                        "INTORG" => in_int = true,
                        "INTEND" => in_int = false,
                        _ => return Err(format!("line {ln}: unknown marker {m}")),
                    }
                    continue;
                }
                let f = if opts.format == MpsFormat::Fixed && f.first() == Some(&"") {
                    &f[1..]
                } else {
                    &f[..]
                };
                if f.len() != 3 && f.len() != 5 {
                    return Err(format!("line {ln}: bad COLUMNS line"));
                }
                let name = f[0];
                let j = match col_index.get(name) {
                    Some(&j) => j,
                    None => {
                        let j = cols.len();
                        col_index.insert(name.to_string(), j);
                        cols.push(ColInfo {
                            integer: in_int,
                            lo: 0.0,
                            hi: f64::INFINITY,
                            had_bound: false,
                            had_lo: false,
                        });
                        costs.push(0.0);
                        col_names.push(name.to_string());
                        j
                    }
                };
                for pair in f[1..].chunks(2) {
                    let (row, val) = (pair[0], number(pair[1], ln)?);
                    match row_index.get(row) {
                        None => return Err(format!("line {ln}: unknown row {row}")),
                        Some(&(RowKind::N, _)) => {
                            if obj_row.as_deref() == Some(row) {
                                costs[j] += val;
                            }
                        }
                        Some(&(_, i)) => row_terms[i].push((j, val)),
                    }
                }
            }
            Section::Rhs | Section::Ranges => {
                let f = if opts.format == MpsFormat::Fixed && !f.is_empty() {
                    &f[1..]
                } else {
                    &f[..]
                };
                // A set name is present when the count of fields is odd (free)
                // or always in field 1 (fixed, possibly empty).
                let (set, pairs) =
                    if opts.format == MpsFormat::Fixed && !f.is_empty() || f.len() % 2 == 1 {
                        (f[0], &f[1..])
                    } else {
                        ("", f)
                    };
                if pairs.is_empty() || pairs.len() % 2 != 0 {
                    return Err(format!("line {ln}: bad RHS/RANGES line"));
                }
                let first = if section == Section::Rhs {
                    &mut rhs_set
                } else {
                    &mut range_set
                };
                match first {
                    None => *first = Some(set.to_string()),
                    Some(s) if s == set => {}
                    Some(_) => continue,
                }
                for pair in pairs.chunks(2) {
                    let (row, val) = (pair[0], number(pair[1], ln)?);
                    match row_index.get(row) {
                        None => return Err(format!("line {ln}: unknown row {row}")),
                        Some(&(RowKind::N, _)) => {
                            if section == Section::Rhs && obj_row.as_deref() == Some(row) {
                                model.obj_offset = -val;
                            } else if section == Section::Ranges {
                                return Err(format!("line {ln}: RANGES on a free row"));
                            }
                        }
                        Some(&(_, i)) => {
                            if section == Section::Rhs {
                                rhs[i] = val;
                            } else {
                                ranges[i] = Some(val);
                            }
                        }
                    }
                }
            }
            Section::Bounds => {
                let kind = f[0].to_ascii_uppercase();
                let needs_value = matches!(kind.as_str(), "UP" | "LO" | "FX" | "LI" | "UI");
                let (set, col, val) = if opts.format == MpsFormat::Fixed {
                    (
                        f.get(1).copied().unwrap_or(""),
                        f.get(2).copied().unwrap_or(""),
                        f.get(3).copied(),
                    )
                } else {
                    match (f.len(), needs_value) {
                        (4, _) => (f[1], f[2], Some(f[3])),
                        (3, true) => ("", f[1], Some(f[2])),
                        (3, false) => {
                            if kind == "BV"
                                && col_index.contains_key(f[1])
                                && number(f[2], ln).is_ok()
                            {
                                ("", f[1], Some(f[2]))
                            } else {
                                (f[1], f[2], None)
                            }
                        }
                        (2, false) => ("", f[1], None),
                        _ => return Err(format!("line {ln}: bad BOUNDS line")),
                    }
                };
                match &bound_set {
                    None => bound_set = Some(set.to_string()),
                    Some(s) if s == set => {}
                    Some(_) => continue,
                }
                let Some(&j) = col_index.get(col) else {
                    return Err(format!("line {ln}: unknown column {col}"));
                };
                let v = match val {
                    Some(v) => Some(number(v, ln)?),
                    None if needs_value => return Err(format!("line {ln}: bound needs a value")),
                    None => None,
                };
                let c = &mut cols[j];
                c.had_bound = true;
                match kind.as_str() {
                    "UP" | "UI" => {
                        let v = v.unwrap();
                        if kind == "UI" {
                            c.integer = true;
                        }
                        c.hi = v;
                        if v < 0.0 && !c.had_lo && c.lo == 0.0 {
                            c.lo = f64::NEG_INFINITY;
                            warnings.push(format!(
                                "line {ln}: negative UP bound on {col} with no LO: lower bound set to -inf"
                            ));
                        }
                    }
                    "LO" | "LI" => {
                        if kind == "LI" {
                            c.integer = true;
                        }
                        c.lo = v.unwrap();
                        c.had_lo = true;
                    }
                    "FX" => {
                        c.lo = v.unwrap();
                        c.hi = c.lo;
                        c.had_lo = true;
                    }
                    "FR" => {
                        c.lo = f64::NEG_INFINITY;
                        c.hi = f64::INFINITY;
                        c.had_lo = true;
                    }
                    "MI" => {
                        c.lo = f64::NEG_INFINITY;
                        c.had_lo = true;
                    }
                    "PL" => c.hi = f64::INFINITY,
                    "BV" => {
                        c.integer = true;
                        c.lo = 0.0;
                        c.hi = 1.0;
                        c.had_lo = true;
                    }
                    k => return Err(format!("line {ln}: unsupported bound type {k}")),
                }
            }
        }
    }
    if !ended {
        warnings.push("no ENDATA".to_string());
    }
    if dropped_n > 0 {
        warnings.push(format!("{dropped_n} extra free rows dropped"));
    }
    for (j, c) in cols.iter().enumerate() {
        let (lo, hi) = if c.integer && !c.had_bound && opts.int_default_binary {
            (0.0, 1.0)
        } else {
            (c.lo, c.hi)
        };
        model.vars.push(Var {
            name: col_names[j].clone(),
            lo,
            hi,
            cost: costs[j],
            domain: if c.integer {
                Domain::Integer
            } else {
                Domain::Real
            },
        });
    }
    for (i, kind) in row_kinds.iter().enumerate() {
        let b = rhs[i];
        let (mut lo, mut hi) = match kind {
            RowKind::E => (b, b),
            RowKind::L => (f64::NEG_INFINITY, b),
            RowKind::G => (b, f64::INFINITY),
            RowKind::N => unreachable!(),
        };
        if let Some(r) = ranges[i] {
            match kind {
                RowKind::E if r > 0.0 => hi = b + r,
                RowKind::E if r < 0.0 => lo = b + r,
                RowKind::E => {}
                RowKind::L => lo = b - r.abs(),
                RowKind::G => hi = b + r.abs(),
                RowKind::N => unreachable!(),
            }
        }
        let idx = model.add_row(&row_terms[i], lo, hi);
        let row: &mut Row = &mut model.rows[idx];
        row.name = row_names[i].clone();
    }
    Ok(MpsRead { model, warnings })
}

fn sense(s: &str, ln: usize) -> Result<Direction, String> {
    match s.to_ascii_uppercase().as_str() {
        "MAX" | "MAXIMIZE" | "MAXIMISE" => Ok(Direction::Maximize),
        "MIN" | "MINIMIZE" | "MINIMISE" => Ok(Direction::Minimize),
        o => Err(format!("line {ln}: unknown OBJSENSE {o}")),
    }
}

/// Writes `model` as free MPS: integer columns inside `INTORG`/`INTEND`
/// markers with explicit bounds, every number in the shortest form that reads
/// back exactly. Names are replaced by `C<j>`/`R<i>` where they hold spaces.
pub fn write(model: &Model) -> String {
    use std::fmt::Write as _;
    let clean = |s: &str, prefix: char, i: usize| {
        if s.is_empty() || s.contains(char::is_whitespace) {
            format!("{prefix}{i}")
        } else {
            s.to_string()
        }
    };
    let cols: Vec<String> = model
        .vars
        .iter()
        .enumerate()
        .map(|(j, v)| clean(&v.name, 'C', j))
        .collect();
    let rows: Vec<String> = model
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| clean(&r.name, 'R', i))
        .collect();
    let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); model.vars.len()];
    for (i, r) in model.rows.iter().enumerate() {
        for &(j, a) in &r.terms {
            by_col[j].push((i, a));
        }
    }
    let mut s = String::new();
    let _ = writeln!(
        s,
        "NAME {}",
        if model.name.is_empty() {
            "model"
        } else {
            &model.name
        }
    );
    if model.direction == Direction::Maximize {
        let _ = writeln!(s, "OBJSENSE\n    MAX");
    }
    let _ = writeln!(s, "ROWS\n N  OBJ");
    for (i, r) in model.rows.iter().enumerate() {
        let t = if r.lo == r.hi {
            "E"
        } else if r.lo == f64::NEG_INFINITY {
            "L"
        } else {
            "G"
        };
        let _ = writeln!(s, " {t}  {}", rows[i]);
    }
    let _ = writeln!(s, "COLUMNS");
    let mut in_int = false;
    for (j, v) in model.vars.iter().enumerate() {
        if v.is_integer() != in_int {
            in_int = v.is_integer();
            let m = if in_int { "INTORG" } else { "INTEND" };
            let _ = writeln!(s, "    MARKER 'MARKER' '{m}'");
        }
        if v.cost != 0.0 || by_col[j].is_empty() {
            let _ = writeln!(s, "    {} OBJ {:?}", cols[j], v.cost);
        }
        for &(i, a) in &by_col[j] {
            let _ = writeln!(s, "    {} {} {:?}", cols[j], rows[i], a);
        }
    }
    if in_int {
        let _ = writeln!(s, "    MARKER 'MARKER' 'INTEND'");
    }
    let _ = writeln!(s, "RHS");
    if model.obj_offset != 0.0 {
        let _ = writeln!(s, "    RHS OBJ {:?}", -model.obj_offset);
    }
    for (i, r) in model.rows.iter().enumerate() {
        let b = if r.lo == f64::NEG_INFINITY {
            r.hi
        } else {
            r.lo
        };
        if b != 0.0 {
            let _ = writeln!(s, "    RHS {} {:?}", rows[i], b);
        }
    }
    let _ = writeln!(s, "RANGES");
    for (i, r) in model.rows.iter().enumerate() {
        if r.lo != r.hi && r.lo.is_finite() && r.hi.is_finite() {
            let _ = writeln!(s, "    RNG {} {:?}", rows[i], r.hi - r.lo);
        }
    }
    let _ = writeln!(s, "BOUNDS");
    for (j, v) in model.vars.iter().enumerate() {
        let c = &cols[j];
        if v.lo == v.hi {
            let _ = writeln!(s, " FX BND {c} {:?}", v.lo);
            continue;
        }
        match (v.lo == f64::NEG_INFINITY, v.hi == f64::INFINITY) {
            (true, true) => {
                let _ = writeln!(s, " FR BND {c}");
            }
            (true, false) => {
                let _ = writeln!(s, " MI BND {c}");
                let _ = writeln!(s, " UP BND {c} {:?}", v.hi);
            }
            (false, inf_hi) => {
                // Always explicit, so an integer column is never read binary
                // by default and a negative UP never moves the lower bound.
                let _ = writeln!(s, " LO BND {c} {:?}", v.lo);
                if inf_hi {
                    let _ = writeln!(s, " PL BND {c}");
                } else {
                    let _ = writeln!(s, " UP BND {c} {:?}", v.hi);
                }
            }
        }
    }
    let _ = writeln!(s, "ENDATA");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = "\
NAME          TEST
OBJSENSE
    MAX
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
 E  RNG
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0
    MARKER                 'MARKER'                 'INTEND'
    X2        COST         2.0   LIM1         1.0
    X2        MYEQN       -1.0   RNG          1.0
    X3        COST        -1.0   MYEQN        1.0
    X4        COST         1.0   LIM2         1.0
RHS
    RHS       COST        -3.5
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0   RNG          2.0
RANGES
    RNG       RNG         -1.5   LIM1         2.5
BOUNDS
 UP BND       X1           4.0
 MI BND       X2
 UP BND       X3          -2.0
 FR BND       X4
ENDATA
";

    #[test]
    fn reads_every_convention() {
        let fixed = to_fixed(SMALL);
        for (text, format) in [(SMALL, MpsFormat::Free), (fixed.as_str(), MpsFormat::Fixed)] {
            let r = parse(
                text,
                &MpsOptions {
                    format,
                    ..MpsOptions::default()
                },
            )
            .unwrap();
            let m = &r.model;
            assert_eq!(m.direction, Direction::Maximize);
            assert_eq!(m.obj_offset, 3.5);
            assert_eq!(m.vars.len(), 4);
            assert_eq!(m.vars[0].domain, Domain::Integer);
            assert_eq!((m.vars[0].lo, m.vars[0].hi), (0.0, 4.0));
            assert_eq!(
                (m.vars[1].lo, m.vars[1].hi),
                (f64::NEG_INFINITY, f64::INFINITY)
            );
            assert_eq!((m.vars[2].lo, m.vars[2].hi), (f64::NEG_INFINITY, -2.0));
            assert_eq!(r.warnings.len(), 1);
            assert_eq!((m.rows[0].lo, m.rows[0].hi), (1.5, 4.0));
            assert_eq!((m.rows[1].lo, m.rows[1].hi), (1.0, f64::INFINITY));
            assert_eq!((m.rows[2].lo, m.rows[2].hi), (7.0, 7.0));
            assert_eq!((m.rows[3].lo, m.rows[3].hi), (0.5, 2.0));
            assert_eq!(m.rows[2].terms, vec![(1, -1.0), (2, 1.0)]);
        }
    }

    /// Rewrites whitespace-separated data lines in fixed columns, with a space
    /// inside every column name to show that fixed fields keep it.
    fn to_fixed(text: &str) -> String {
        let mut out = String::new();
        let mut section = "";
        for line in text.lines() {
            if !line.starts_with(' ') {
                section = line.split_whitespace().next().unwrap_or("");
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let t: Vec<String> = line
                .split_whitespace()
                .map(|w| match w.strip_prefix('X') {
                    Some(rest) => format!("X {rest}"),
                    None => w.to_string(),
                })
                .collect();
            if line.contains("MARKER") || section == "OBJSENSE" {
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let (f1, rest): (&str, &[String]) = match section {
                "ROWS" | "BOUNDS" => (&t[0], &t[1..]),
                _ => ("", &t[..]),
            };
            let mut l = format!(" {f1:<2} ");
            let widths = [8usize, 10, 12, 10, 12];
            let starts = [4usize, 14, 24, 39, 49];
            for (k, v) in rest.iter().enumerate() {
                while l.len() < starts[k] {
                    l.push(' ');
                }
                l.push_str(&format!("{v:<w$}", w = widths[k]));
            }
            out.push_str(l.trim_end());
            out.push('\n');
        }
        out
    }

    #[test]
    fn free_format_without_set_names_and_defaults() {
        let text =
            "NAME x\nOBJSENSE MIN\nROWS\n N obj\n L c\nCOLUMNS\n MARKER 'MARKER' 'INTORG'\n \
                    y obj 1 c 2\n MARKER 'MARKER' 'INTEND'\n z obj 1 c 1\nRHS\n c 3\nBOUNDS\n \
                    BV z\nENDATA\n";
        let m = parse(text, &MpsOptions::default()).unwrap().model;
        assert_eq!((m.vars[0].lo, m.vars[0].hi), (0.0, 1.0));
        assert_eq!(m.vars[1].domain, Domain::Integer);
        assert_eq!(m.rows[0].hi, 3.0);
    }

    #[test]
    fn write_reads_back_exactly() {
        let m = parse(SMALL, &MpsOptions::default()).unwrap().model;
        let back = parse(&write(&m), &MpsOptions::default()).unwrap();
        assert_eq!(back.model.vars, m.vars);
        assert_eq!(back.model.rows, m.rows);
        assert_eq!(back.model.obj_offset, m.obj_offset);
        assert_eq!(back.model.direction, m.direction);
        assert!(back.warnings.is_empty());
    }

    #[test]
    fn rejects_unsupported_sections() {
        let text = "NAME x\nROWS\n N obj\nCOLUMNS\nSOS\nENDATA\n";
        assert!(parse(text, &MpsOptions::default()).is_err());
    }
}
