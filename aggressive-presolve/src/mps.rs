//! MPS reader (fixed and free) and a free-MPS writer.
//!
//! Sections: NAME, OBJSENSE (`MIN`/`MAX`/`MINIMIZE`/`MAXIMIZE`, on the same
//! line or the next), OBJNAME, ROWS, COLUMNS (with `'MARKER'`
//! `'INTORG'`/`'INTEND'`), RHS, RANGES, BOUNDS, ENDATA. Conventions:
//!
//! * the first `N` row is the objective (unless OBJNAME names one); entries on
//!   other `N` rows are ignored;
//! * an RHS entry on the objective row is the *negated* objective constant;
//! * RANGES follow the standard table: `E` row with `R >= 0` gives
//!   `[b, b + R]`, with `R < 0` gives `[b + R, b]`; `L` gives `[b − |R|, b]`;
//!   `G` gives `[b, b + |R|]`;
//! * BOUNDS `UP LO FX FR MI PL BV LI UI`. A negative `UP` with no `LO` on the
//!   column moves its lower bound to `-inf` (classic MPS). An integer column
//!   with no BOUNDS entry at all defaults to `[0, 1]` (MPSX, which MIPLIB 3
//!   assumes) unless [`MpsOptions::integer_default_binary`] is off, when it
//!   is `[0, +inf)`;
//! * only the first RHS, RANGES and BOUNDS vectors are read;
//! * the set name of an RHS/RANGES/BOUNDS line may be omitted (free MPS);
//! * a fixed-format line whose whitespace tokens do not make sense (names
//!   with spaces) is read by column positions.
//!
//! Anything else (SOS, QUADOBJ, QMATRIX, QCMATRIX, INDICATORS, CSECTION,
//! semi-continuous `SC` bounds, ...) is an error, never a guess.

use crate::model::{Direction, Model, VarKind};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct MpsOptions {
    /// Integer columns with no BOUNDS entry are `[0, 1]` (else `[0, inf)`).
    pub integer_default_binary: bool,
}

impl Default for MpsOptions {
    fn default() -> Self {
        MpsOptions {
            integer_default_binary: true,
        }
    }
}

/// Reads a model from a file, by extension: `.mps`, `.lp`, either `.gz`.
pub fn read_file(path: &Path) -> Result<Model, String> {
    let text = read_text(path)?;
    let name = path.to_string_lossy().to_lowercase();
    let stem = name.strip_suffix(".gz").unwrap_or(&name);
    let mut model = if stem.ends_with(".lp") {
        crate::lp::parse(&text)?
    } else {
        parse(&text, &MpsOptions::default())?
    };
    if model.name.is_empty() {
        model.name = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
    }
    Ok(model)
}

/// A file's text, gunzipped in memory for `.gz`.
pub fn read_text(path: &Path) -> Result<String, String> {
    let err = |e: &dyn std::fmt::Display| format!("cannot read {}: {}", path.display(), e);
    if path.extension().is_some_and(|e| e == "gz") {
        let f = std::fs::File::open(path).map_err(|e| err(&e))?;
        let mut s = String::new();
        flate2::read::GzDecoder::new(std::io::BufReader::new(f))
            .read_to_string(&mut s)
            .map_err(|e| err(&e))?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).map_err(|e| err(&e))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum RowType {
    N,
    L,
    G,
    E,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Section {
    None,
    Name,
    ObjSense,
    ObjName,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
}

/// Fixed-format fields 1–6 by column position (1-based columns 2–3, 5–12,
/// 15–22, 25–36, 40–47, 50–61), trimmed, empty fields dropped.
fn fixed_fields(line: &str) -> Vec<String> {
    let spans = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)];
    let chars: Vec<char> = line.chars().collect();
    spans
        .iter()
        .filter_map(|&(a, b)| {
            if a >= chars.len() {
                return None;
            }
            let s: String = chars[a..b.min(chars.len())].iter().collect();
            let s = s.trim().to_string();
            (!s.is_empty()).then_some(s)
        })
        .collect()
}

fn num(tok: &str, line_no: usize) -> Result<f64, String> {
    let t = tok.trim();
    let v = match t.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => f64::INFINITY,
        "-inf" | "-infinity" => f64::NEG_INFINITY,
        _ => t
            .parse::<f64>()
            .map_err(|_| format!("line {}: bad number {:?}", line_no, tok))?,
    };
    if v.is_nan() {
        return Err(format!("line {}: NaN", line_no));
    }
    Ok(v)
}

fn is_num(tok: &str) -> bool {
    num(tok, 0).is_ok()
}

struct ColData {
    entries: Vec<(usize, f64)>,
    obj: f64,
    integer: bool,
    lo: Option<f64>,
    hi: Option<f64>,
    any_bound: bool,
}

/// Parses MPS text.
pub fn parse(text: &str, opts: &MpsOptions) -> Result<Model, String> {
    let mut name = String::new();
    let mut direction = Direction::Minimize;
    let mut obj_name: Option<String> = None;
    let mut row_index: HashMap<String, Option<usize>> = HashMap::new(); // None: N row
    let mut rows: Vec<(String, RowType)> = Vec::new();
    let mut obj_row: Option<String> = None;
    let mut cols: Vec<(String, ColData)> = Vec::new();
    let mut col_index: HashMap<String, usize> = HashMap::new();
    let mut rhs: Vec<Option<f64>> = Vec::new();
    let mut ranges: Vec<Option<f64>> = Vec::new();
    let mut obj_constant = 0.0;
    let mut rhs_set: Option<String> = None;
    let mut ranges_set: Option<String> = None;
    let mut bounds_set: Option<String> = None;
    let mut in_int = false;
    let mut section = Section::None;

    for (ln, raw) in text.lines().enumerate() {
        let line_no = ln + 1;
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() || line.starts_with('*') {
            continue;
        }
        let header = !line.starts_with(' ') && !line.starts_with('\t');
        if header {
            let mut toks = line.split_whitespace();
            let kw = toks.next().unwrap_or("").to_ascii_uppercase();
            let rest: Vec<&str> = toks.collect();
            section = match kw.as_str() {
                "NAME" => {
                    name = rest.join(" ");
                    Section::Name
                }
                "OBJSENSE" => {
                    if let Some(s) = rest.first() {
                        direction = sense(s, line_no)?;
                    }
                    Section::ObjSense
                }
                "OBJSENCE" => Section::ObjSense,
                "OBJNAME" => {
                    if let Some(s) = rest.first() {
                        obj_name = Some(s.to_string());
                    }
                    Section::ObjName
                }
                "ROWS" => Section::Rows,
                "COLUMNS" => Section::Columns,
                "RHS" => Section::Rhs,
                "RANGES" => Section::Ranges,
                "BOUNDS" => Section::Bounds,
                "ENDATA" => break,
                // Free MPS may put OBJSENSE's value on a header-looking line.
                "MAX" | "MAXIMIZE" | "MIN" | "MINIMIZE" if section == Section::ObjSense => {
                    direction = sense(&kw, line_no)?;
                    Section::ObjSense
                }
                other => {
                    return Err(format!(
                        "line {}: unsupported MPS section {:?}",
                        line_no, other
                    ))
                }
            };
            continue;
        }
        let ws: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        match section {
            Section::None | Section::Name => {
                return Err(format!("line {}: data outside a section", line_no))
            }
            Section::ObjSense => direction = sense(&ws[0], line_no)?,
            Section::ObjName => obj_name = Some(ws[0].clone()),
            Section::Rows => {
                let f = if ws.len() == 2 { ws } else { fixed_fields(line) };
                if f.len() != 2 {
                    return Err(format!("line {}: bad ROWS line", line_no));
                }
                let t = match f[0].to_ascii_uppercase().as_str() {
                    "N" => RowType::N,
                    "L" => RowType::L,
                    "G" => RowType::G,
                    "E" => RowType::E,
                    t => return Err(format!("line {}: row type {:?}", line_no, t)),
                };
                let rname = f[1].clone();
                if row_index.contains_key(&rname) {
                    return Err(format!("line {}: duplicate row {}", line_no, rname));
                }
                if t == RowType::N {
                    let is_obj = match &obj_name {
                        Some(o) => *o == rname,
                        None => obj_row.is_none(),
                    };
                    if is_obj {
                        obj_row = Some(rname.clone());
                    }
                    row_index.insert(rname, None);
                } else {
                    row_index.insert(rname.clone(), Some(rows.len()));
                    rows.push((rname, t));
                    rhs.push(None);
                    ranges.push(None);
                }
            }
            Section::Columns => {
                let is_marker = ws.len() >= 3 && ws[1].trim_matches('\'').eq_ignore_ascii_case("MARKER");
                if is_marker {
                    match ws[2].trim_matches('\'').to_ascii_uppercase().as_str() {
                        "INTORG" => in_int = true,
                        "INTEND" => in_int = false,
                        m => return Err(format!("line {}: marker {:?}", line_no, m)),
                    }
                    continue;
                }
                let ok = |f: &[String]| {
                    (f.len() == 3 || f.len() == 5)
                        && is_num(&f[2])
                        && (f.len() == 3 || is_num(&f[4]))
                        && row_index.contains_key(&f[1])
                        && (f.len() == 3 || row_index.contains_key(&f[3]))
                };
                let f = if ok(&ws) { ws } else { fixed_fields(line) };
                if !ok(&f) {
                    return Err(format!("line {}: bad COLUMNS line {:?}", line_no, line));
                }
                let c = match col_index.get(&f[0]) {
                    Some(&c) => c,
                    None => {
                        col_index.insert(f[0].clone(), cols.len());
                        cols.push((
                            f[0].clone(),
                            ColData {
                                entries: Vec::new(),
                                obj: 0.0,
                                integer: in_int,
                                lo: None,
                                hi: None,
                                any_bound: false,
                            },
                        ));
                        cols.len() - 1
                    }
                };
                for pair in f[1..].chunks(2) {
                    let v = num(&pair[1], line_no)?;
                    match row_index[&pair[0]] {
                        Some(r) => cols[c].1.entries.push((r, v)),
                        None => {
                            if obj_row.as_deref() == Some(pair[0].as_str()) {
                                cols[c].1.obj += v;
                            }
                        }
                    }
                }
            }
            Section::Rhs | Section::Ranges => {
                // [set] row value [row value]
                let parse_pairs = |f: &[String]| -> Option<(Option<String>, Vec<(String, String)>)> {
                    let (set, rest) = if f.len() % 2 == 1 {
                        (Some(f[0].clone()), &f[1..])
                    } else {
                        (None, f)
                    };
                    if rest.is_empty() || rest.len() > 4 {
                        return None;
                    }
                    let pairs: Vec<(String, String)> = rest
                        .chunks(2)
                        .map(|p| (p[0].clone(), p[1].clone()))
                        .collect();
                    pairs
                        .iter()
                        .all(|(r, v)| row_index.contains_key(r) && is_num(v))
                        .then_some((set, pairs))
                };
                let (set, pairs) = match parse_pairs(&ws) {
                    Some(p) => p,
                    None => parse_pairs(&fixed_fields(line))
                        .ok_or_else(|| format!("line {}: bad {:?} line", line_no, section))?,
                };
                let slot = if section == Section::Rhs {
                    &mut rhs_set
                } else {
                    &mut ranges_set
                };
                let set = set.unwrap_or_default();
                match slot {
                    None => *slot = Some(set),
                    Some(s) if *s == set => {}
                    Some(_) => continue, // only the first vector
                }
                for (r, v) in pairs {
                    let v = num(&v, line_no)?;
                    match row_index[&r] {
                        Some(i) => {
                            if section == Section::Rhs {
                                rhs[i] = Some(v);
                            } else {
                                ranges[i] = Some(v);
                            }
                        }
                        None => {
                            if section == Section::Rhs && obj_row.as_deref() == Some(r.as_str()) {
                                obj_constant = -v;
                            } else if section == Section::Ranges {
                                return Err(format!("line {}: RANGES on a free row", line_no));
                            }
                        }
                    }
                }
            }
            Section::Bounds => {
                let parse_bound = |f: &[String]| -> Option<(String, Option<String>, String, Option<String>)> {
                    let t = f.first()?.to_ascii_uppercase();
                    let valued = matches!(t.as_str(), "UP" | "LO" | "FX" | "LI" | "UI" | "SC");
                    let valueless = matches!(t.as_str(), "FR" | "MI" | "PL");
                    let known = |c: &String| col_index.contains_key(c);
                    match (f.len(), t.as_str()) {
                        (4, _) if (valued || t == "BV") && known(&f[2]) && is_num(&f[3]) => {
                            Some((t, Some(f[1].clone()), f[2].clone(), Some(f[3].clone())))
                        }
                        (3, _) if valued && known(&f[1]) && is_num(&f[2]) => {
                            Some((t, None, f[1].clone(), Some(f[2].clone())))
                        }
                        (3, "BV") if known(&f[1]) && is_num(&f[2]) => {
                            Some((t, None, f[1].clone(), Some(f[2].clone())))
                        }
                        (3, _) if (valueless || t == "BV") && known(&f[2]) => {
                            Some((t, Some(f[1].clone()), f[2].clone(), None))
                        }
                        (2, _) if (valueless || t == "BV") && known(&f[1]) => {
                            Some((t, None, f[1].clone(), None))
                        }
                        _ => None,
                    }
                };
                let (t, set, cname, val) = match parse_bound(&ws) {
                    Some(b) => b,
                    None => parse_bound(&fixed_fields(line))
                        .ok_or_else(|| format!("line {}: bad BOUNDS line {:?}", line_no, line))?,
                };
                let set = set.unwrap_or_default();
                match &bounds_set {
                    None => bounds_set = Some(set),
                    Some(s) if *s == set => {}
                    Some(_) => continue,
                }
                let c = col_index[&cname];
                let v = val.map(|v| num(&v, line_no)).transpose()?;
                let col = &mut cols[c].1;
                col.any_bound = true;
                match t.as_str() {
                    "UP" => col.hi = v,
                    "LO" => col.lo = v,
                    "FX" => {
                        col.lo = v;
                        col.hi = v;
                    }
                    "FR" => {
                        col.lo = Some(f64::NEG_INFINITY);
                        col.hi = Some(f64::INFINITY);
                    }
                    "MI" => col.lo = Some(f64::NEG_INFINITY),
                    "PL" => col.hi = Some(f64::INFINITY),
                    "BV" => {
                        col.integer = true;
                        col.lo = Some(0.0);
                        col.hi = Some(1.0);
                    }
                    "LI" => {
                        col.integer = true;
                        col.lo = v;
                    }
                    "UI" => {
                        col.integer = true;
                        col.hi = v;
                    }
                    "SC" => {
                        return Err(format!(
                            "line {}: semi-continuous bound on {} is not supported",
                            line_no, cname
                        ))
                    }
                    t => return Err(format!("line {}: bound type {:?}", line_no, t)),
                }
            }
        }
    }

    let mut model = Model::new(direction);
    model.name = name;
    model.obj_offset = obj_constant;
    let mut row_terms: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows.len()];
    for (j, (cname, col)) in cols.iter().enumerate() {
        let (lo, hi) = if col.integer && !col.any_bound {
            if opts.integer_default_binary {
                (0.0, 1.0)
            } else {
                (0.0, f64::INFINITY)
            }
        } else {
            let hi = col.hi.unwrap_or(f64::INFINITY);
            let lo = match col.lo {
                Some(l) => l,
                // Classic MPS: a negative UP with no LO frees the lower side.
                None if hi < 0.0 => f64::NEG_INFINITY,
                None => 0.0,
            };
            (lo, hi)
        };
        let kind = if col.integer {
            VarKind::Integer
        } else {
            VarKind::Continuous
        };
        model.add_var(cname.clone(), lo, hi, col.obj, kind);
        for &(r, a) in &col.entries {
            row_terms[r].push((j, a));
        }
    }
    for (i, ((rname, t), terms)) in rows.iter().zip(row_terms).enumerate() {
        let b = rhs[i].unwrap_or(0.0);
        let (lo, hi) = match (*t, ranges[i]) {
            (RowType::E, None) => (b, b),
            (RowType::L, None) => (f64::NEG_INFINITY, b),
            (RowType::G, None) => (b, f64::INFINITY),
            (RowType::E, Some(r)) if r >= 0.0 => (b, b + r),
            (RowType::E, Some(r)) => (b + r, b),
            (RowType::L, Some(r)) => (b - r.abs(), b),
            (RowType::G, Some(r)) => (b, b + r.abs()),
            (RowType::N, _) => unreachable!("N rows are not stored"),
        };
        model.add_range_row(rname.clone(), terms, lo, hi);
    }
    model.validate()?;
    Ok(model)
}

fn sense(tok: &str, line_no: usize) -> Result<Direction, String> {
    match tok.to_ascii_uppercase().as_str() {
        "MAX" | "MAXIMIZE" | "MAXIMISE" => Ok(Direction::Maximize),
        "MIN" | "MINIMIZE" | "MINIMISE" => Ok(Direction::Minimize),
        t => Err(format!("line {}: OBJSENSE {:?}", line_no, t)),
    }
}

/// Writes `model` as free MPS (names must not contain spaces). Integer
/// columns always get explicit bounds, so no reader default applies.
pub fn write(model: &Model) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let g = |v: f64| format!("{:e}", v);
    let _ = writeln!(s, "NAME {}", if model.name.is_empty() { "model" } else { &model.name });
    let _ = writeln!(
        s,
        "OBJSENSE\n    {}",
        match model.direction {
            Direction::Minimize => "MIN",
            Direction::Maximize => "MAX",
        }
    );
    let _ = writeln!(s, "ROWS\n N obj");
    let rname = |i: usize| format!("R{}", i);
    let cname = |j: usize| format!("C{}", j);
    for (i, r) in model.rows.iter().enumerate() {
        let t = if r.lo == r.hi {
            "E"
        } else if r.lo.is_finite() {
            "G"
        } else {
            "L"
        };
        let _ = writeln!(s, " {} {}", t, rname(i));
    }
    let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); model.vars.len()];
    for (i, r) in model.rows.iter().enumerate() {
        for &(j, a) in &r.terms {
            by_col[j].push((i, a));
        }
    }
    let _ = writeln!(s, "COLUMNS");
    let mut in_int = false;
    for (j, v) in model.vars.iter().enumerate() {
        if v.is_integer() != in_int {
            in_int = v.is_integer();
            let _ = writeln!(
                s,
                "    MARKER 'MARKER' '{}'",
                if in_int { "INTORG" } else { "INTEND" }
            );
        }
        let _ = writeln!(s, "    {} obj {}", cname(j), g(v.obj));
        for &(i, a) in &by_col[j] {
            let _ = writeln!(s, "    {} {} {}", cname(j), rname(i), g(a));
        }
    }
    if in_int {
        let _ = writeln!(s, "    MARKER 'MARKER' 'INTEND'");
    }
    let _ = writeln!(s, "RHS");
    if model.obj_offset != 0.0 {
        let _ = writeln!(s, "    RHS obj {}", g(-model.obj_offset));
    }
    for (i, r) in model.rows.iter().enumerate() {
        let b = if r.lo == r.hi || r.lo.is_finite() {
            r.lo
        } else {
            r.hi
        };
        if b != 0.0 && b.is_finite() {
            let _ = writeln!(s, "    RHS {} {}", rname(i), g(b));
        }
    }
    let ranged: Vec<_> = model
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.lo != r.hi && r.lo.is_finite() && r.hi.is_finite())
        .collect();
    if !ranged.is_empty() {
        let _ = writeln!(s, "RANGES");
        for (i, r) in ranged {
            let _ = writeln!(s, "    RNG {} {}", rname(i), g(r.hi - r.lo));
        }
    }
    let _ = writeln!(s, "BOUNDS");
    for (j, v) in model.vars.iter().enumerate() {
        let c = cname(j);
        if v.lo == v.hi {
            let _ = writeln!(s, " FX BND {} {}", c, g(v.lo));
            continue;
        }
        match (v.lo.is_finite(), v.hi.is_finite()) {
            (false, false) => {
                let _ = writeln!(s, " FR BND {}", c);
            }
            (false, true) => {
                let _ = writeln!(s, " MI BND {}", c);
                let _ = writeln!(s, " UP BND {} {}", c, g(v.hi));
            }
            (true, hi_finite) => {
                let _ = writeln!(s, " LO BND {} {}", c, g(v.lo));
                if hi_finite {
                    let _ = writeln!(s, " UP BND {} {}", c, g(v.hi));
                } else {
                    let _ = writeln!(s, " PL BND {}", c);
                }
            }
        }
    }
    let _ = writeln!(s, "ENDATA");
    s
}
