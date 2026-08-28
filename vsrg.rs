//! .vsrg V1 reader
//!
//! Everything borrows from the input buffer, so `parse` allocates nothing
//! except the `resources` index. `.grid` and `.notes` stay as raw byte slices
//! that you iterate lazily (or hand straight to numpy / a GPU buffer).
//!
//! Integers are little-endian, so the fixed-stride sections can be mmapped
//! and cast directly rather than decoded field by field.

use core::str;

pub const MAGIC: &[u8] = b"beatsoup";
pub const END: &[u8] = &[0xDE, 0xAD, 0xBE, 0xA7];
pub const VERSION: u16 = 1;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Eof,
    BadMagic,
    BadEndDelimiter,
    /// Sections are positional, so an unknown version must be rejected
    /// outright. There is no way to skip what we don't recognise.
    UnsupportedVersion(u16),
    /// A section's declared size is not a whole number of rows.
    RaggedSection { section: &'static str, size: usize, stride: usize },
    MetaSizeMismatch { declared: usize, actual: usize },
    Utf8,
}

// ─────────────────────────────────────────────────────────────────────────
// Cursor
// ─────────────────────────────────────────────────────────────────────────

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cur { b, p: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.p.checked_add(n).ok_or(Error::Eof)?;
        let s = self.b.get(self.p..end).ok_or(Error::Eof)?;
        self.p = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// One NUL-terminated UTF-8 string; consumes the terminator.
    fn cstr(&mut self) -> Result<&'a str, Error> {
        let rest = self.b.get(self.p..).ok_or(Error::Eof)?;
        let n = rest.iter().position(|&c| c == 0).ok_or(Error::Eof)?;
        let s = str::from_utf8(&rest[..n]).map_err(|_| Error::Utf8)?;
        self.p += n + 1;
        Ok(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Rows
// ─────────────────────────────────────────────────────────────────────────

/// A view over a row's trailing `u32[]` parameter block.
///
/// Kept as bytes rather than `&[u32]` because row stride is `10 + 4n` for
/// notes, so parameters land on a 2-mod-4 offset and can't be safely cast.
#[derive(Debug, Clone, Copy)]
pub struct Params<'a>(&'a [u8]);

impl<'a> Params<'a> {
    pub fn len(&self) -> usize {
        self.0.len() / 4
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<u32> {
        let b = self.0.get(i * 4..i * 4 + 4)?;
        Some(u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + 'a {
        self.0
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GridRow<'a> {
    pub tick: u32,
    pub function: u32,
    pub params: Params<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct Note<'a> {
    pub column: u8,
    pub note_type: u8,
    pub start_tick: u32,
    pub end_tick: u32,
    pub params: Params<'a>,
}

impl Note<'_> {
    /// `start == end` is a tap; anything else spans ticks.
    pub fn is_held(&self) -> bool {
        self.end_tick != self.start_tick
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Iterators
// ─────────────────────────────────────────────────────────────────────────

pub struct GridIter<'a> {
    b: &'a [u8],
    stride: usize,
}

impl<'a> Iterator for GridIter<'a> {
    type Item = GridRow<'a>;

    fn next(&mut self) -> Option<GridRow<'a>> {
        if self.b.len() < self.stride {
            return None;
        }
        let (r, rest) = self.b.split_at(self.stride);
        self.b = rest;
        Some(GridRow {
            tick: u32::from_le_bytes(r[0..4].try_into().unwrap()),
            function: u32::from_le_bytes(r[4..8].try_into().unwrap()),
            params: Params(&r[8..]),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.b.len() / self.stride;
        (n, Some(n))
    }
}

impl ExactSizeIterator for GridIter<'_> {}

pub struct NoteIter<'a> {
    b: &'a [u8],
    stride: usize,
}

impl<'a> Iterator for NoteIter<'a> {
    type Item = Note<'a>;

    fn next(&mut self) -> Option<Note<'a>> {
        if self.b.len() < self.stride {
            return None;
        }
        let (r, rest) = self.b.split_at(self.stride);
        self.b = rest;
        Some(Note {
            column: r[0],
            note_type: r[1],
            start_tick: u32::from_le_bytes(r[2..6].try_into().unwrap()),
            end_tick: u32::from_le_bytes(r[6..10].try_into().unwrap()),
            params: Params(&r[10..]),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.b.len() / self.stride;
        (n, Some(n))
    }
}

impl ExactSizeIterator for NoteIter<'_> {}

// ─────────────────────────────────────────────────────────────────────────
// The file
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Vsrg<'a> {
    pub version: u16,

    // .meta
    pub title: &'a str,
    pub artist: &'a str,
    pub tags: &'a str,
    pub song_file: &'a str,

    /// `.resources`, indexable as `resources[id]`.
    pub resources: Vec<&'a str>,

    // .chart
    pub column_count: u8,
    pub note_type: u8,
    pub grid_param_count: u8,
    pub note_param_count: u8,

    /// Raw `.grid` bytes. Stride is [`Vsrg::grid_stride`].
    pub grid: &'a [u8],
    /// Raw `.notes` bytes. Stride is [`Vsrg::note_stride`].
    pub notes: &'a [u8],
}

impl<'a> Vsrg<'a> {
    pub fn grid_stride(&self) -> usize {
        8 + 4 * self.grid_param_count as usize
    }

    pub fn note_stride(&self) -> usize {
        10 + 4 * self.note_param_count as usize
    }

    /// Derived from section size, since the spec no longer stores a count.
    pub fn num_grid(&self) -> usize {
        self.grid.len() / self.grid_stride()
    }

    pub fn num_notes(&self) -> usize {
        self.notes.len() / self.note_stride()
    }

    pub fn grid_rows(&self) -> GridIter<'a> {
        GridIter { b: self.grid, stride: self.grid_stride() }
    }

    pub fn note_rows(&self) -> NoteIter<'a> {
        NoteIter { b: self.notes, stride: self.note_stride() }
    }
}

pub fn parse(buf: &[u8]) -> Result<Vsrg<'_>, Error> {
    let mut c = Cur::new(buf);

    if c.take(MAGIC.len())? != MAGIC {
        return Err(Error::BadMagic);
    }

    // ── .meta ───────────────────────────────────────────────────────────
    let meta_start = c.p;

    let version = c.u16()?;
    if version != VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    c.take(62)?; // reserved, bytes 3-64

    let meta_size = c.u32()? as usize;
    let resources_size = c.u32()? as usize;
    let chart_size = c.u32()? as usize;
    let grid_size = c.u32()? as usize;
    let notes_size = c.u32()? as usize;

    let title = c.cstr()?;
    let artist = c.cstr()?;
    let tags = c.cstr()?;
    let song_file = c.cstr()?;

    let meta_actual = c.p - meta_start;
    if meta_actual != meta_size {
        return Err(Error::MetaSizeMismatch { declared: meta_size, actual: meta_actual });
    }

    // ── .resources ──────────────────────────────────────────────────────
    // Size-bounded, so we read strings until the section is consumed.
    let res_bytes = c.take(resources_size)?;
    let mut rc = Cur::new(res_bytes);
    let mut resources = Vec::new();
    while rc.p < res_bytes.len() {
        resources.push(rc.cstr()?);
    }

    // ── .chart ──────────────────────────────────────────────────────────
    let chart = c.take(chart_size)?;
    let mut cc = Cur::new(chart);
    let column_count = cc.u8()?;
    let note_type = cc.u8()?;
    let grid_param_count = cc.u8()?;
    let note_param_count = cc.u8()?;

    // ── .grid / .notes ──────────────────────────────────────────────────
    let grid_stride = 8 + 4 * grid_param_count as usize;
    let note_stride = 10 + 4 * note_param_count as usize;

    if grid_size % grid_stride != 0 {
        return Err(Error::RaggedSection {
            section: ".grid",
            size: grid_size,
            stride: grid_stride,
        });
    }
    if notes_size % note_stride != 0 {
        return Err(Error::RaggedSection {
            section: ".notes",
            size: notes_size,
            stride: note_stride,
        });
    }

    let grid = c.take(grid_size)?;
    let notes = c.take(notes_size)?;

    // Landing exactly on the sentinel validates every size in the header
    // at once. Anything upstream that lied shows up here.
    if c.take(END.len())? != END {
        return Err(Error::BadEndDelimiter);
    }

    Ok(Vsrg {
        version,
        title,
        artist,
        tags,
        song_file,
        resources,
        column_count,
        note_type,
        grid_param_count,
        note_param_count,
        grid,
        notes,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(s.as_bytes());
        v.push(0);
    }

    /// 4K chart, 0 grid params, 1 note param, 1 grid row, 2 notes.
    fn sample() -> Vec<u8> {
        let mut meta = Vec::new();
        meta.extend_from_slice(&VERSION.to_le_bytes());
        meta.extend_from_slice(&[0u8; 62]);
        let sizes_at = meta.len(); // byte 64, 4-byte aligned
        meta.extend_from_slice(&[0u8; 20]);
        for s in ["Song", "Artist", "tag1 tag2", "audio.ogg"] {
            push_str(&mut meta, s);
        }

        let mut res = Vec::new();
        for s in ["audio.ogg", "bg.png"] {
            push_str(&mut res, s);
        }

        let chart = vec![4u8, 0, 0, 1];

        let mut grid = Vec::new();
        grid.extend_from_slice(&0u32.to_le_bytes()); // tick
        grid.extend_from_slice(&0u32.to_le_bytes()); // function

        let mut notes = Vec::new();
        for (col, kind, start, end, param) in
            [(0u8, 0u8, 0u32, 0u32, 7u32), (3, 1, 48, 96, 9)]
        {
            notes.push(col);
            notes.push(kind);
            notes.extend_from_slice(&start.to_le_bytes());
            notes.extend_from_slice(&end.to_le_bytes());
            notes.extend_from_slice(&param.to_le_bytes());
        }

        let sizes = [
            meta.len() as u32,
            res.len() as u32,
            chart.len() as u32,
            grid.len() as u32,
            notes.len() as u32,
        ];
        for (i, s) in sizes.iter().enumerate() {
            let at = sizes_at + i * 4;
            meta[at..at + 4].copy_from_slice(&s.to_le_bytes());
        }

        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&meta);
        f.extend_from_slice(&res);
        f.extend_from_slice(&chart);
        f.extend_from_slice(&grid);
        f.extend_from_slice(&notes);
        f.extend_from_slice(END);
        f
    }

    #[test]
    fn parses() {
        let buf = sample();
        let v = parse(&buf).unwrap();

        assert_eq!(v.title, "Song");
        assert_eq!(v.song_file, "audio.ogg");
        assert_eq!(v.resources, ["audio.ogg", "bg.png"]);
        assert_eq!(v.column_count, 4);
        assert_eq!(v.num_grid(), 1);
        assert_eq!(v.num_notes(), 2);

        let n: Vec<_> = v.note_rows().collect();
        assert_eq!(n[0].column, 0);
        assert!(!n[0].is_held());
        assert_eq!(n[1].column, 3);
        assert_eq!(n[1].start_tick, 48);
        assert_eq!(n[1].end_tick, 96);
        assert!(n[1].is_held());
        assert_eq!(n[1].params.get(0), Some(9));
    }

    /// Every truncation must be an Err, never a panic. This is the property
    /// worth handing to cargo-fuzz later.
    #[test]
    fn truncation_never_panics() {
        let buf = sample();
        for n in 0..buf.len() {
            let _ = parse(&buf[..n]);
        }
    }

    #[test]
    fn rejects_bad_version() {
        let mut buf = sample();
        buf[MAGIC.len()] = 2; // low byte, little-endian
        assert!(matches!(parse(&buf), Err(Error::UnsupportedVersion(2))));
    }

    #[test]
    fn catches_ragged_note_section() {
        let mut buf = sample();
        let v = parse(&buf).unwrap();
        let stride = v.note_stride();
        // Claim one more note than the bytes can hold.
        let notes_size_at = MAGIC.len() + 64 + 16;
        let bad = (v.notes.len() + 1) as u32;
        buf[notes_size_at..notes_size_at + 4].copy_from_slice(&bad.to_le_bytes());
        assert!(matches!(
            parse(&buf),
            Err(Error::RaggedSection { section: ".notes", stride: s, .. }) if s == stride
        ));
    }
}
