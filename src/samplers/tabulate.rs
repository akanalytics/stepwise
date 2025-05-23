use super::{Destination, Sampler, SamplingOutcome};
use std::fmt::{Alignment as FmtAlignment, Display, Write as _};

/// A Tabulator or table generator which can display to Stderr or Stdout
///
/// It's a sampler of tuples of items implementing Display
///
/// It will skip display/selection in either of the following cases:
/// - the relevant stdout/stderr is not a terminal (eg output redirected to a pipe)
/// - NO_COLOR env variable is set
/// - the last display was withing 50ms
///
///
/// It will always return [`SamplingOutcome::Selected`]
///
/// The template taken by [`new`](Self::new) must have the same number of columns
/// as the tuple has items, and must use a consistent column seperator of
/// '\t' ',' '|' or ':'.
///
/// "|" renders markdown, and ":" renders invisibly (for plain text)
///
/// Width and justification are taken from the template. Errors encounterd writing/flushing to stdout/stderr are ignored
///
/// # Example templates:
/// ```text
/// template = "Iter, Name,City   , State, Cost:.2"
///
/// - Comma seperated
/// - Header text col 1= "Iter", Right aligned (default), width=4
/// - Header text col 2= "Name", Right aligned, width=5
/// - Header text col 3= "City", Left aligned , width=7
/// - Header text col 4= "State", Right aligned, width=6
/// - Header text col 5= "Cost", Right aligned, width=5, precision=2
///```
///
/// # Example:
/// ```
/// use stepwise::{samplers, samplers::Sampler as _};
/// let mut v = Vec::new();
/// let mut tab = samplers::Tabulate::new("  Iter,  Value:.2 ").on_vec(&mut v);
/// tab.sample((1, 2.14159));
/// tab.sample((2, 1.71828));
///
/// assert_eq!(v[0], "  Iter,  Value");
/// assert_eq!(v[1], "     1,   2.14");
/// assert_eq!(v[2], "     2,   1.72");
/// ```
///
#[derive(Debug)]
pub struct Tabulate<'a> {
    headers: Vec<String>,
    format_hints: Vec<FormatHint>,
    separator: Option<char>,
    headers_rendered: bool,

    #[allow(dead_code)]
    header_template: String, // used in Debug::fmt

    #[allow(dead_code)]
    destination: Destination<'a>,
}

#[derive(Debug, Clone)]
struct FormatHint {
    width: usize,
    precision: Option<usize>,
    align: FmtAlignment,
}

fn parse_header_row(row: &str, sep: Option<char>) -> (Vec<String>, Vec<FormatHint>) {
    let cols: Vec<&str> = match sep {
        Some(sep) => row.split(sep).collect(),
        None => row.split_whitespace().collect(),
    };

    let mut names = Vec::new();
    let mut hints = Vec::new();

    for raw in cols {
        let (trimmed, raw, width, precision) = if let Some((label, fmt)) = raw.rsplit_once(':') {
            let width = label.len();
            let fmt = fmt.trim();
            let p = if let Some((_fw, fp)) = fmt.split_once('.') {
                fp.parse().ok()
            } else {
                None
            };
            (label.trim().to_string(), label, width, p)
        } else {
            (raw.trim().to_string(), raw, raw.len(), None)
        };

        let left_space = raw.starts_with(' ');
        let right_space = raw.ends_with(' ');

        let align = match (left_space, right_space) {
            (true, false) => FmtAlignment::Right,
            (false, true) => FmtAlignment::Left,
            (true, true) => FmtAlignment::Center,
            _ => FmtAlignment::Right,
        };

        names.push(trimmed);
        hints.push(FormatHint {
            width,
            precision,
            align,
        });
    }

    (names, hints)
}

impl<'a> Tabulate<'a> {
    pub fn new(template: &str) -> Self {
        let separator = if template.contains(',') {
            Some(',')
        } else if template.contains('\t') {
            Some('\t')
        } else if template.contains('|') {
            Some('|')
        } else if template.contains(':') {
            Some(':')
        } else {
            None
        };

        let (headers, format_hints) = parse_header_row(template, separator);
        Self {
            header_template: template.to_string(),
            headers,
            headers_rendered: false,
            format_hints,
            separator,
            destination: Destination::Stdout,
        }
    }

    pub fn on_stdout(self) -> Self {
        let destination = Destination::Stdout.check_visibility();
        Self {
            destination,
            ..self
        }
    }

    pub fn on_stderr(self) -> Self {
        let destination = Destination::Stderr.check_visibility();
        Self {
            destination,
            ..self
        }
    }

    pub fn on_vec(self, vec: &'a mut Vec<String>) -> Self {
        let destination = Destination::StringVecRef(vec).check_visibility();
        Self {
            destination,
            ..self
        }
    }

    fn render_sep(sep: Option<char>) -> String {
        match sep {
            None | Some(':') => "".to_string(),
            Some(sep) => sep.to_string(),
        }
    }

    pub fn render_headers(&mut self) {
        let headers = self.headers.clone(); // avoid self borrow
        let header_refs: Vec<&dyn Display> = headers.iter().map(|s| s as &dyn Display).collect();
        self.render_data(&header_refs, true);
        if self.separator == Some('|') {
            let underline_row: Vec<String> = self
                .format_hints
                .iter()
                .map(|hint| {
                    let w = hint.width.saturating_sub(2);
                    match hint.align {
                        FmtAlignment::Left => format!(":{}", "-".repeat((w + 1).max(3))),
                        FmtAlignment::Right => format!("{}:", "-".repeat((w + 1).max(3))),
                        FmtAlignment::Center => format!(":{}:", "-".repeat(w.max(3))),
                    }
                    .to_string()
                })
                .collect();
            let underline_refs: Vec<&dyn Display> =
                underline_row.iter().map(|s| s as &dyn Display).collect();
            self.render_data(&underline_refs, true);
        }
    }

    fn render_row(&mut self, values: &[&dyn Display]) {
        if !self.headers_rendered {
            self.render_headers();
            self.headers_rendered = true;
        }
        self.render_data(values, false);
    }

    fn render_data(&mut self, values: &[&dyn Display], headers: bool) {
        if values.len() != self.headers.len() {
            panic!("the number of headers in '{}' (len={}) must match the number of Display items (len={})", 
                self.header_template,
                self.headers.len(),
                values.len());
        }
        let mut s = String::new();
        for (i, value) in values.iter().enumerate() {
            if i > 0 || self.separator == Some('|') {
                write!(s, "{}", Self::render_sep(self.separator)).unwrap();
            }

            let hint = self.format_hints.get(i).unwrap_or(&FormatHint {
                width: 10,
                precision: None,
                align: FmtAlignment::Right,
            });

            let mut formatted = String::new();
            if headers {
                write!(&mut formatted, "{}", value).unwrap();
            } else if let Some(p) = hint.precision {
                write!(&mut formatted, "{:.*}", p, value).unwrap(); // Precision formatting
            } else {
                write!(&mut formatted, "{}", value).unwrap();
            }

            let width = hint.width;
            match hint.align {
                FmtAlignment::Left => write!(s, "{:<width$}", formatted, width = width).unwrap(),
                FmtAlignment::Right => write!(s, "{:>width$}", formatted, width = width).unwrap(),
                FmtAlignment::Center => write!(s, "{:^width$}", formatted, width = width).unwrap(),
            }
        }
        if self.separator == Some('|') {
            write!(s, "{}", Self::render_sep(self.separator)).unwrap();
        }
        self.destination.render_on(&s);
    }
}

impl<D1> Sampler<(D1,)> for Tabulate<'_>
where
    D1: Display,
{
    fn sample(&mut self, (d1,): (D1,)) -> SamplingOutcome<(D1,)> {
        self.render_row(&[&d1]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2> Sampler<(D1, D2)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
{
    fn sample(&mut self, (d1, d2): (D1, D2)) -> SamplingOutcome<(D1, D2)> {
        self.render_row(&[&d1, &d2]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3> Sampler<(D1, D2, D3)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
{
    fn sample(&mut self, (d1, d2, d3): (D1, D2, D3)) -> SamplingOutcome<(D1, D2, D3)> {
        self.render_row(&[&d1, &d2, &d3]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4> Sampler<(D1, D2, D3, D4)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
{
    fn sample(&mut self, (d1, d2, d3, d4): (D1, D2, D3, D4)) -> SamplingOutcome<(D1, D2, D3, D4)> {
        self.render_row(&[&d1, &d2, &d3, &d4]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5> Sampler<(D1, D2, D3, D4, D5)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5): (D1, D2, D3, D4, D5),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6> Sampler<(D1, D2, D3, D4, D5, D6)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6): (D1, D2, D3, D4, D5, D6),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7> Sampler<(D1, D2, D3, D4, D5, D6, D7)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7): (D1, D2, D3, D4, D5, D6, D7),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6, &d7]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7, D8> Sampler<(D1, D2, D3, D4, D5, D6, D7, D8)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
    D8: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7, d8): (D1, D2, D3, D4, D5, D6, D7, D8),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7, D8)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7, D8, D9> Sampler<(D1, D2, D3, D4, D5, D6, D7, D8, D9)>
    for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
    D8: Display,
    D9: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7, d8, d9): (D1, D2, D3, D4, D5, D6, D7, D8, D9),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7, D8, D9)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10> Sampler<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10)>
    for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
    D8: Display,
    D9: Display,
    D10: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10): (D1, D2, D3, D4, D5, D6, D7, D8, D9, D10),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9, &d10]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11>
    Sampler<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
    D8: Display,
    D9: Display,
    D10: Display,
    D11: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11): (
            D1,
            D2,
            D3,
            D4,
            D5,
            D6,
            D7,
            D8,
            D9,
            D10,
            D11,
        ),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11)> {
        self.render_row(&[&d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9, &d10, &d11]);
        SamplingOutcome::Selected
    }
}

impl<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12>
    Sampler<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12)> for Tabulate<'_>
where
    D1: Display,
    D2: Display,
    D3: Display,
    D4: Display,
    D5: Display,
    D6: Display,
    D7: Display,
    D8: Display,
    D9: Display,
    D10: Display,
    D11: Display,
    D12: Display,
{
    fn sample(
        &mut self,
        (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12): (
            D1,
            D2,
            D3,
            D4,
            D5,
            D6,
            D7,
            D8,
            D9,
            D10,
            D11,
            D12,
        ),
    ) -> SamplingOutcome<(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12)> {
        self.render_row(&[
            &d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9, &d10, &d11, &d12,
        ]);
        SamplingOutcome::Selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tabulate() {
        let mut v = Vec::new();
        let mut tab = Tabulate::new("  Iter,  Value:.2 ").on_vec(&mut v);

        assert_eq!(tab.format_hints[0].width, 6);
        assert_eq!(tab.format_hints[1].width, 7);

        assert_eq!(tab.format_hints[0].precision, None);
        assert_eq!(tab.format_hints[1].precision, Some(2));

        assert_eq!(tab.format_hints[0].align, FmtAlignment::Right);
        assert_eq!(tab.format_hints[1].align, FmtAlignment::Right);

        tab.sample((1, 2.14159));
        tab.sample((2, 1.71828));
        assert_eq!(v[0], "  Iter,  Value");
        assert_eq!(v[1], "     1,   2.14");
        assert_eq!(v[2], "     2,   1.72");

        let mut tab = Tabulate::new("N|  Value:6.2|   Word|  Int").on_vec(&mut v);
        // println!("{tab:#?}");
        tab.sample((1, 2.14159, "hello", 42));
        assert_eq!(v[3], "|N|  Value|   Word|  Int|");
        assert_eq!(v[4], "|---:|------:|------:|----:|");
        assert_eq!(v[5], "|1|   2.14|  hello|   42|");
    }
}
