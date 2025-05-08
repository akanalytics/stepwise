use std::{
    io::{IsTerminal as _, Write},
    time::Duration,
};

use crate::{step::format_duration, Step};

pub(crate) fn display_progress_bar(step: &Step, delay: Duration) -> Duration {
    if !std::io::stdout().is_terminal()
        || std::env::var("NO_COLOR").is_ok()
        || (step.elapsed() < delay && step.progress_percentage() != Some(100.0))
    {
        return delay;
    }

    // iteration 1 is the first time we print the progress bar
    if step.iteration() > 0 {
        // TODO
        // \x1B[F  - moves the cursor up one line
        // \x1B[2K - clears the entire line
        print!("\x1B[F\x1B[2K")
    }
    let percentage = step.progress_percentage();
    let width = 80;
    let bar = if let Some(percentage) = percentage {
        let filled = ((percentage / 100.0) * width as f64).round() as usize;
        let empty = width - filled;
        format!(
            "[{}{}] {:>6.2}%",
            "=".repeat(filled),
            " ".repeat(empty),
            percentage
        )
    } else {
        format!(
            "[{iter:>6}] elapsed: {elapsed}",
            iter = step.iteration,
            elapsed = format_duration(step.elapsed())
        )
    };
    println!("{bar}");
    std::io::stdout().flush().unwrap();
    step.elapsed() + Duration::from_millis(50)
}
