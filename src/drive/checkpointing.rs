use super::{driver::State, format_duration, RecoveryFile, Step};
use crate::{
    algo::{FileReader, FileWriter},
    log_trace, Algo, DriverError, FileFormat,
};
use std::{
    error::Error,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct CheckpointData {
    pub(super) path: PathBuf,
    pub(super) every: Duration,
    pub(super) next_elapsed: Duration,
}

#[derive(Clone, Debug)]
pub struct Recovery {
    pub(super) path: PathBuf,
    pub(super) action: RecoveryFile,
}

impl CheckpointData {
    pub fn needs_checkpoint(&mut self, step: &Step) -> bool {
        step.elapsed() >= self.next_elapsed
    }

    pub fn checkpoint<A>(&mut self, algo: &mut A, step: &Step) -> Result<(), DriverError>
    where
        A: Algo,
    {
        if !self.needs_checkpoint(step) {
            return Ok(());
        }
        self.next_elapsed += self.every;

        let path = replace_file_name_parameters(&self.path, step.iteration(), step.elapsed());
        Self::write_to_file(&path, algo)?;

        let path = replace_file_ext(&self.path);
        let path = replace_file_name_parameters(&path, step.iteration(), step.elapsed());
        Self::write_to_file(&path, step)?;
        Ok(())
    }

    fn write_to_file<T>(path: &Path, t: &T) -> Result<(), DriverError>
    where
        T: FileWriter,
    {
        log_trace!("[     ] writing file {path:?}");
        let format = FileFormat::from_path(path);
        let file = create_file_with_dirs(path).map_err(|e| convert(e.into(), path))?;
        let mut w = BufWriter::new(file);
        t.write_file(&format, &mut w)
            .map_err(|e| convert(e.into(), path))?;
        w.flush().map_err(|e| convert(e.into(), path))?;
        Ok(())
    }
}

impl Recovery {
    pub fn recover<A>(&self, algo: &mut A, state: &mut State) -> Result<(), DriverError>
    where
        A: Algo,
    {
        if matches!(self.action, RecoveryFile::Ignore) || state.step.iteration() > 0 {
            return Ok(());
        }

        let path_check =
            replace_file_name_parameters(&self.path, state.step.iteration(), state.step.elapsed());
        if path_check != self.path {
            return Err(convert(
                Box::new(DriverError::General(
                    "recovery filename cannot contain replaceable parameters".to_string(),
                )),
                &self.path,
            ));
        }
        // recover Algo
        self.read_from_file(&self.path, algo)?;
        let path = replace_file_ext(&self.path);

        // recover Step
        self.read_from_file(&path, &mut state.step)?;
        Ok(())
    }

    fn read_from_file<T>(&self, path: &Path, t: &mut T) -> Result<(), DriverError>
    where
        T: FileReader,
    {
        log_trace!("[     ] reading file {path:?}");
        let format = FileFormat::from_path(path);
        let file = File::open(path).map_err(|e| convert(Box::new(e), path))?;
        let mut r = BufReader::new(file);
        t.read_file(&format, &mut r)
            .map_err(|e| convert(Box::new(e), path))?;
        Ok(())
    }
}

pub(crate) fn replace_file_ext(filename: &Path) -> PathBuf {
    filename.with_extension(FileFormat::Control.to_extension())
}

fn create_file_with_dirs(path: &Path) -> std::io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    File::create(path) // Then create the file
}

// replace ${iter} and ${elapsed} ${time} with their values
pub(crate) fn replace_file_name_parameters(
    filename: &Path,
    iter: usize,
    elapsed: Duration,
) -> PathBuf {
    let elapsed = format_duration(elapsed).to_string();
    let iter = format!("{iter:>07}");
    let pid = std::process::id().to_string();

    filename
        .to_string_lossy()
        .replace("{iter}", &iter)
        .replace("{elapsed}", &elapsed)
        .replace("{pid}", &pid)
        .into()
}

pub(crate) fn convert(e: Box<dyn Error + Send + Sync + 'static>, path: &Path) -> DriverError {
    DriverError::Checkpoint {
        filename: path.to_string_lossy().to_string(),
        error: Arc::from(e),
    }
}
