use std::fs;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(5);
const RESUME_MARGIN: i32 = 5;
const UNKNOWN: i32 = i32::MIN;

/// Reads the temperature on its own thread: nvidia-smi takes about 100 ms, and
/// calling it from the mining loop would leave the GPU idle for that long.
pub struct Monitor {
    latest: Arc<AtomicI32>,
}

impl Monitor {
    pub fn start(device_name: String) -> Self {
        let latest = Arc::new(AtomicI32::new(
            read_temperature(&device_name).unwrap_or(UNKNOWN),
        ));
        let shared = Arc::clone(&latest);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL_INTERVAL);
                let temperature = read_temperature(&device_name).unwrap_or(UNKNOWN);
                shared.store(temperature, Ordering::Relaxed);
            }
        });
        Self { latest }
    }

    pub fn temperature(&self) -> Option<i32> {
        match self.latest.load(Ordering::Relaxed) {
            UNKNOWN => None,
            temperature => Some(temperature),
        }
    }
}

/// Stops at `limit` and resumes a few degrees below it.  A limit of 0 disables the
/// guard, and so does a sensor that cannot be read.
pub struct Guard {
    limit: i32,
    cooling: bool,
}

impl Guard {
    pub fn new(limit: u8) -> Self {
        Self {
            limit: i32::from(limit),
            cooling: false,
        }
    }

    pub const fn cooling(&self) -> bool {
        self.cooling
    }

    pub const fn resume_at(&self) -> i32 {
        self.limit - RESUME_MARGIN
    }

    pub fn must_wait(&mut self, temperature: Option<i32>) -> bool {
        let threshold = if self.cooling {
            self.resume_at()
        } else {
            self.limit
        };
        self.cooling = self.limit > 0 && temperature.is_some_and(|t| t >= threshold);
        self.cooling
    }
}

fn read_temperature(device_name: &str) -> Option<i32> {
    if device_name.contains("NVIDIA") {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=name,temperature.gpu", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout), device_name)
    } else {
        read_amdgpu_sysfs()
    }
}

/// The hottest card with this name: OpenCL and nvidia-smi can number GPUs differently.
fn parse_nvidia_smi(output: &str, device_name: &str) -> Option<i32> {
    output
        .lines()
        .filter_map(|line| line.rsplit_once(", "))
        .filter(|(name, _)| name.trim() == device_name.trim())
        .filter_map(|(_, temperature)| temperature.trim().parse().ok())
        .max()
}

/// The hottest amdgpu edge sensor.  Only exists on Linux.
fn read_amdgpu_sysfs() -> Option<i32> {
    fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .filter_map(|card| fs::read_dir(card.path().join("device/hwmon")).ok())
        .flatten()
        .flatten()
        .filter(|hwmon| {
            fs::read_to_string(hwmon.path().join("name")).is_ok_and(|name| name.trim() == "amdgpu")
        })
        .filter_map(|hwmon| fs::read_to_string(hwmon.path().join("temp1_input")).ok())
        .filter_map(|millidegrees| millidegrees.trim().parse::<i32>().ok())
        .max()
        .map(|millidegrees| millidegrees / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: &str = "NVIDIA GeForce RTX 3070 Laptop GPU";

    #[test]
    fn stops_at_limit_and_resumes_below_margin() {
        let mut guard = Guard::new(85);
        assert!(!guard.must_wait(Some(84)));
        assert!(guard.must_wait(Some(85)));
        assert!(guard.must_wait(Some(81)));
        assert!(!guard.must_wait(Some(79)));
        assert!(!guard.must_wait(Some(84)));
    }

    #[test]
    fn resumes_exactly_at_the_margin_only_below_it() {
        let mut guard = Guard::new(85);
        assert!(guard.must_wait(Some(90)));
        assert!(guard.must_wait(Some(80)));
        assert!(!guard.must_wait(Some(79)));
    }

    #[test]
    fn zero_limit_and_lost_sensor_do_not_stop_mining() {
        assert!(!Guard::new(0).must_wait(Some(120)));
        let mut guard = Guard::new(85);
        assert!(guard.must_wait(Some(90)));
        assert!(!guard.must_wait(None));
    }

    #[test]
    fn picks_this_device_from_nvidia_smi() {
        let output = format!("NVIDIA GeForce GT 710, 40\r\n{NAME}, 71\r\n");
        assert_eq!(parse_nvidia_smi(&output, NAME), Some(71));
    }

    #[test]
    fn identical_cards_report_the_hottest() {
        let output = format!("{NAME}, 60\n{NAME}, 82\n{NAME}, 75\n");
        assert_eq!(parse_nvidia_smi(&output, NAME), Some(82));
    }

    #[test]
    fn unreadable_nvidia_smi_output_gives_no_temperature() {
        assert_eq!(parse_nvidia_smi(&format!("{NAME}, [N/A]\n"), NAME), None);
        assert_eq!(parse_nvidia_smi("", NAME), None);
        assert_eq!(parse_nvidia_smi("NVIDIA GeForce GT 710, 40\n", NAME), None);
    }
}
