# egui-video, a video playing library for [`egui`](https://github.com/emilk/egui)
[![crates.io](https://img.shields.io/crates/v/egui-video)](https://crates.io/crates/egui-video)
[![docs](https://docs.rs/egui-video/badge.svg)](https://docs.rs/egui-video/latest/egui_video/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/n00kii/egui-video/blob/main/README.md)

https://github.com/n00kii/egui-video/assets/57325298/c618ff0a-9ad2-4cf0-b14a-dda65dc54b23

plays videos in egui from file path or from bytes

## dependancies:
 - requires ffmpeg 6 or 7. follow the build instructions [here](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building)
 - requires sdl2. by default, a feature is enabled to automatically compile it for you, but you are free to disable it and follow [these instructions](https://github.com/Rust-SDL2/rust-sdl2#requirements)

## Installation

### macOS Setup

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install required dependencies**:
   ```bash
   brew install pkg-config cmake ffmpeg
   ```

3. **Set up environment variables** (add to your `~/.zshrc` or `~/.bash_profile`):
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
   export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
   ```

4. **Clone and run the project**:
   ```bash
   git clone https://github.com/n00kii/egui-video.git
   cd egui-video
   ./run.sh  # Use the provided script (see below)
   ```

### Quick Start Script

We've included a `run.sh` script that handles the environment setup and runs the example:

```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the example
./run.sh
```

The script automatically:
- Sets the required environment variables
- Builds and runs the example in release mode
- Uses the correct feature flags to avoid SDL2 compilation issues

### Manual Running

If you prefer to run manually:

```bash
# Set environment variables
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"

# Run the example
cargo run --example main --release --no-default-features
```

**Note**: The `--no-default-features` flag is important on macOS to avoid SDL2 compilation issues. The bundled SDL2 feature has compatibility problems with newer CMake versions.

## usage:
```rust
/* called once (top level initialization) */

{ // if using audio...
    let audio_device = egui_video::AudioDevice::new()?;
    
    // don't let audio_device drop out of memory! (or else you lose audio)

    add_audio_device_to_state_somewhere(audio_device);
}
```
```rust
/* called once (creating a player) */

let mut player = egui_video::Player::new(ctx, my_media_path)?;

{ // if using audio...
    player = player.with_audio(&mut my_state.audio_device)
}
```
```rust
/* called every frame (showing the player) */
player.ui(ui, player.size);
```

## Running the Example

The example application provides a GUI where you can:
1. Click the text field to select a video file
2. Click "load" to load and play the video
3. Use the controls to play, pause, seek, and adjust volume
4. View video information like frame rate, size, and duration

Supported video formats: mp4, gif, webm, mkv, ogg

## contributions
are welcome :)

### current caveats
 - need to compile in `release` or `opt-level=3` otherwise limited playback performance
