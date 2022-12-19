# egui-video
### current caveats
 - need to compile in `release` or `opt-level=3` otherwise limited playback performance
 - bad (playback, seeking) performance with large resolution streams
 - seeking can be slow (is there better way of dropping packets?)
 - depending on the specific stream, seeking can fail and mess up playback/seekbar (something to do with dts?)
 - no audio playback
