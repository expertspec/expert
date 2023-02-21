# target video fps (real video fps --> target video fps)
video_fps = 10
# target audio fps (real audio fps --> target audio fps)
audio_fps = 16000
# time window (seconds)
window_secs = 10
# target window size (frames)
window = video_fps * window_secs


type = "Interviews"
root = "data/interviews"
window = window
video_fps = video_fps
# audio_fps=1 to drop audio
audio_fps = audio_fps
# train valid split
split = dict(valid_size=0.2, balanced_valid_set=True, by_file=True)
