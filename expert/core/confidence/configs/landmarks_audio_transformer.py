batch_size = 16

# target video fps (real video fps --> target video fps)
video_fps = 10
# target audio fps (real audio fps --> target audio fps)
audio_fps = 16000
# time window (seconds)
window_secs = 10
# target window size (frames)
window = video_fps * window_secs


# dataset config
dataset = dict(
    type="Interviews",
    root="data/interviews",
    window=window,
    video_fps=video_fps,
    # audio_fps=1 to drop audio
    audio_fps=audio_fps,
    # train valid split
    split=dict(valid_size=0.2, balanced_valid_set=True, by_file=True),
)

# number of features (landmarks + angles + audio features == features_dims)
features_dims = 1455
# embed size (features_dims --(Linear)--> embed_dims)
embed_dims = 512
# number of target classes (binary == 2)
num_classes = 2


# model pipeline
model = dict(
    type="LieDetector",
    # model to extract features from video
    video_model=dict(
        type="FaceLandmarks",
        window=window,
        init=True,
        init_cfg=dict(
            type="PretrainedInit", checkpoint="./weights/angles_regressor.pth"
        ),
    ),
    # model to extract features from audio
    audio_model=dict(
        type="AudioFeatures",
        fps=window_secs,
        chunk_length=1,
        sr=audio_fps,
        normalization=True,
    ),
    features_dims=features_dims,
    embed_dims=embed_dims,
    # time model to extract time-dependent features from time-independent ones
    time_model=dict(
        type="TransformerEncoder",
        encoder_layer=dict(
            type="TransformerEncoderLayer",
            d_model=embed_dims,
            nhead=16,
            dim_feedforward=embed_dims * 4,
            dropout=0.5,
            batch_first=True,
        ),
        num_layers=1,
        norm=dict(type="LayerNorm", normalized_shape=embed_dims),
    ),
    # classifier
    cls_head=dict(
        type="Linear",
        in_features=embed_dims,
        out_features=num_classes,
    ),
    init_cfg=dict(
        type="PretrainedInit",
        checkpoint="./weights/landmarks_audio_transformer.pth",
    ),
)

runner = dict(
    type="LieDetectorRunner",
)
