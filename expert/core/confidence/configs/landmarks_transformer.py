batch_size = 16

# target video fps (real video fps --> target video fps)
video_fps = 10
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
    audio_fps=1,
    # train valid split
    split=dict(valid_size=0.2, balanced_valid_set=True, by_file=True),
)

# number of features (landmarks + angles == features_dims)
features_dims = 1437
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
            type="PretrainedInit", checkpoint="weights/angles_regressor.pth"
        ),
    ),
    features_dims=features_dims,
    embed_dims=embed_dims,
    # time model to extract time-dependent features from time-independent ones
    time_model=dict(
        type="TransformerEncoder",
        encoder_layer=dict(
            type="TransformerEncoderLayer",
            d_model=embed_dims,
            nhead=8,
            dim_feedforward=embed_dims * 4,
            dropout=0.0,
            batch_first=True,
        ),
        num_layers=3,
        norm=dict(type="LayerNorm", normalized_shape=embed_dims),
    ),
    # classifier
    cls_head=dict(
        type="Linear", in_features=embed_dims, out_features=num_classes
    ),
    init=True,
    init_cfg=dict(
        type="PretrainedInit", checkpoint="weights/landmarks_transformer.pth"
    ),
)

runner = dict(type="LieDetectorRunner")
