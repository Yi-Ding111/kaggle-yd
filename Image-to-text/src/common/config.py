from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# re-sized image's size
cfg.image_height = 640
cfg.image_width = 640

# label mapping (encoder)
for key, value in {
    "dot": 0,
    "scatter": 1,
    "horizontal_bar": 2,
    "line": 3,
    "vertical_bar": 4,
}.items():
    setattr(cfg, key, value)

# decoder
# print(cfg.__dict__["0"])
for key, value in {
    "0": "dot",
    "1": "scatter",
    "2": "horizontal_bar",
    "3": "line",
    "4": "vertical_bar",
}.items():
    setattr(cfg, key, value)



# Image recognation (ViT)
# Dataset parameters
cfg.vit_image_cls_height = 320
cfg.vit_image_cls_width = 320
cfg.vit_num_workers = 4
# Total dataset size multiplier
cfg.vit_ds_multiplier = 2.0
# DataModule parameters
cfg.vit_initial_augment_prob = 0.8
cfg.vit_final_augment_prob = 0.3
# pretrained model local path
cfg.vit_local_config_path = "./pretrained_models/vit-base-patch16-224-in21k"
# trainer parameters
cfg.vit_num_epochs = 50
cfg.vit_batch_size = 32




# Object detection (YOLO11)
for key, value in {
    "plot": 0,
    "chart_title": 1,
    "axis_title": 2,
    "tick_label": 3,
    "marker": 4,
    "visual_element":5
}.items():
    setattr(cfg, key, value)
