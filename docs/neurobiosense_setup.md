# NeuroBioSense Dataset Setup

## Dataset Overview

The NeuroBioSense dataset ([Mendeley Data](https://data.mendeley.com/datasets/7md7yty9sk/2)) contains facial reactions of participants viewing different types of advertisements. This document provides instructions for setting up and preprocessing the dataset for the FER project.

## Dataset Statistics

- **Total Participants**: 58 (30 female, 23 male)
- **Age Range**: 18-66 years (mean = 27.4, SD = 11.3)
- **Total Videos**: 1,045
- **Processed Frames**: 10,450 (10 frames per video)

### Category Distribution
- Car and Technology: 20 participants (3,020 images, 2,510 interested)
- Food and Market: 20 participants (3,370 images, 1,860 interested)
- Cosmetics and Fashion: 18 participants (4,060 images, 2,650 interested)

## Download and Setup

1. Download the dataset from Mendeley Data:
```bash
# Create data directory
mkdir -p data/raw/neurobiosense

# Download using wget (or manually from the website)
wget -O neurobiosense_data.zip https://data.mendeley.com/datasets/7md7yty9sk/2/files/XXXXX
unzip neurobiosense_data.zip -d data/raw/neurobiosense/
```

2. Organize the raw data:
```bash
data/raw/neurobiosense/
├── car_tech/
│   ├── participant_001/
│   ├── participant_002/
│   └── ...
├── food_market/
│   ├── participant_021/
│   ├── participant_022/
│   └── ...
└── cosmetics_fashion/
    ├── participant_041/
    ├── participant_042/
    └── ...
```

## Preprocessing Pipeline

1. Frame Extraction
```bash
# Extract frames from videos
python scripts/video_processing/frame_extraction.py \
    --input_dir data/raw/neurobiosense \
    --output_dir data/processed/neurobiosense \
    --config configs/preprocessing_config.yaml
```

2. Data Split Creation
```bash
python scripts/data_processing/create_splits.py \
    --input_dir data/processed/neurobiosense \
    --output_dir data/processed/splits \
    --config configs/preprocessing_config.yaml
```

## Data Format

### Directory Structure After Processing
```
data/processed/neurobiosense/
├── train/
│   ├── interested/
│   └── not_interested/
├── val/
│   ├── interested/
│   └── not_interested/
└── test/
    ├── interested/
    └── not_interested/
```

### Image Naming Convention
```
{participant_id}_{video_id}_frame_{frame_number}.jpg
```

### Metadata Format (CSV)
```csv
image_path,participant_id,category,reaction_type,split
path/to/image.jpg,001,food_market,interested,train
```

## Usage Notes

1. **Video Quality**
   - Original videos are recorded at 30 FPS
   - Resolution: 1920x1080
   - Format: MP4 (H.264 codec)

2. **Frame Extraction**
   - 10 frames extracted per video at equal intervals
   - Faces detected and cropped to 224x224 pixels
   - Frames with no detected faces are skipped

3. **Data Splits**
   - Train: Food/Market category (3,370 images)
   - Validation: Cosmetics/Fashion category (4,060 images)
   - Test: Car/Technology category (3,020 images)
   - This ensures testing on entirely different content

## License and Citation

The NeuroBioSense dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). When using this dataset, please cite:

```bibtex
@dataset{kocacinar_burak_2024_7md7yty9sk,
  author       = {Kocacinar, Burak and
                  Others},
  title        = {{NeuroBioSense Dataset}},
  month        = jan,
  year         = 2024,
  publisher    = {Mendeley Data},
  version      = {2},
  doi          = {10.17632/7md7yty9sk.2},
  url          = {https://data.mendeley.com/datasets/7md7yty9sk/2}
}
```

## Troubleshooting

### Common Issues

1. **Video Loading Fails**
   - Ensure ffmpeg is installed
   - Check video codec compatibility
   - Try converting to mp4: `ffmpeg -i input.avi output.mp4`

2. **Face Detection Issues**
   - Adjust minimum face size in config
   - Try different face detection models
   - Check lighting conditions

3. **Memory Issues**
   - Process videos in batches
   - Reduce frame extraction rate
   - Use memory-efficient processing

### Error Messages

| Error | Solution |
|-------|----------|
| `VideoCapture failed` | Check video file integrity |
| `No faces detected` | Adjust face detection parameters |
| `Out of memory` | Reduce batch size or image resolution |