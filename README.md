# Traffic Sign Recognition

A PyTorch CNN model for classifying traffic signs using the GTSRB dataset.

## Setup

### Installation
```
pip install -r requirements.txt
```

### Usage
```
python traffic_sign_recognition.py \
    --data_dir /path/to/GTSRB \
    --test_csv /path/to/GT-final_test.csv
```

Optional arguments:

--epochs: Number of training epochs (default: 5)
--batch_size: Batch size (default: 64)
--output_model: Path to save model (default: traffic_sign_model.pth)
