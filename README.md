# Bird Brain

A bird species classifier that identifies 200 species from a photo using transfer learning on ResNet18, trained on the CUB-200-2011 dataset.

## Demo
[https://github.com/vvkon/bird-species-classifier/raw/main/demo.mp4](https://youtu.be/W5sea31q980)

## Model Details
- Architecture: ResNet18 (fine-tuned)
- Dataset: CUB-200-2011 (11,788 images, 200 species)
- Test Accuracy: 68%
- Framework: PyTorch

## Tech Stack
Python, PyTorch, Flask, HTML/CSS/JS

## How to Run
1. Clone the repo
2. Create a virtual environment and activate it
3. pip install -r requirements.txt
4. python app.py
5. Go to http://127.0.0.1:5000
