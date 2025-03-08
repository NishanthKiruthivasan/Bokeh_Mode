# Bokeh_Mode

This repository provides a Python-based implementation to apply a **variable bokeh effect** to images. It uses **depth estimation** and **human detection** to blur the background while keeping humans in focus. This code currently works for only photos with people facing fully forward. This will be rectified in future commits though.

## Features
- Uses **MiDaS** for depth estimation.
- Uses **YOLOv5** for human detection.
- Applies variable blur intensity based on depth information.
- Outputs a processed image with a bokeh effect.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/NishanthKiruthivasan/Bokeh_Mode.git
   cd Bokeh_Mode
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies
This project requires the following libraries:
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `timm`
- `ultralytics`

You can install them manually if needed:
```sh
pip install torch torchvision opencv-python numpy timm ultralytics
```

## Usage
1. Place your input image in the project folder.
2. Run the script with:
   ```sh
   python bokeh.py --input input.jpg --output output.jpg
   ```
3. The processed image will be saved as `output.jpg`.

## Code Explanation
- **Depth Estimation:** Uses MiDaS to estimate the depth of objects in an image.
- **Human Detection:** Uses YOLOv5 to detect and extract bounding boxes for humans.
- **Variable Blur:** Applies Gaussian blur with varying intensity, ensuring humans remain sharp.

## Example
**Input.jpg**

![input](https://github.com/user-attachments/assets/e85cc017-03a0-45a7-8987-96ff635e3b31)

**Output.jpg**

![output](https://github.com/user-attachments/assets/f1605945-636b-4621-ac11-abdd1eaae180) 



## Troubleshooting
- If `ModuleNotFoundError` occurs, ensure all dependencies are installed.
- If the script runs slowly, try reducing the input image size.

## License
This project is open-source and available under the **MIT License**.

## Contributing
Feel free to submit pull requests or report issues!

---
Made with ❤️ by Nishanth Kiruthivasan
