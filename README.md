# Pixelizing - Image Processing Toolkit

## 🚀 Overview
Pixelizing is a comprehensive image processing application designed to demonstrate and apply various image manipulation techniques such as noise addition, filtering, edge detection, and more. This toolkit is developed using Python and OpenCV, along with custom algorithms crafted from scratch, leveraging Object-Oriented Programming (OOP) principles for modularity, scalability, and maintainability.

<img src="https://github.com/user-attachments/assets/0aa92ef3-5098-45b9-a77f-ede290870fc6" width="75%" alt="App UI">

## 🎯 Features

### Noise Addition
- **Uniform Noise**: Adds noise with a constant probability density function, affecting the image uniformly across its entirety. This is ideal for testing the effectiveness of various filters under uniform noise conditions.
  <img src="https://github.com/user-attachments/assets/ba1affb7-feec-4862-b260-e05c5355ef18" width="50%" alt="Uniform Noise">

### Filters
Filters are implemented to reduce noise and enhance image clarity:
- **Median Filter**: Utilizes a non-linear digital filtering technique, replacing each pixel value with the median value of neighboring pixels. This is particularly effective in removing 'salt and pepper' noise without significantly reducing the sharpness of the image.

  <img src="https://github.com/user-attachments/assets/3cca2a4a-70a8-40cd-8256-d0acc3492d8c" width="50%" alt="Median Filter">

### Edge Detection
Edge detection techniques highlight significant transitions in intensity in an image:
- **Prewitt Edge Detection**: Identifies edges using the Prewitt operator which emphasizes horizontal and vertical changes. This method uses a set of optimized kernels to capture more subtle changes in image texture compared to simpler methods like the Roberts operator.

  <img src="https://github.com/user-attachments/assets/9be6e416-ad0b-4714-9b25-5d6825705cf6" width="50%" alt="Prewitt Edge Detection">

### Image Enhancements
- **Histogram and CDF**: Visualizes the distribution of pixel intensities (histogram) and the cumulative distribution function (CDF) which is used for histogram equalization to improve image contrast.
  
  <img src="https://github.com/user-attachments/assets/7ae6c22a-2feb-4d18-a9b5-0c17e398e114" width="50%" alt="Histogram and CDF">

- **Image Equalization**: Applies histogram equalization to redistribute the image's light pixels, enhancing contrast especially in darker areas.
  
  <img src="https://github.com/user-attachments/assets/c5817782-4b50-475b-920e-3ebc948ee1f7" width="50%" alt="Image Equalization">

### Color to Grayscale
- **Grayscaling**: Converts color images to grayscale, reducing complexity and focusing on intensity alone, which is crucial for various processing tasks like edge detection.
  
  <img src="https://github.com/user-attachments/assets/c15c2fad-471d-4632-8f0d-593d2b91a812" width="50%" alt="Grayscaling">

## 📌 Installation
```bash
pip install -r requirements.txt
```

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries & Tools:** OpenCV, NumPy, PyQt for GUI. Custom algorithms are extensively used to implement features that require tailored processing beyond the capabilities of standard OpenCV functions.

## 📊 Usage
To start using the Pixelizing toolkit, run the main application file after installation:
```python
python main.py
```
Interact with the GUI to choose images and apply different processing techniques.

## Contributors
<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br />
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/nariman-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126989278?v=4" width="150px;" alt="Nariman Ahmed"/>
        <br />
        <sub><b>Nariman Ahmed</b></sub>
      </a>
    </td>    
    <td align="center">
      <a href="https://github.com/nancymahmoud1" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125357872?v=4" width="150px;" alt="Nancy Mahmoud"/>
        <br />
        <sub><b>Nancy Mahmoud</b></sub>
      </a>
    </td>    
    <td align="center">
      <a href="https://github.com/madonna-mosaad" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="150px;" alt="Madonna Mosaad"/>
        <br />
        <sub><b>Madonna Mosaad</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>

---
