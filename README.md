# OCR-Based-Image-Processing-and-Text-Recognition-System
This project implements an advanced system for extracting and mapping text from images using Optical Character Recognition (OCR). It preprocesses images to enhance text readability, performs OCR using the TrOCR model, maps detected text to a specific format, and communicates results to a remote server via REST API and socket programming. The system is designed for industrial applications requiring reliable text extraction.

Features





Image Preprocessing: Rotates, crops, and enhances images using OpenCV and PIL for optimal OCR performance.



OCR Processing: Utilizes the TrOCR model for accurate text extraction.



Text Mapping: Applies custom character mapping to format detected text.



Real-Time Communication: Uses socket programming for device interaction and FastAPI for server communication.



Data Logging: Saves results to CSV with timestamps and serial numbers.



Performance Optimization: Implements multithreading and thread pool execution for efficient processing.

Tech Stack





Languages: Python



Libraries: OpenCV, TrOCR, FastAPI, PIL, NumPy, Requests



Tools: Socket Programming, Multithreading, REST API, CSV

Installation





Clone the repository:

git clone https://github.com/yourusername/OCR-Image-Processing-System.git
cd OCR-Image-Processing-System



Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



Download the TrOCR model weights (if not automatically handled):

# Follow instructions from https://huggingface.co/microsoft/trocr-large-printed

Usage





Place an input image (e.g., org.bmp) in the data/ directory.



Run the main script:

python src/main.py



View results in the console and check ocr_data.csv for logged data.

Image Preprocessing Steps

The project applies several preprocessing steps to enhance the image for better OCR accuracy. Below are the steps with corresponding images:





Initial Image (Before Cropping): The original image after rotation and shifting, showing the text region before cropping.




After Cropping: The image is cropped to focus on the text region, removing unnecessary areas.




Contrast Enhancement: The cropped image undergoes contrast enhancement using PIL to make the text more readable.




Bilateral Filtering and Initial Erosion-Dilation: The image is processed with Gaussian blur, bilateral filtering, and an initial erosion-dilation step to reduce noise while preserving edges.




Final Erosion-Dilation: A final erosion-dilation step is applied to further refine the image, followed by resizing for OCR input.


These steps ensure the text is clear and optimized for the TrOCR model to accurately extract the text.

Example Output

{
    "code": "mapped_text_12345678901",
    "cord": [1030, 870, 2660, 3224]
}

Project Structure





src/: Core source code for image processing, OCR, and networking.



data/: Sample input/output data.



docs/images/: Images for documentation (e.g., preprocessing steps).



tests/: Unit tests for key functions.



requirements.txt: Python dependencies.



README.md: Project documentation.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

Contact

For questions, reach out to your.email@example.com.
