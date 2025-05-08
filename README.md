# OCR-Based Image Processing and Text Recognition System

This project implements an advanced system for extracting and mapping text from images using Optical Character Recognition (OCR). It preprocesses images to enhance text readability, performs OCR using the TrOCR model, maps detected text to a specific format, and communicates results to a remote server via REST API and socket programming. The system is designed for industrial applications requiring reliable text extraction.

## Features
- **Image Preprocessing**: Rotates, crops, and enhances images using OpenCV and PIL for optimal OCR performance.
- **OCR Processing**: Utilizes the TrOCR model for accurate text extraction.
- **Text Mapping**: Applies custom character mapping to format detected text.
- **Real-Time Communication**: Uses socket programming for device interaction and FastAPI for server communication.
- **Data Logging**: Saves results to CSV with timestamps and serial numbers.
- **Performance Optimization**: Implements multithreading and thread pool execution for efficient processing.

## Tech Stack
- **Languages**: Python
- **Libraries**: OpenCV, TrOCR, FastAPI, PIL, NumPy, Requests
- **Tools**: Socket Programming, Multithreading, REST API, CSV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rkarahul/OCR-Image-Processing-System.git
   cd OCR-Image-Processing-System
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the TrOCR model weights (if not automatically handled):
   ```bash
   # Follow instructions from https://huggingface.co/microsoft/trocr-large-printed
   ```

## Usage
1. Place an input image (e.g., `org.bmp`) in the `data/` directory.
2. Run the main script:
   ```bash
   python src/main.py
   ```
3. View results in the console and check `ocr_data.csv` for logged data.

## Example Output
```json
{
    "code": "mapped_text_12345678901",
    "cord": [1030, 870, 2660, 3224]
}
```

## Project Structure
- `src/`: Core source code for image processing, OCR, and networking.
- `data/`: Sample input/output data.
- `tests/`: Unit tests for key functions.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## Contact
For questions, reach out to rahul.kumarbihar245@gmail.com.
