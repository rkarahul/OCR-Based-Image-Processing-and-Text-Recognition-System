import cv2
import numpy as np
import io
import base64
import time
import socket
import platform
import subprocess
import os
import sys
import requests
import multiprocessing
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import csv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time
import traceback
from threading import Thread
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import json
import uvicorn
from PIL import Image, ImageEnhance


# Define the ImageRequest model for the FastAPI endpoint
class ImageRequest(BaseModel):
    image: str

# Initialize FastAPI app
app = FastAPI()

# Global variables
mapped_text = ''
response = ''
response_history = []

### Define the expiration date for the script
##exp_date = datetime(2025, 4, 30)
##
### Check if the script has expired
##if datetime.now() > exp_date:
##    print("The code has expired. Deleting the script...")
##    os.remove(sys.argv[0])  # Delete the script file
##    sys.exit()

# Utility functions

def get_available_ips():
    # Get the host name and the IP address of the connected WiFi
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)

    # Print the connected WiFi IP address
    print(f"Connected WiFi IP Address: {host_ip}")

    # Get the available IPs using a subprocess based on the platform
    if platform.system().lower() == "windows":
        try:
            result = subprocess.check_output(['arp', '-a'], text=True)
            # Extract IP addresses from the result
            available_ips = [line.split(' ')[0] for line in result.splitlines() if '.' in line]
        except subprocess.CalledProcessError:
            available_ips = []
    else:
        # For non-Windows systems, use a simple socket-based approach
        try:
            available_ips = [socket.gethostbyname(socket.gethostname() + f".{i}") for i in range(1, 255)]
        except socket.error:
            available_ips = []

    # Print the available IP addresses
    print("Available IP Addresses:")
    for ip in available_ips:
        print(ip)

    return host_ip

def check_port(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((ip, port))
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False

def trim_until_condition_met(text):
    while len(text) >= 11:
        # Check last 11th character and 10th character from end
        last_11th_char = text[-11]
        last_10th_chars = text[-10]
        if last_11th_char == '2' and last_10th_chars == '5':
            return text  # Condition met, return current text
        text = text[:-1]  # Remove one character from end
    return text
def map_detected_text(ocr_text):
    ocr_text = trim_until_condition_met(ocr_text)
    print("ocr_text : ", ocr_text)

    # Get the last 11 characters
    text1 = ocr_text[-11:]
    # Get characters from the 17th last to the 11th last
    text2 = ocr_text[-17:-11]

    character_mapping = {
        'O': '0',
        'o': '0',
        'i': '1',
        'I': '1',
        'L': '1',
        'Z': '2',
        'z': '2',
        'S': '5',
        's': '5',
        'B': '8',
        'b': '6',
        'l': '1',
        'G': '6',
        'g': '9',
        'A': '4',
        '/': '7',
        'R': '2',
    }

    # Only apply character_mapping to text1
    mapped_text1 = ''.join(character_mapping.get(char, char) for char in text1)
    # Use text2 directly without mapping
    mapped_text2 = text2

    mapped_text = mapped_text2 + mapped_text1
    print("Final Text mapped : ", mapped_text)

    # Check if all characters in the string are digits
    if mapped_text[-11:].isdigit():
        return mapped_text
    else:
        return "InvalidCode"


def save_to_csv(detected_text):
    serial_number = len(open('ocr_data.csv').readlines())

    with open('ocr_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['Serial Number', 'Timestamp', 'Detected Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if serial_number == 0:
            writer.writeheader()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow({'Serial Number': serial_number + 1, 'Timestamp': timestamp, 'Detected Text': detected_text})
        
def crop_rotated_rect_old(center, size, angle, image):
    center_x, center_y = center
    width, height = size

    # Create the rotated rectangle
    rect = ((center_x, center_y), (width, height), angle)

    # Calculate the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Get image dimensions
    (h, w) = image.shape[:2]

    # Rotate the entire image
    rotated_image = cv2.warpAffine(image, rot_mat, (w, h))

    # Get the bounding box of the rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Get the bounding box for cropping
    x, y, w, h = cv2.boundingRect(box)
    print(x, y, w, h)
    x, y, w, h=int(x), int(y), int(w), int(h)
    # Crop the rotated rectangle
    cropped_image = rotated_image[y:y+h, x:x+w]
    cv2.imwrite("0unshifted0.png",cropped_image)

    return cropped_image
def crop_rotated_rect(center, size, angle, image):
    center_x, center_y = center
    width, height = size

    # Create the rotated rectangle
    rect = ((center_x, center_y), (width, height), angle)

    # Calculate the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Get image dimensions
    (h, w) = image.shape[:2]

    # Rotate the entire image
    rotated_image = cv2.warpAffine(image, rot_mat, (w, h))

    # Save and debug rotated image
    rotated_image=rotated_image[900:1850,:-100]
    cv2.imwrite("rotated_image.png", rotated_image)

##    # Get the bounding box of the rotated rectangle
##    box = cv2.boxPoints(rect)
##    box = np.int0(box)
##
##    # Get the bounding box for cropping
##    x, y, w, h = cv2.boundingRect(box)
##    print(f'Bounding box: x={x}, y={y}, w={w}, h={h}')
##
##    # Ensure bounding box is within image dimensions
##    x, y, w, h = max(0, x), max(0, y), min(w, rotated_image.shape[1] - x), min(h, rotated_image.shape[0] - y)
##
##    # Crop the rotated rectangle
##    cropped_image = rotated_image[y:y+h, x:x+w]
##
##    # Check if cropped image is valid
##    if cropped_image.size == 0:
##        print("Cropped image is empty!")
##    else:
##        cv2.imwrite("0unshifted0.png", cropped_image)

    return rotated_image
def shift_image(image_data):
    # Open the source image
    source_img = Image.fromarray(image_data)

    # Define the polygon points
    old_polygon_points = np.array([[2229, 2802], [2573, 2554], [2960, 2325], [3318, 2139], [3681, 1991], 
                  [4139, 1828], [4560, 1709], [4817, 1652], [4832, 2062], [4364, 2186], 
                  [3862, 2377], [3433, 2578], [3022, 2807], [2463, 3122]])
    polygon_points_old = np.array([[2120, 1270],[2425, 1308],[2740, 1375],[3041, 1461],[3337, 1580],[3695, 1733],
                               [3838, 1819],[3648, 2129],[3308, 1962],[2969, 1833],[2616, 1723],[2124, 1633]])
    polygon_pointss = np.array([[2832, 1515],[3078, 1457],[3368, 1409],[3652, 1384],[3956, 1365],[4236, 1370],[4352, 1380],[4309, 1688],
                               [4082, 1655],[3715, 1659],[3377, 1684],[3040, 1746],[2890, 1775]])
    polygon_points_old = np.array([[2002, 1630],
 [2359, 1505],
 [2726, 1399],
 [2726, 1399],
 [3131, 1326],
 [3131, 1326],
 [3483, 1302],
 [3483, 1302],
 [3831, 1293],
 [4140, 1322],
 [4140, 1317],
 [4087, 1621],
 [3749, 1602],
 [3749, 1602],
 [3348, 1611],
 [2933, 1669],
 [2620, 1741],
 [3136, 1824],#
 [2827, 1857],#                              
 [2330, 1838],
 [2113, 1910]])
    polygon_points = np.array([[1080, 2663],
 [1387, 2513],
 [1725, 2400],
 [2078, 2325],
 [2453, 2280],
 [2820, 2273],
 [3256, 2310],
 [3211, 2708],
 [2903, 2678],
 [2528, 2678],
 [2108, 2723],
 [1680, 2828],
 [1282, 3016]])
    
    source_img_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    height, width, _ = image_data.shape

    # Initialize an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the polygon on the mask
    mask=cv2.fillPoly(mask, [polygon_points], color=255)

    # Convert the mask to PIL Image format
    contour_img = Image.fromarray(mask)

    assert source_img.size == contour_img.size

    contour_arr = np.array(contour_img) != 0  # convert to boolean array
    col_offsets = np.argmax(
        contour_arr, axis=0
    )  # find the first non-zero row for each column
    assert len(col_offsets) == source_img.size[0]  # sanity check
    min_nonzero_col_offset = np.min(
        col_offsets[col_offsets > 0]
    )  # find the minimum non-zero row

    # Create a new target image
    target_img = Image.new("RGB", source_img.size, (255, 255, 255))
    for x, col_offset in enumerate(col_offsets):
        offset = col_offset - min_nonzero_col_offset if col_offset > 0 else 0
        target_img.paste(
            source_img.crop((x, offset, x + 1, source_img.size[1])), (x, 0)
        )
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    #cv2.imwrite("unbeforecropshifted0.png",target_img)
    cv2.imwrite("before_crop.png",target_img)
    target_img = target_img[2309:2707, 1075:2943]
    # target_img=target_img[2270:2600,1100:3260]#target_img[1150:1800,1700:3850]#change corr if text is croped1760
    cv2.imwrite("after_crop.png",target_img)
    #cv2.imwrite("unshifted0.png",target_img)
    #target_img=target_img[1650:2200,2900:4750]
    #target_img = crop_rotated_rect((2120,1510), (1290,995), 25, target_img)#center, size(w,h), angle, image
    return target_img

def erode_then_dilate(image, kernel_size=(5, 5), iterations=2):
    """
    Perform erosion followed by dilation on an image.

    Parameters:
    - image: Input image
    - kernel_size: Size of the kernel (default is (5, 5))
    - iterations: Number of times erosion and dilation are applied (default is 1)

    Returns:
    - result: Image after erosion followed by dilation
    """
    #image = cv2.GaussianBlur(image, (5,5), 0)
    # Create the structuring element (kernel)
    kernel = np.ones(kernel_size, np.uint8)

    # Perform erosion
    result = cv2.erode(image, kernel, iterations=iterations)
    
    # Perform dilation
    result = cv2.dilate(result, kernel, iterations=2)
    result = cv2.resize(result, (575, 160))
    cv2.imwrite("erosiondilation.png",result)
    return result

def blur_and_bilateral_filter(img):
    # Read the image
    #img = cv2.imread(image_path)

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Apply bilateral filter
    bilateral_img = cv2.bilateralFilter(blurred_img, 15, 75, 75)
    #cv2.imwrite("filter_img.jpg", bilateral_img)
    
    # Apply erode then dilate
    erode_dilate_img = erode_then_dilate(bilateral_img)
    cv2.imwrite("erode_dilate_img.png",erode_dilate_img)
    return erode_dilate_img

def enhance_contrast_opencv_image(cv2_image):
    # Convert the OpenCV image (BGR) to a PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    # Enhance the contrast of the PIL image
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil_image = enhancer.enhance(3)  # Adjust the factor as needed

    # Convert the enhanced PIL image back to an OpenCV image
    enhanced_cv2_image = cv2.cvtColor(np.array(enhanced_pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("enhanced_cv2_image.png",enhanced_cv2_image)

    return enhanced_cv2_image

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
tt=cv2.imread(r"org.bmp")
t_values = processor(tt, return_tensors="pt").pixel_values
generated_t = model.generate(t_values)
t_text = processor.batch_decode(generated_t, skip_special_tokens=True)[0]
##global output_queue
##output_queue = Queue()
##output_queue.put('')
##processing_complete = True
def perform_ocr_from_image(image_data):#,output_queue
##    global processing_complete
    image_data=shift_image(image_data)

    image_data=enhance_contrast_opencv_image(image_data)
    image_data=blur_and_bilateral_filter(image_data)
    
    pixel_values = processor(images=image_data, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("generated_text : ",generated_text)
    generated_text = generated_text.replace(":", "").replace("-", "")
    generated_text = generated_text.replace(" ", "")
    if generated_text and len(generated_text)>=16:
        mapped_text = map_detected_text(generated_text)
    else:
        mapped_text=""
    print("new mapped_text :  ",mapped_text)
##    output_queue.put( mapped_text)
##    processing_complete = True
    return mapped_text

def string_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def send_command_and_receive_response(host, port, command):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.sendall(command.encode())
        response = client_socket.recv(1024).decode()
        return response.strip()
    except Exception as e:
        return f"Error: {e}"
    finally:

        client_socket.close()

def send_data_to_server(mapped_text, response):
    url = "http://172.16.32.15:9001/hard_turning/laser_scanned_data/"
    payload = {
        "machine_name": "MC-LMM014",
        "model_number": "TS02",
        "old_part_name": mapped_text, #"oldpart00011", #
        "new_part_name": response, #"newpart00011", #
        "previous_stage_condition": False
    }
    
    try:
        # Convert payload to JSON string
        json_payload = json.dumps(payload)
        if mapped_text!="InvalidCode" or mapped_text!="":
            # Send POST request with JSON payload and set Content-Type header
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                print("Data sent successfully to the machine.")
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
                print(f"Response content: {response.content}")
    except requests.RequestException as e:
        print(f"Error occurred: {e}")

#thread pool executor to run CPU-bound tasks in separate threads
executor = ThreadPoolExecutor()   
invalid_count=0

def process_image(image):
    global invalid_count

    try:
        s = time.time()

        # If image is a file path, read it
        if isinstance(image, str):
            image = Image.open(image)

        # Convert PIL Image to NumPy array
        numpy_image = np.array(image)

        # Validate dimensions
        if len(numpy_image.shape) < 2:
            raise ValueError("Invalid image shape, expected 2D or 3D image array")

        # Crop columns from image
        numpy_image = numpy_image[:, 1100:]

        # Coordinates to include in result
        crop_corr = [1030, 870, 2660, 3224]  # x1, y1, x2, y2

        # Rotate image
        pil_image = Image.fromarray(numpy_image)
        rotated_image = pil_image.rotate(-45, expand=True)
        numpy_image = np.array(rotated_image)

        # Save image for debugging
        cv2.imwrite("test.png", numpy_image)

        # Run OCR
        last_ocr = perform_ocr_from_image(numpy_image)
        print("OCR Output:", last_ocr)

        e = time.time()
        print("Total processing time:", e - s)

        if last_ocr == "InvalidCode":
            invalid_count += 1

        print("Invalid count:", invalid_count)

        dict_data = {"code": last_ocr, "cord": crop_corr}
        return dict_data

    except Exception as ex:
        traceback.print_exc()
        return {"error": str(ex)}



# Example usage
if __name__ == "__main__":
    from PIL import Image

    image_path = "org.bmp"  # Replace with your image file path
    result = process_image(image_path)
    print("Final Result:", result)
