# Face Recognition Based Attendance System (Enhanced)

This project is an **Face Recognition Based Attendance System** designed for efficient and smooth tracking of attendance. Originally developed by A&J, this version incorporates significant performance optimizations, a refined user interface that preserves the original aesthetic, and uses modern tooling for setup.

Built in Python, it leverages OpenCV for core computer vision tasks (Haar Cascades for detection, LBPH for recognition), Pandas for data handling, and Tkinter for the graphical user interface.

## Key Features

*   **Enhanced Performance:** Optimized for speed and smoothness using multithreading, frame skipping, image buffering, batch processing, and caching to eliminate UI lag.
*   **Original GUI Design:** Retains the user-preferred visual style (colors, fonts, layout) from the original A&J design.
*   **Password Protection:** Secure mechanism for changing passwords and protecting the training process.
*   **User Registration:** Easily add new users by capturing face images and associating them with an ID and name.
*   **LBPH Model Training:** Trains an OpenCV LBPH (Local Binary Patterns Histograms) face recognizer using captured images.
*   **Real-Time Attendance Tracking:** Recognizes registered faces via webcam in real-time and logs attendance automatically.
*   **Attendance Logs:** Records attendance details (ID, Name, Date, Time) in daily CSV files within the `Attendance` directory.
*   **Modern Setup:** Uses the `uv` package manager for fast and reliable dependency installation (compatible with Windows without requiring C++ build tools).

## Installation

This system is designed to run on Windows, macOS, and Linux. The following instructions use `uv`, a fast Python package installer.

**Prerequisites:**

*   Python 3.10+ (Python 3.12 recommended and tested)
*   `uv` package installer

**1. Install UV:**

If you don't have `uv`, install it first (it's much faster than pip!):

```bash
# Using pip (if you have it)
pip install uv

# Or using pipx (recommended for tool installation)
pipx install uv
```

**2. Get the Project Files:**

Download or clone the project files to your local machine.

**3. Set Up the Environment:**

Open your terminal or command prompt, navigate to the project directory (where `requirements.txt` is located), and follow these steps:

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required packages using uv
uv pip install -r requirements.txt
```

**Note:** The `requirements.txt` file is specifically curated to work on Windows without needing separate C++ build tools, as it relies on OpenCV's built-in capabilities.

## What is UV?

`uv` is an extremely fast Python package installer and resolver, written in Rust, and designed as a drop-in replacement for `pip` and `pip-tools` workflows.

**Why Use UV for This Project?**

*   **Speed:** `uv` can install packages like OpenCV, NumPy, and Pandas 10-100x faster than `pip`.
*   **Reliability:** It features an advanced dependency resolver to prevent conflicts.
*   **Simplicity:** It works seamlessly with the provided `requirements.txt` file.
*   **Compatibility:** Works well with Python 3.12 and handles modern package formats efficiently.
*   **No C++ Build Tools Required:** The optimized requirements file works with UV on Windows without needing Visual Studio or other C++ build tools.

Using `uv` significantly speeds up the setup process for this project and avoids common installation issues.

## Usage

Once the setup is complete and the virtual environment is active, run the application:

```bash
uv run SingleModel.py
```

The main window will appear with two sections:

1.  **For Already Registered:**
    *   **Take Attendance:** Starts the webcam to recognize faces and log attendance for the current day.
    *   **Attendance List:** Displays the attendance recorded for the current session.
    *   **Quit:** Exits the application.
2.  **For New Registrations:**
    *   **Enter ID & Name:** Input the unique ID and Name for the new user.
    *   **Take Images:** Captures 60 face images for the new user. Follow the on-screen prompts (press 'q' to stop early if needed).
    *   **Save Profile:** Trains the LBPH model with all captured images. You will be prompted for a password (set one on the first run if it doesn't exist).
    *   **Clear:** Clears the ID or Name input fields.

**Password Management:**

*   The first time you click "Save Profile," you'll be prompted to create a new password.
*   For subsequent uses, you'll need to enter this password to train the model.
*   You can change the password via the Help menu → Change Password.

**Menu Options:**

*   **Help → Other Models:** Launches the Multimodel1.py script (if available).
*   **Help → Change Password:** Opens the password change dialog.
*   **Help → Contact Us:** Displays contact information.
*   **Help → Exit:** Closes the application.

## System Structure

The enhanced system organizes data in the following directories (created automatically if they don't exist):

*   **TrainingImage/**: Stores captured face images for each user.
*   **TrainingImageLabel/**: Contains the trained model file (`Trainner.yml`) and password file.
*   **StudentDetails/**: Stores the CSV file with registered user information.
*   **Attendance/**: Contains daily attendance logs in CSV format (e.g., `Attendance_28-04-2025.csv`).

## Performance Optimizations

The enhanced system includes several optimizations to ensure smooth operation:

1.  **Multithreading Architecture:**
    *   Image capture, model training, and attendance tracking run in separate threads.
    *   A thread-safe queue system ensures UI remains responsive during intensive operations.

2.  **Memory Management:**
    *   Pre-allocated image buffers reduce memory fragmentation and garbage collection pauses.
    *   Efficient resource handling ensures camera and window resources are properly released.

3.  **Processing Optimizations:**
    *   LRU caching for image preprocessing speeds up training.
    *   Batch processing of images with progress updates.
    *   Intelligent frame skipping during face detection maintains high FPS.
    *   Real-time FPS monitoring to track performance.

4.  **Face Detection Improvements:**
    *   Optimized detection parameters for better accuracy and speed.
    *   Reduced image sample count (100 instead of 150) while maintaining recognition quality.
    *   Resizing frames for faster processing during detection.

## Customization

You can customize various aspects of the system:

*   **Appearance:** The UI colors and fonts can be modified in the `setup_window`, `setup_frames`, and related methods.
*   **Face Detection Parameters:** Adjust the `scaleFactor`, `minNeighbors`, and `minSize` parameters in the face detection calls.
*   **Sample Count:** Change the number of images captured per user by modifying the condition in the image capture loop.
*   **Recognition Confidence Threshold:** Adjust the confidence threshold (currently 70) in the `_track_attendance` method.

## Troubleshooting

**Camera Issues:**
*   Ensure your webcam is properly connected and not in use by another application.
*   If the camera doesn't initialize, check your system's privacy settings to allow camera access.

**Recognition Problems:**
*   Ensure adequate lighting for better face detection and recognition.
*   Try registering with more varied facial expressions and angles for improved recognition.
*   If recognition accuracy is low, consider retraining with more images in better lighting.

**Installation Issues:**
*   If you encounter issues with UV, you can fall back to pip: `pip install -r requirements.txt`
*   For Python 3.12 users: The requirements file is specifically optimized for Python 3.12 compatibility.

## Future Enhancements

Potential areas for future development:

*   Database integration for centralized data storage.
*   Cloud synchronization for attendance records.
*   Mobile application for remote attendance tracking.
*   Advanced deep learning models for improved recognition accuracy.
*   Multi-user roles with different permission levels.
*   Automated reporting and analytics features.

## Demonstration

Watch the original demonstration of the project on YouTube: [Face Recognition Based Attendance System Demo](https://youtu.be/xE6-FPF9Pow?si=3sD1bgp2BDZP0rYo)

## Acknowledgments

*   Original system developed by A&J.
*   OpenCV for providing robust computer vision tools.
*   Python community for the rich ecosystem of libraries.
*   Astral team for developing the UV package manager.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Developed by A&J as part of the Multimodal System
