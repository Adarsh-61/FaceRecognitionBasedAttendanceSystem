# Face Recognition Based Attendance System
This project is a **Face Recognition Based Attendance System** that simplifies the process of tracking and recording attendance using computer vision and machine learning techniques. It is built in Python and leverages libraries such as OpenCV, NumPy, and Pandas for face detection, recognition, and data management.

## Features
- **Password Protection:** Secure login mechanism to restrict unauthorized access.
- **New Registrations:** Add new user profiles by capturing face images and associating them with unique IDs and names.
- **Image Capture:** Capture multiple images per user for improved accuracy during recognition.
- **Model Training:** Train a face recognition model using the Local Binary Patterns Histograms (LBPH) algorithm.
- **Attendance Tracking:** Recognize faces in real-time via webcam and mark attendance in a CSV file.
- **Tkinter GUI:** User-friendly graphical interface for accessing all system features.

## How It Works
1. **Login:** The system requires a password for access. You can customize the password in the script.
2. **Register New Users:** Add new users by providing a unique ID and name. The system captures multiple face images for each user.
3. **Train the Model:** The captured images are used to train the LBPH face recognizer, enabling it to identify registered users.
4. **Track Attendance:** The system uses the webcam to detect and recognize faces in real-time. Recognized faces are marked as present in the attendance log.
5. **View Attendance:** Attendance is recorded in a CSV file named `Attendance.csv`, including the user's ID, name, and timestamp.

## Customization
- **Password:** Update the `password` variable in the script to set a custom password.
- **Face Recognition Algorithm:** The project uses LBPH but can be extended to support other algorithms like Eigenfaces or Fisherfaces.

## Limitations
- Recognition accuracy depends on lighting conditions and image quality.
- Requires consistent facial positioning for optimal performance.

## Future Enhancements
- Integration with a database for centralized data storage.
- Support for cloud-based attendance logs.
- Improved accuracy with deep learning models like FaceNet or Dlib.
- Adding multi-user roles with separate permissions.

## Acknowledgments
- OpenCV for providing robust computer vision tools.
- Python community for the rich ecosystem of libraries.
- Inspiration from various open-source face recognition projects.

## Contribution
Feel free to fork the repository, submit issues, or suggest improvements. Contributions are always welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Developed by **A&J** as part of the Multimodal System.
