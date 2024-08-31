// ignore_for_file: prefer_const_constructors, use_key_in_widget_constructors, library_private_types_in_public_api

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:google_fonts/google_fonts.dart';

class VideoPickerPage extends StatefulWidget {
  @override
  _VideoPickerPageState createState() => _VideoPickerPageState();
}

class _VideoPickerPageState extends State<VideoPickerPage> {
  PlatformFile? pickedFile; // To store the selected file

  // Function to pick an audio, video, or image file
  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp4'],
    );

    if (result != null) {
      setState(() {
        pickedFile = result.files.first;
      });
    } else {
      // User canceled the picker
      setState(() {
        pickedFile = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF181818), // Dark Background
      appBar: AppBar(
        backgroundColor: Colors.transparent, // Dark Background
        // leading: Icon(Icons.menu),
        elevation: 0, // Remove the default shadow
      ),
      body: Center(
        child: Column(
          children: [
            Text('Video Deepfake Detection', style: GoogleFonts.bebasNeue(
              fontSize: 40
            )),
            SizedBox(height: 50),
            Container(
              width: double.infinity,
              margin: EdgeInsets.symmetric(horizontal: 16.0),
              padding: EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Color(0xFF2A2A2A), // Lighter Gray Background
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Column(
                children: [
                  Text('Upload Video for Deepfake Detection',
                      style: TextStyle(
                          color: Color(0xFFFFFFFF),
                          fontSize: 18.0)), // White Text
                  SizedBox(height: 20),
                  pickedFile == null
                      ? Container(
                          width: double.infinity,
                          height: 150,
                          color: Color(0xFFC1D0D8), // Light Blue Placeholder
                          child: Center(
                            child: Text('No file selected',
                                style: TextStyle(
                                    color: Color(0xFF9E9E9E))), // Gray Text
                          ),
                        )
                      : Container(
                          width: double.infinity,
                          height: 150,
                          color: Color(0xFFBCC4BE), // Greenish-Gray Placeholder
                          child: Center(
                            child: Text('Selected file: ${pickedFile!.name}',
                                style: TextStyle(
                                    color: Color(0xFF9E9E9E))), // Gray Text
                          ),
                        ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: pickFile,
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.white,
                      backgroundColor:
                          Color(0xFFFF5C3C), // Text color of the button
                      padding: EdgeInsets.symmetric(
                          horizontal: 20.0, vertical: 10.0),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    child: Text('Add Video'),
                  ),
                  SizedBox(height: 10),
                  Text('Select a video file for detection.',
                      style: TextStyle(color: Color(0xFF9E9E9E))), // Gray Text
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
