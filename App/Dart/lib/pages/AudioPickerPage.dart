import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

class AudioPickerPage extends StatefulWidget {
  @override
  _AudioPickerPageState createState() => _AudioPickerPageState();
}

class _AudioPickerPageState extends State<AudioPickerPage> {
  PlatformFile? pickedFile; // To store the selected file
  String result = ''; // Initialize result as an empty string

  // Function to pick an audio file
  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp3'],
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

  // Function to upload the file to the Flask server
  Future<void> uploadFile() async {
    if (pickedFile == null) {
      setState(() {
        result = "No file selected. Please choose a file first.";
      });
      return;
    }

    try {
      var request = http.MultipartRequest('POST', Uri.parse('http://127.0.0.1:5000/predict'));

      if (pickedFile!.bytes != null) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            pickedFile!.bytes!,
            filename: pickedFile!.name,
          ),
        );
      } else {
        setState(() {
          result = "Error: File bytes are null";
        });
        return;
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var decodedResponse = json.decode(responseData);

        // Process the response to format it as desired
        String formattedResult = "Upload successful:\n";
        for (var item in decodedResponse) {
          formattedResult += "Label: ${item['label']}, Score: ${item['score']}\n";
        }

        setState(() {
          result = formattedResult;
        });
      } else {
        setState(() {
          result = "Upload failed with status: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        result = "An error occurred: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF181818), // Dark Background
      appBar: AppBar(
        backgroundColor: Colors.transparent, // Dark Background
        elevation: 0, // Remove the default shadow
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center, // Center the content
          children: [
            Text('Audio Deepfake Detection', style: GoogleFonts.bebasNeue(fontSize: 40)),
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
                  Text('Upload Audio for Deepfake Detection',
                      style: TextStyle(color: Color(0xFFFFFFFF), fontSize: 18.0)), // White Text
                  SizedBox(height: 20),
                  pickedFile == null
                      ? Container(
                          width: double.infinity,
                          height: 150,
                          color: Color(0xFFC1D0D8), // Light Blue Placeholder
                          child: Center(
                            child: Text('No file selected',
                                style: TextStyle(color: Color(0xFF9E9E9E))), // Gray Text
                          ),
                        )
                      : Container(
                          width: double.infinity,
                          height: 150,
                          color: Color(0xFFBCC4BE), // Greenish-Gray Placeholder
                          child: Center(
                            child: Text('Selected file: ${pickedFile!.name}',
                                style: TextStyle(color: Color(0xFF9E9E9E))), // Gray Text
                          ),
                        ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: pickFile,
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.white,
                      backgroundColor: Color(0xFFFF5C3C), // Text color of the button
                      padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 10.0),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    child: Text('Add Audio'),
                  ),
                  SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: uploadFile,
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.white,
                      backgroundColor: Color(0xFFFF5C3C), // Text color of the button
                      padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 10.0),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                    ),
                    child: Text('Submit for Detection'),
                  ),
                  SizedBox(height: 20),
                  Container(
                    width: double.infinity,
                    padding: EdgeInsets.all(16.0),
                    decoration: BoxDecoration(
                      color: Color(0xFF2A2A2A), // Lighter Gray Background
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                    child: Text(
                      result.isEmpty ? 'Select an audio file for detection.' : result,
                      style: TextStyle(color: Color(0xFFFFFFFF), fontSize: 18.0), // White Text
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
