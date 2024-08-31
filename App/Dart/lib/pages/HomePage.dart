// ignore_for_file: prefer_const_constructors, use_key_in_widget_constructors, library_private_types_in_public_api

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import './AudioPickerPage.dart';
import './VideoPickerPage.dart';

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Deepfake Detection', style: GoogleFonts.bebasNeue(fontSize: 40)),
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
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => AudioPickerPage()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFF3A5A98), // Button color
                    ),
                    child: Text('Go to Audio Picker',
                    style: TextStyle(color: Colors.white),),
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => VideoPickerPage()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFFFF5C3C), // Button color
                    ),
                    child: Text('Go to Video Picker',
                    style: TextStyle(color: Colors.white),),
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
