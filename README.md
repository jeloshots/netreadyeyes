## netready-eyes Netrunner card recognizer

Explanation:
Webcam Access:

Uses OpenCV to access the webcam and display the feed in a Tkinter GUI using a Label.
Image Recognition:

Once you select a folder with PNG images, the script uses OpenCV’s matchTemplate function to find the image in the webcam feed and draw a rectangle around it.
Export to OBS:

The script uses obs-websocket-py to communicate with OBS. When an image match is found in the webcam feed, it will send a request to OBS to show the image as an image source.
You need to configure OBS WebSocket (with the password set in the script) and have the appropriate scene and image source already configured in OBS.
GUI:

Provides buttons to start/stop the webcam, select the image folder, and configure OBS WebSocket for integration.
OBS WebSocket Setup:
Install OBS WebSocket Plugin: OBS WebSocket GitHub
Enable WebSocket in OBS: Open OBS → Tools → WebSockets Server Settings → Enable WebSockets and set a password.
Running the Script:
You can run this script on your Windows machine. It will open a GUI to let you start the webcam, load images, and recognize them.