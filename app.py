import cv2
from flask import Flask
from flask import Response
 
app = Flask(__name__)


# @app.route("/video")
# def mem():
#     return {"mem": ["ishant", "ishanr11"]}

@app.route('/video')
def video_feed():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a generator function t
    # hat yields the video frames
    def gen():
        while True:
            # Capture a frame from the camera
            success, frame = cap.read()

            # Encode the frame as a JPEG image
            ret, jpeg = cv2.imencode('.jpg', frame)

            # Yield the encoded frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # Return a Response object that uses the generator function as its source
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/react')
def get_video_link():
    link = "http://localhost:5000/video"
    return link

app.run(debug=True)