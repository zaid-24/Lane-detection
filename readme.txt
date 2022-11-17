main is the only python file in this project
input videos folder contain folder which were given to us as task 
output videos folder contain output videos
Undistorting image:
As different cameras uses different type of lens So there arises a need of undistorting the image , So first I have undistorted the image using chessboard images and some open CV inbuilt function .

Perspective Transform:
Here I have apply perspective transform to get bird’s eye view ,bird’s eye view is helpful in calculating the curvature of the road . The other benefit of top-down view is it fixes the issue where lane lines seem to merge whereas lane lines are infinitely parallel lines as long as the road runs.
 

Sobel Filtering:
In order to tackle things like different colour of road we use sobel filter , we will be apply sobel filter one two colour channels of HSL colorspace to detect change in saturation and lightness . 
Histogram Peak Detection:
To find the starting point of lane line we have make use of histogram peak , the point where there is a peak of histogram will be the starting point of lane line.
Sliding Window Search:
I have used this to detect the difference between the left and right lane .It Start from the initial position, the first window(which is kind of box) measures how many pixels are located inside the window, If the amount of pixels reaches a certain threshold, it shifts the next window to the average lateral position of the detected pixels. If not enough pixels are detected, the next window starts in the same lateral position. This continues until the windows reach the other edge of the image. 
 	
Curve Fitting:
Then using polynomial regression we drawn suitable curve and then clipped the video and applied the whole algorithm to it .
