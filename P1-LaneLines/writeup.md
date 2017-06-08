# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline description

My pipeline consisted of 9 steps: 
    1. Read image (by caller)
    2. convert the images to grayscale 
    3. gaussian smoothing - with selected kernel size
    4. Canny edge - with selected parameters
    5. Hough transform - with selected parameters
    6. detect lines
    7. extrapolate lines
    8. plot lines on scene
    9. save to test_videos_output

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to separate the lines based on positive or negative slope using the polyfit function. I then did a line fit to each group of points and plotted the 2 lines on the image in the region of interest.  

When separating by slope, I set a minimum threshold of +/- 0.4 to keep horizontal lines out of the fit to reduce noise.


### 2. Potential shortcomings with current pipeline

One shortcoming is that the region of interest mask doesn't adapt different image sizes or to varying angles between the car and road, due to hills and sharp curves.
 
Another shortcoming would be not adapting to changes in lighting, like shadows, night time, or rain/fog.


### 3. Possible improvements to the pipeline

A possible improvement would be to use an HSV transform with thresholds to detect the expected line colors better.

Another potential improvement could be to fit the lines with a 2nd degree polynomial to match curved roadway better. Also, not draw the lines above the point where they cross would look a lot more intelligent
