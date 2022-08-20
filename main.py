# name-Zaid ahmed khan
# roll no-21CH10079
import cv2
import numpy as np
import pickle
import glob
from moviepy.editor import VideoFileClip

def undistort(img):
    camera_dir='camera_cal/cal_pickle.p'
    with open(camera_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def undistort_img():
    # List to Stores all object points & img points from all images
    objpoints = []
    imgpoints = []
    objectPts =np.zeros((6*9,3), np.float32)
    objectPts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
   
    # list containing directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for index, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            objpoints.append(objectPts)
            imgpoints.append(corners)
    img2size = (img.shape[1], img.shape[0])
    # Calibrating camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img2size, None,None)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )



undistort_img()
img = cv2.imread('camera_cal/calibration1.jpg')
dst = undistort(img)

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and removing the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Applying sobel filter in x 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.42,0.65),(0.57,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img2size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img2size
    dst = dst * np.float32(dst_size)
    #calculating the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspect(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img2size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img2size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped
def Histogram(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

right_1, right_2, right_3 = [],[],[]
left_1, left_2, left_3 = [],[],[]
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = pipeline(img)
dst = perspective(dst, dst_size=(1280,720))
def sliding_algorithm(img, NoOfWindows=9, margin=150, pixels = 1):
    global left_1, left_2, left_3,right_1, right_2, right_3 
    left_fit1= np.empty(3)
    right_fit1 = np.empty(3)
    output_img = np.dstack((img, img, img))*255
    histogram = Histogram(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    left_2ase = np.argmax(histogram[:midpoint])
    right_2ase = np.argmax(histogram[midpoint:]) + midpoint
    window_height = int(img.shape[0]/NoOfWindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = left_2ase
    rightx_current = right_2ase
    # Create empty lists to receive left and right lane pixel indices
    left_lane_index = []
    right_lane_index = []
    # Going throught windows
    for window in range(NoOfWindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(output_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(100,255,255), 3) 
        cv2.rectangle(output_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_index.append(good_left_inds)
        right_lane_index.append(good_right_inds)
        # If you found > pixels, recenter next window on their mean position
        if len(good_left_inds) > pixels:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > pixels:        
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    right_lane_index = np.concatenate(right_lane_index)
    left_lane_index = np.concatenate(left_lane_index)
    rightx = nonzerox[right_lane_index]
    righty = nonzeroy[right_lane_index] 
    leftx = nonzerox[left_lane_index]
    lefty = nonzeroy[left_lane_index] 
    # Fit a second order polynomial to left and right part
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    right_1.append(right_fit[0])
    right_2.append(right_fit[1])
    right_3.append(right_fit[2])
    left_1.append(left_fit[0])
    left_2.append(left_fit[1])
    left_3.append(left_fit[2])
    left_fit1[0] = np.mean(left_1[-10:])
    left_fit1[1] = np.mean(left_2[-10:])
    left_fit1[2] = np.mean(left_3[-10:])
    
    right_fit1[0] = np.mean(right_1[-10:])
    right_fit1[1] = np.mean(right_2[-10:])
    right_fit1[2] = np.mean(right_3[-10:])
    # finding x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit1[0]*ploty**2 + left_fit1[1]*ploty + left_fit1[2]
    right_fitx = right_fit1[0]*ploty**2 + right_fit1[1]*ploty + right_fit1[2]
    output_img[nonzeroy[left_lane_index], nonzerox[left_lane_index]] = [255, 0, 100]
    output_img[nonzeroy[right_lane_index], nonzerox[right_lane_index]] = [0, 100, 255]
    return output_img, (left_fitx, right_fitx), (left_fit1, right_fit1), ploty

def drawing_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    cv2.fillPoly(color_img, np.int_(points), (0,255,0))
    inv_perspective = inv_perspect(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def appropriate_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    xm_per_pix = 0.005138
    ym_per_pix = 0.042361
    # Fitting new polynomials to x,y in world space
    left_fit1cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit1cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculating radius of curvature
    left_curvature = ((1 + (2*left_fit1cr[0]*y_eval*ym_per_pix + left_fit1cr[1])**2)**1.5) / np.absolute(2*left_fit1cr[0])
    right_curvature = ((1 + (2*right_fit1cr[0]*y_eval*ym_per_pix + right_fit1cr[1])**2)**1.5) / np.absolute(2*right_fit1cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit1cr[0]*img.shape[0]**2 + left_fit1cr[1]*img.shape[0] + left_fit1cr[2]
    r_fit_x_int = right_fit1cr[0]*img.shape[0]**2 + right_fit1cr[1]*img.shape[0] + right_fit1cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curvature, right_curvature, center)

output_img, curves, lanes, ploty = sliding_algorithm(dst)
print(np.asarray(curves).shape)
curverad=appropriate_curve(img, curves[0],curves[1])
print(curverad)
img2 = drawing_lanes(img, curves[0], curves[1])
def vid_pipeline(img):
    global average, index
    img2 = pipeline(img)
    img2 = perspective(img2)
    output_img, curves, lanes, ploty = sliding_algorithm(img2)
    img = drawing_lanes(img, curves[0], curves[1])
    return img

right_curve, left_curve = [],[]
for i in range(1,4):
    myclip = VideoFileClip('./input_videos/Level-'+str(i)+'.mp4')
    final_video = './output_videos/Level-'+str(i)+'_output.mp4'
    clip = myclip.fl_image(vid_pipeline)
    clip.write_videofile(final_video, audio=False)