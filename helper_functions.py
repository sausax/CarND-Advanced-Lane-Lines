import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from moviepy.editor import VideoFileClip


class Line:
    def __init__(self):
        self.left_values = []
        self.right_values = []


    def check_new_points(self, left_val, right_val):
        size = 20 # Keep last 10 values

        if len(self.left_values) < size:
            self.left_values.append(left_val)
            self.right_values.append(right_val)
            return (left_val, right_val)


        last_left = self.left_values[-1]
        last_right = self.right_values[-1]

        if left_val >= 0.99*last_left and left_val <= 1.01*last_left and \
            right_val >= 0.99*last_right and right_val <= 1.01*last_right:

            self.left_values.pop(0)
            self.left_values.append(left_val)

            self.right_values.pop(0)
            self.right_values.append(right_val)


        return np.mean(self.left_values), np.mean(self.right_values)




# Function to display image
def display_image(img, gray=False):
    plt.figure(figsize=(48, 18))
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

# Calibration function.
# Uses chessboard images for calibration
def calibrate():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')

    # Step through the list and search for chessboard corners
    img = None
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    return (ret, mtx, dist)

# Function to undistort images
def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


# Perspective transformation function
def warp(undist, nx, ny, mtx, dist):
    src = np.float32([[580,460],[710,460],[1150,720],[150,720]])

    offset1 = 200 # offset for dst points x value
    offset2 = 0 # offset for dst points bottom y value
    offset3 = 0 # offset for dst points top y value
    img_size = (undist.shape[1], undist.shape[0])
    dst = np.float32([[offset1, offset3],[img_size[0]-offset1, offset3],[img_size[0]-offset1, img_size[1]-offset2], 
                      [offset1, img_size[1]-offset2]])
    
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


# Threshold and color transformation
def transform(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.zeros_like(sxbinary)
    color_binary[(sxbinary > 0) | (s_binary > 0)] = 1 

    y_len = color_binary.shape[0]
    x_len = color_binary.shape[1]
    vertices = np.array([[
        (450, 500),
        (500, 320),
        (x_len, y_len),
        (0, y_len)
        ]])
    mask = np.zeros_like(color_binary)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(color_binary, mask)
    return masked_image

# Get max intensite points using a histogram
def get_max_points(binary_warped, line=None):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    #print("Output image size: ", out_img.shape)

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if line is not None:
        return line.check_new_points(leftx_base, rightx_base)
    else:
        return (leftx_base, rightx_base)


# Function to get lane indices
def get_lane_indices(binary_warped, leftx_base, rightx_base):
    nwindows=9

    window_height = np.int(binary_warped.shape[0]/nwindows)

    #print("Window height: ", window_height)

    nonzero = binary_warped.nonzero()
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0]

    #print("Nonzerox: ", nonzerox)
    #print("Nonzeroy: ", nonzeroy)

    margin = 100
    minpix = 0

    leftx_current = leftx_base
    rightx_current = rightx_base


    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return (left_lane_inds, right_lane_inds)

# Function to plot curvature on image
def plot_curvature(binary_warped, left_lane_inds, right_lane_inds, ploty, left_fitx, right_fitx):
    binary_warped = binary_warped.copy()
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.figure(figsize=(48, 18))
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

# Function to get points from fitted polynomial
def get_fitted_poly(binary_warped, left_lane_inds, right_lane_inds):
    nonzero = binary_warped.nonzero()
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0]
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 

    return ploty, left_fitx, right_fitx

# Function to draw polynomial lines on images
def draw_lines(image, warped, ploty, left_fitx, right_fitx, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    #result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

def calculate_radius(ploty, leftx, rightx):
    y_eval = np.max(ploty)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return (left_curverad, right_curverad)

def calculate_distance_from_center(left, right, image_width):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    lane_center = (left+right)/2
    image_center = image_width/2
    return (lane_center - image_center) * xm_per_pix

# Function to add radius/distance from center text to images
def add_radius_to_img(img, ploty, leftx, rightx):
    left_curverad, right_curverad = calculate_radius(ploty, leftx, rightx)
    text = 'Curve radius: ' + '{:04.2f}'.format(left_curverad) + 'm'
    cv2.putText(img, text, (40,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)

    dist_from_center = calculate_distance_from_center(leftx[-1], rightx[-1], img.shape[1])
    if dist_from_center > 0:
        text = '{:04.3f}'.format(dist_from_center) + 'm' + ' left of center'
    else:
        dist_from_center = abs(dist_from_center)
        text = '{:04.3f}'.format(dist_from_center) + 'm' + ' right of center'
    cv2.putText(img, text, (40,110), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)

# Function with all pipeline functions
def all_steps(test_img, mtx, dist, line=None):
    nx = 9
    ny = 6
    undist = undistort(test_img, mtx, dist) 
    masked_img = transform(undist)
    warped, M, Minv = warp(masked_img, nx, ny, mtx, dist)
    leftx_base, rightx_base = get_max_points(warped, line)
    left_lane_inds, right_lane_inds = get_lane_indices(warped, leftx_base, rightx_base)
    ploty, left_fitx, right_fitx = get_fitted_poly(warped, left_lane_inds, right_lane_inds)
    final_img = draw_lines(test_img, warped, ploty, left_fitx, right_fitx, Minv)
    add_radius_to_img(final_img, ploty, left_fitx, right_fitx)
    return final_img


# Function to generate final video
def generate_video(mtx, dist):
    line = Line()
    input_video = 'project_video.mp4' 
    clip1 = VideoFileClip(input_video)
    output_file = 'output_project_video.mp4'
    output_clip = clip1.fl_image(lambda img: all_steps(img, mtx, dist, line))
    output_clip.write_videofile(output_file, audio=False)