import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

def calibrate(test_img):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')
    #print("Total images: ", len(images))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


    img_size = (test_img.shape[1], test_img.shape[0])

    # Do camera calibration given object points and image points
    #print("\nObject points: ", objpoints)
    #print("\nImage points: ", imgpoints)
    #print("\nImage size: ", img_size)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    return (ret, mtx, dist)


# Perspective
# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #src = np.float32([[150+430,440],[1150-440,440],[1150,720],[150,720]])
    src = np.float32([[150+430,460],[1150-440,460],[1150,720],[150,720]])
    #src = np.float32([[573,467],[710,467],[950,620],[357,620]])

    #dst = np.float32([[100, 100],\
    #                     [undist.shape[1]-100, 100],\
    #                    [undist.shape[1]-100, undist.shape[0]-100],\
    #                    [100, undist.shape[0]-100],\
    #                    ])
    
    offset1 = 200 # offset for dst points x value
    offset2 = 0 # offset for dst points bottom y value
    offset3 = 0 # offset for dst points top y value
    img_size = (undist.shape[1], undist.shape[0])
    dst = np.float32([[offset1, offset3],[img_size[0]-offset1, offset3],[img_size[0]-offset1, img_size[1]-offset2], 
                      [offset1, img_size[1]-offset2]])
    
    #dst = np.float32([[200,150], [img_size[1]-]])
    
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    #img_size=(400, 600)
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    return warped, M, Minv



def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
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
    #print(y_len, x_len)
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


def get_max_points(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    #print("Output image size: ", out_img.shape)

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return (leftx_base, rightx_base)


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
        
        #print(win_xright_low, win_xright_high)
        
        # Draw boxes
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
       
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #print(win_y_low, win_y_high, win_xright_low, win_xleft_high, good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            #print("New leftx_current: ", leftx_current)
            
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #print("New rightx_current: ", rightx_current)
        

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return (left_lane_inds, right_lane_inds)


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


def all_steps(test_img, mtx, dist):
    nx = 9
    ny = 6
    masked_img = pipeline(test_img)
    warped, M, Minv = unwarp(masked_img, nx, ny, mtx, dist)
    leftx_base, rightx_base = get_max_points(warped)
    left_lane_inds, right_lane_inds = get_lane_indices(warped, leftx_base, rightx_base)
    ploty, left_fitx, right_fitx = get_fitted_poly(warped, left_lane_inds, right_lane_inds)
    final_img = draw_lines(test_img, warped, ploty, left_fitx, right_fitx, Minv)
    add_radius_to_img(final_img, ploty, left_fitx, right_fitx)
    return final_img


def generate_video(mtx, dist):
    input_video = 'project_video.mp4' 
    clip1 = VideoFileClip(input_video)
    output_file = 'output_project_video.mp4'
    output_clip = clip1.fl_image(lambda img: all_steps(img, mtx, dist))
    output_clip.write_videofile(output_file, audio=False)