import math
import cv2
import os
import numpy as np

# Helper function to draw arrows.
def arrowdraw(img, x1, y1, x2, y2):
    radians = math.atan2(x1-x2, y2-y1)
    x11 = 0
    y11 = 0
    x12 = -10
    y12 = -10

    u11 = 0
    v11 = 0
    u12 = 10
    v12 = -10
    
    x11_ = x11*math.cos(radians) - y11*math.sin(radians) + x2
    y11_ = x11*math.sin(radians) + y11*math.cos(radians) + y2

    x12_ = x12 * math.cos(radians) - y12 * math.sin(radians) + x2
    y12_ = x12 * math.sin(radians) + y12 * math.cos(radians) + y2
    
    u11_ = u11 * math.cos(radians) - v11 * math.sin(radians) + x2
    v11_ = u11 * math.sin(radians) + v11 * math.cos(radians) + y2

    u12_ = u12 * math.cos(radians) - v12 * math.sin(radians) + x2
    v12_ = u12 * math.sin(radians) + v12 * math.cos(radians) + y2

    img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    img = cv2.line(img, (int(x11_), int(y11_)), (int(x12_), int(y12_)), 
    (255, 0, 0), 2)
    img = cv2.line(img, (int(u11_), int(v11_)), (int(u12_), int(v12_)), 
    (255, 0, 0), 2)
    
    return img

# USYD CODE CITATION ACKNOWLEDGEMENT
# I declare that the following lines of code have been copied from the
# tutorial lab checkpoints (LCP) of Week 4 with only minor changes and it is not my own work. 

# Tutorial lab checkpoints for week 4 from COMP3419 course
# https://canvas.sydney.edu.au/courses/35639/files/19261515?wrap=1

# Opens input video and extract its frames.
def extract_frames(path):
    expected_path = os.path.join(os.getcwd(), "frames")
    if not os.path.isdir(expected_path):
        os.mkdir("frames")
    else:
        print("\"frames\" directory already exists")

    framenum = 0
    video = cv2.VideoCapture(path)
    fheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fwidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    while(1):
        ret, frame = video.read()
        if not ret:
            print("Video reached end.")
            break
        print("Extracting {}".format(framenum))
        cv2.imwrite("frames/{}.tif".format(framenum), frame)
        framenum +=1
    video.release()

    return (fheight, fwidth)

# end of copied code

# Find best match for macroblock matching.
def find_best_match(current, nextfr, x, y, size_x, size_y, width, height):

    # Thresholding range.
    tmin = 130
    tmax = 150

    sa = 1 # Search area radius.
    ssd_list = []

    # X and Y location of starting point when iterating through pixels in a macroblock.
    nx = x - (size_x*sa) # starts at leftmost pixel x location of of a macroblock.
    ny = y - (size_y*sa) # starts topmost pixel y location of of a macroblock.

    ystart = y - (size_y*sa) # topmost y location of pixel in a macroblock.
    cblock = current[y: y+size_y, x:x+size_x, :] # source block.

    padding = 1+(sa*2) # padding for search area of neighbouring macroblocks.
    midx = size_x//2 # x location for central pixel of a macroblock.
    midy = size_y//2 # y location for central pixel of a macroblock.

    # Iterate through the neighbouring macroblocks.
    for row in range (padding):
        ny = ystart
        for col in range(padding):
            current_ssd = [] # sqrt(SSD) of current neighbouring block of next(reference) frame.

            # Ignores neighbouring blocks that are outside the scope of the frame.
            if ny+size_y <= 0 or nx+size_x <= 0 or ny+size_y > height or nx+size_x > width:
                continue

            nblock = nextfr[ny:ny+size_y, nx:nx+size_x, :] # get a neighbouring block on next(reference)frame.

            # Calculating sqrt(SSD) of source block and a target block within the neighbouring blocks.
            current_ssd.append(np.sqrt(np.sum((np.power((cblock-nblock),2)))))
            current_ssd.append(nx+midx) # add the x location of central pixel of the target block.
            current_ssd.append(ny+midy) # add the y location of central pixel of the target block.

            ssd_list.append(current_ssd) # append sqrt(SSD) of current target block to list.

            ny+=size_y


        nx+=size_x

    
    if len(ssd_list) != 0:
        min_ssd = min(ssd_list) # find best match by calculating minimum sqrt(SSD).

        # if best match does not have same location as the source block, and its sqrt(SSD) is within thresholding range,
        # then it is the best match that we will draw motion vector to.
        if (min_ssd[1] != x+midx and min_ssd[2] != y+midy) and min_ssd[0] < tmax and min_ssd[0] > tmin:
            return min_ssd

        else: 
            return None
    else:
        return None

# Produce motion estimation for every frame using macroblock matching by calculating sqrt(SSD).
def track_per_frame(counter, framenum, kx, ky, width, height):

    # Iterate through frames of input video.
    while counter <=framenum:
        current = cv2.imread("frames/{}.tif".format(counter)) # current frame.
        nextfr = None

        if counter+1 <=framenum:
            nextfr = cv2.imread("frames/{}.tif".format(counter+1)) # next frame (reference frame).
        else:
            break

        xlen= current.shape[1] # 720 pixels (width)
        ylen = current.shape[0] # 576 pixels (height)

        # Iterate through every macroblock of current frame.
        for x in range(0, xlen, kx):
            ssd = None
            for y in range(0, ylen, ky):
                
                # calculate SSD and determine target block best match.
                match = find_best_match(current,nextfr, x, y, kx, ky, width, height)
                if match != None:
                    arrowdraw(current, x, y, ssd[1], ssd[2]) # draw arrow from source block to target block best match if there is motion.

        cv2.imwrite('composite/composite%d.tif' % counter, current) # create frame with motion estimation arrow.
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
        counter += 1

# USYD CODE CITATION ACKNOWLEDGEMENT
# I declare that the following lines of code have been copied from the
# tutorial lab checkpoints (LCP) of Week 4 with only minor changes and it is not my own work. 

# Tutorial lab checkpoints for week 4 from COMP3419 course
# https://canvas.sydney.edu.au/courses/35639/files/19261515?wrap=1

# Convert frames to video.
def convert_to_video(video_path, width, height, framenum):
    count = 0
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(width), int(height)))
    while count <= framenum:
        img = cv2.imread('composite/composite%d.tif' % count)
        if img is None:
            print('No more frames to be loaded')
            break

        out.write(img)
        count += 1
        print('Saving video: %d%%' % int(100*count/framenum))
        
    out.release()
    cv2.destroyAllWindows()
    
# end of copied code

def run():
    vid_path = 'monkey.avi' # path to input video.

    start_frame = 0
    last_frame = 751
    size_x = 15
    size_y = 9

    height, width = extract_frames(vid_path) # Open input video and extract frames.
    track_per_frame(start_frame, last_frame, size_x, size_y, width, height) # Motion extimation for every frame.
    new_path = "./SID490055892_Ass1a.mov" # Output video path.
    convert_to_video(new_path, width, height, 600) # Convert frames with motion extimation into output video.

if __name__ == "__main__":
    run() # runs program.