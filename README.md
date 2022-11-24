# Motion-Estimation

This project uses Macroblock Matching to perform motion estimation. The source code uses cv2 and numpy library.  
##### To run the source code:
1. Clone the GitHub repository
2. Make sure you have an empty folder called “frames” and another empty folder called “composite” ready in the same directory as the source code. These folders are for the extracted frames and for frames with motion estimation respectively.
3. Make sure you have the input video ready in the same director as source code. The default input video is "monkey.avi". If you are using your own input video see instructions below.
4. You can choose to either run the python file or the ipynb File.
5. For the python file, run the command: `python3 motion_estimation.py`
6. For the ipynb file, press “run all” to run program.


##### Personalisation
1. To change input video path, update the variables `vid_path` in `run()` function. 
2. To change macroblock size, update the variables `size_x` and `size_y` in `run()` function. 
3. To change thresholding range, update the variables `tmin` and `tmax` in  `find_best_match()` function. 
4. To change search area radius, update the variable `sa` in `find_best_match()` function. 
