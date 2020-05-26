import numpy as np
from numpy.fft import fft
import cv2 as cv
from random import randint
from skimage.transform import rescale, rotate
from skimage import measure

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='path to input video')
parser.add_argument('--output', required=True, help='path to result video')
parser.add_argument('--degree', default=0)

args = parser.parse_args()

def binarize(image, treshold):
    if image.ndim == 2:
        mask_white = image[:,:] > treshold
    elif image.ndim == 3:
        mask_white = image[:,:,0] > treshold
    image_binarized = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
    image_binarized[mask_white] = 255
    if image.ndim == 2:
        return image_binarized        
    elif image.ndim == 3:
        return image_binarized.reshape(image_binarized.shape[0], image_binarized.shape[1], -1)
    
def normalize_intensity(image, linear=True):
    """
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    image = image.astype('float')    
    for i in range(image.shape[2]):
        #min_intensity = np.amin(image[:,:,:])
        #max_intensity = np.amax(image[:,:,:])
        min_intensity = np.percentile(image[:,:,i],0)
        max_intensity = np.percentile(image[:,:,i],100)
        diff = max_intensity - min_intensity
        avg = np.mean(image[:,:,i])
        if min_intensity != max_intensity:
            if linear: 
                factor = 255.0 / diff
                image[:,:,i] = (image[:,:,i] - min_intensity) * factor
            else: 
                image[:,:,i] = 255.0 / (1.0 + np.exp((avg - image[:,:,i]) / diff))
    image = np.clip(image, 0.0, 255.0)
                
    return image.astype('uint8')

def sliding_window(image, stride, window_size):
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            yield (j, i, image[i:i+window_size[0], j:j+window_size[1]])
            
def detection(image, stride, window_side):
    window_no = 1
    windows_to_be_checked = set()
    for (j, i, window) in sliding_window(image, stride, 
                                         window_size=(window_side, window_side)):
        # Check whether window is fully contained within image
        if window.shape[0] != window_side or window.shape[1] != window_side:
            continue
        
        # Check whether window contains too few binary values of one kind
        if np.count_nonzero(window) < 50 or np.count_nonzero(255-window) < 50:
            continue
        
        # Check whether the window edges contain white pixels
        # (in case they do, it's better not to consider this window since it might be too difficult to 
        #  correctly classify the object it contains)
        edges_list = [window[0,:-1], window[:-1,-1], window[-1,::-1], window[-2:0:-1,0]]
        edges_list = np.concatenate(edges_list)
        if 255 in edges_list:
            continue

        window_no += 1
        
        windows_to_be_checked.add((j, i, window_side))
    # print(len(windows_to_be_checked))
    return windows_to_be_checked

def windows_overlap(i_1_min, j_1_min, i_1_max, j_1_max, i_2_min, j_2_min, i_2_max, j_2_max):
    box_1_in_box_2 = (((i_2_min <= i_1_min <= i_2_max) or (i_2_min <= i_1_max <= i_2_max)) 
                      and ((j_2_min <= j_1_min <= j_2_max) or (j_2_min <= j_1_max <= j_2_max)))
    box_2_in_box_1 = (((i_1_min <= i_2_min <= i_1_max) or (i_1_min <= i_2_max <= i_1_max)) 
                      and ((j_1_min <= j_2_min <= j_1_max) or (j_1_min <= j_2_max <= j_1_max)))
    return box_1_in_box_2 or box_2_in_box_1

def find_window_with_most_white_pixels(image, i_1_min, j_1_min, i_1_max, j_1_max, i_2_min, j_2_min, i_2_max, j_2_max):
    window_1 = image[i_1_min:i_1_max, j_1_min:j_1_max]
    window_2 = image[i_2_min:i_2_max, j_2_min:j_2_max]
    if np.count_nonzero(window_1) >= np.count_nonzero(window_2):
        return 0
    else:
        return 1   

def remove_overlapping_windows_with_same_window_side(image, windows):
    windows_list = list(windows.copy())
    window_side = windows_list[0][2]
    to_be_removed = []
    for k in range(len(windows_list)):
        j_1 = windows_list[k][0]
        i_1 = windows_list[k][1]
        if (j_1, i_1, window_side) not in to_be_removed:
            for l in range(k+1, len(windows_list)):
                j_2 = windows_list[l][0]
                i_2 = windows_list[l][1]
                if windows_overlap(i_1, j_1, i_1+window_side, j_1+window_side, 
                                   i_2, j_2, i_2+window_side, j_2+window_side):
                    window_to_be_kept = find_window_with_most_white_pixels(image, 
                                                i_1, j_1, i_1+window_side, j_1+window_side, 
                                                i_2, j_2, i_2+window_side, j_2+window_side)
                    if window_to_be_kept == 0:
                        to_be_removed.append((j_2, i_2, window_side))
                    else: 
                        to_be_removed.append((j_1, i_1, window_side))
    to_be_removed = set(to_be_removed)
    return windows.difference(to_be_removed)

def pad(image, side):
    diff = int((side - image.shape[0]) / 2)
    return cv.copyMakeBorder(image, diff, diff, diff, diff, cv.BORDER_CONSTANT)

def random_translation(image, side): 
    result = np.zeros((side, side))
    
    trans_range = side-image.shape[0]
    trans_x = randint(0, trans_range-1)
    trans_y = randint(0, trans_range-1)
    
    side = image.shape[0]
    result[trans_y:trans_y+side, trans_x:trans_x+side] = image
    
    return result

def random_scaling(image):
    scale = randint(50, 200)/100
    return rescale(image, scale)

def random_rotation(image): 
    degrees = randint(0, 360-1)
    return rotate(image, degrees)

def random_transformation(image, side, rotate=False, rescale=False, translate=False):
    # side is the output image side length, 
    # but please try side=64 because random_scaling might otherwise act difficult
    # (the latter is not of importance to us since we use 64x64 input images anyway)
    assert(image.shape == (28, 28))
    
    if rotate:
        result = random_rotation(image)
    else:
        result = image
        
    if rescale:
        result = random_scaling(result)
        
    if translate:
        return random_translation(result, side)
    else:
        return pad(result, side)
    
def remove_nines(x, y):
    nines_indices = np.argwhere(y==9)
    x_result = np.delete(x, nines_indices, axis=0)
    y_result = y[y<9]
    return (x_result, y_result)

def classify_operator(image):
    contours = measure.find_contours(image, 0)
    num_contours = len(contours)
    if num_contours == 3:
        return '/'
    elif num_contours == 2:
        return '='
    else: # Only remaining option: num_contours == 1
        contour = max(contours, key=len)
        dft_coeff = fft(contour[:,1]+contour[:,0]*1j)
        if (np.abs(dft_coeff[3]) > np.abs(dft_coeff[5])) and (np.abs(dft_coeff[3]) > np.abs(dft_coeff[7])):
            return '-'
        elif np.abs(dft_coeff[5]) > np.abs(dft_coeff[7]):
            return '+'
        else:
            return '*'

### CNN architecture

class NClassifierNet5(nn.Module):
    def __init__(self):

        super(NClassifierNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3) # 28 -> 26 (13) / 14 -> 12
        self.bn1 = nn.BatchNorm2d(num_features = 32) 
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3) # 26 -> 24 (13 -> 11 (5))/ 12 -> 10
        self.bn2 = nn.BatchNorm2d(num_features = 64)
        self.conv3 = nn.Conv2d(64,128, kernel_size = 3) # 24 -> 22 / 10 -> 8
        self.bn3 = nn.BatchNorm2d(num_features = 128)
        self.conv4 = nn.Conv2d(128,256, kernel_size = 3)
        self.bn4 = nn.BatchNorm1d(num_features = 256*2*2)
        self.fc2 = nn.Linear(256*2*2,9)

    def forward(self, xA):

        A = F.relu(F.max_pool2d(self.conv1(xA),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv2(self.bn1(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv3(self.bn2(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.relu(F.max_pool2d(self.conv4(self.bn3(A)),kernel_size = 2, stride = 2))
        #print(A.shape)
        A = F.leaky_relu(self.fc2(self.bn4(A.view(-1, 256*2*2))))
        #print(A.shape)

        return A

def classify_number(padded_box):
    device = 'cpu'
    PATH = './collab_noNines_cnn5_TTT_bs1000_ne50.pth'

    eta = 1e-3
    model, criterion = NClassifierNet5(), nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=eta, betas=[0.99,0.999])

    model.to(device)
    criterion.to(device)

    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Predict
    model.eval()
    img = pad(padded_box/255.,64)
    image_tensor = torch.Tensor(img).view(-1,1,64,64).float().to(device)
    guess = model(image_tensor)
    prob,output = torch.max(guess,1)
    num = str(int(output))
    return num


if __name__=="__main__":
    input_path = args.input
    output_path = args.output
    test_degree = float(args.degree)
    
    ### 0. EXTRACTING THE FRAMES
    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS) # store fps for the video that we have to make
    frames = [] # creating a list for all consecutive frames

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        if (test_degree != 0):
            frame = rotate(frame,test_degree).astype('float32')*255.
            frame = frame.astype('uint8')

        frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv.destroyAllWindows()
    
    first_frame = frames[0]
    last_frame = frames[-1]
    (im_h, im_w) = first_frame.shape[:2]
    
    first_frame_gray = 255-cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    last_frame_gray = 255-cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)
    
    first_frame_binarized = binarize(first_frame_gray, 140)
    last_frame_binarized = binarize(last_frame_gray, 140)
    
    ## 0.0 TRACING THE ARROW
    first_frame_normalized = normalize_intensity(first_frame, linear=True)
    
    normalized_frames = [normalize_intensity(frame, linear=True) for frame in frames]

    masks_arrow = [np.logical_and.reduce((frame[:,:,0]>120,
                                          frame[:,:,1]<90,
                                          frame[:,:,2]<180)) for frame in normalized_frames]

    arrows_coords = [np.array(np.where(mask_arrow == True)) for mask_arrow in masks_arrow]

    arrow_locations = [np.round(np.mean(arrow_coords, axis=1)).astype(int) for arrow_coords in arrows_coords]

    ### 1. CONSTRUCTING BOUNDING BOXES
    for k in range(80, 40, -8):
        if len(detection(first_frame_binarized, stride=2, window_side=k)) == 0:
            min_window_side = k+8
            break
        else:
            min_window_side = 32
        
    windows_final = detection(first_frame_binarized, stride=2, window_side=min_window_side)
    windows_final = remove_overlapping_windows_with_same_window_side(first_frame_binarized, windows_final)

    for k in range(min_window_side+8, 80, 8):
        windows_temp = detection(first_frame_binarized, stride=2, window_side=k)
        to_be_removed = set()
        for (j_1, i_1, window_side_1) in windows_final:
            for (j_2, i_2, window_side_2) in windows_temp:
                if (windows_overlap(i_1, j_1, i_1+window_side_1, j_1+window_side_1,
                                    i_2, j_2, i_2+window_side_2, j_2+window_side_2)):
                    to_be_removed.add((j_2, i_2, window_side_2))

        windows_temp = windows_temp.difference(to_be_removed)
        if windows_temp != set():
            windows_temp = remove_overlapping_windows_with_same_window_side(first_frame_binarized, windows_temp)
        windows_final = windows_final.union(windows_temp)
        
    to_be_removed = set()
    for (j, i, window_side) in windows_final:
        if np.count_nonzero(last_frame_binarized[i:i+window_side,j:j+window_side]) < 50:
            to_be_removed.add((j, i, window_side))
    windows_final = windows_final.difference(to_be_removed)

    windows_final = list(windows_final)
    
    boxes = []
    for (j, i, window_size) in windows_final:
        image = first_frame_binarized[i:i+window_size, j:j+window_size]
        boxes.append(image)
        
    ### 2. IDENTIFYING THE CHARACTERS LOCATED WITHIN THE BOXES
    boxes_dicts = []

    for index in range(len(windows_final)):
        (j, i, window_slide) = windows_final[index]
    
        image_box = boxes[index]
    
        image_filtered = np.zeros((im_h, im_w))
        image_filtered[i:i+window_slide,j:j+window_slide] = first_frame_binarized[i:i+window_slide,j:j+window_slide]
        pixels_locations = np.array(np.nonzero(image_filtered))
        pixels_center = np.round(np.mean(pixels_locations, axis=1)).astype(int)
    
        pixels = first_frame_normalized[pixels_locations[0,:], pixels_locations[1,:]]
        black_pixels = np.logical_and.reduce((pixels[:,0] < 150, 
                                          pixels[:,1] < 150, 
                                          pixels[:,2] < 80))
        nb_pixels = pixels_locations.shape[1]
        nb_black_pixels = np.sum(black_pixels)
        ratio_black = nb_black_pixels / nb_pixels
        if (ratio_black < 0.5):
            character_type = 'operator'
        else:
            character_type = 'digit'
    
        boxes_dicts.append(dict(image_box=image_box, center=pixels_center, character_type=character_type, ratio_black=ratio_black)) # The last attribute ratio_black can help with calibrating the mask
        
    ### 3. CLASSIFICATION
    ## 3.1 DIGIT CLASSIFIER: TRANSLATED/SCALED/ROTATED MNIST CNN
    
    ## 3.2 OPERATOR CLASSIFIER: [see the function 'classify_operator']
    
    ## 3.3 CHARACTER CLASSIFIER
    for dct in boxes_dicts:
        character_type = dct['character_type']
        image_box = dct['image_box']
        if character_type == 'operator':
            dct['value'] = classify_operator(image_box)
        else:
            dct['value'] = classify_number(image_box)
            
    ### 4. TRACING THE FORMULA
    char_sequence = '' # This string will represent the traced formula
    char_sequence_list = [] # This list will represent at each index 
                            # the current state of the formula for the corresponding frame (needed for video)
    char_locs = [dct['center'] for dct in boxes_dicts]
    char_vals = [dct['value'] for dct in boxes_dicts]

    for i in range(len(frames)):
        arrow_location = arrow_locations[i]
    
        dist_function = lambda p1, p2: np.sqrt(((p1-p2)**2).sum())
        distances = [dist_function(loc, arrow_location) for loc in char_locs]
        closest_distance = np.min(distances)
    
        if (closest_distance < 50): # Should we take a larger/smaller threshold?
            closest_char = char_vals[np.argmin(distances)]
            if (not char_sequence) or (char_sequence[-1] != closest_char):
                char_sequence += closest_char
        char_sequence_list.append(char_sequence)
        
    assert(char_sequence[-1] == '=')
    result = eval(char_sequence[:-1]) # We discard the '=' operator and then evaluate the formula
    
    ### 5. RE-MAKING THE VIDEO
    frames_newvid = []
    position = (30, 450)
    tracking_color = (191, 16, 21)

    for i in range(len(frames)):
        frame = frames[i].copy()
        formula = char_sequence_list[i]
        if formula:
            if formula[-1] == '=':
                formula += str(result)
    
        frame_newvid = cv.putText(frame,
                                  'Formula: '+formula,
                                  position, 
                                  cv.FONT_HERSHEY_SIMPLEX,
                                  1, # font size
                                  (0, 80, 280, 255),
                                  2) # thickness

        for j in range(i-3):
            location_1 = (arrow_locations[j][1], arrow_locations[j][0])
            location_2 = (arrow_locations[j+1][1], arrow_locations[j+1][0])
            cv.line(frame_newvid, location_1, location_2, tracking_color, 1) 
            cv.circle(frame_newvid, location_2, radius=3, color=tracking_color, thickness=-1)
    
        frames_newvid.append(cv.cvtColor(frame_newvid, cv.COLOR_BGR2RGB))

    video = cv.VideoWriter(output_path, 0, fps, (im_w, im_h))

    for frame_newvid in frames_newvid:
        video.write(frame_newvid)

    cv.destroyAllWindows()
    video.release()