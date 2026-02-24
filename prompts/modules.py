MODULES_SIGNATURES = """
\"\"\"
Locates objects in an image. Object prompts should be simple and contain few words. To return all objects, pass "objects" as the prompt.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate.
Returns:
    list: A list of bounding boxes [xmin, ymin, xmax, ymax] for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about an object shown in a bounding box.

Args:
    image (image): Image of the scene.
    question (string): Question about the object in the bounding box.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, bbox):

\"\"\"
Returns the depth of an object (specified by a bounding box) in meters.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    float: The depth of the object specified by the coordinates in meters.
\"\"\"
def depth(image, bbox):

\"\"\"
Checks if two bounding boxes correspond to the same object.

Args:
    image (image): Image of the scene.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing object1.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing object2.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, bbox1, bbox2):

\"\"\"
Returns the width and height of the object in 2D pixel space.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    tuple: (width, height) of the object in 2D pixel space.
\"\"\"
def get_2D_object_size(image, bbox):

"""

MODULES_SIGNATURES_CLEVR = """
\"\"\"
Locates objects in an image. Object prompts should be 1 WORD MAX.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate. Examples: "spheres", "objects".
Returns:
    list: A list of x,y coordinates for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about the attributes of an object specified by an x,y coordinate.
Should not be used for other kinds of questions.

Args:
    image (image): Image of the scene.
    question (string): Question about the objects attribute to answer. Examples: "What color is this?", "What material is this?"
    x (int): X coordinate of the object in pixel space.
    y (int): Y coordinate of the object in pixel space. 
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, x, y):

\"\"\"
Returns the depth of an object specified by an x,y coordinate.

Args:
    image (image): Image of the scene.
    x (int): X coordinate of the object in pixel space.
    y (int): Y coordinate of the object in pixel space.

Returns:
    float: The depth of the object specified by the coordinates.
\"\"\"
def depth(image, x, y):

\"\"\"
Checks if two pairs of coordinates correspond to the same object.

Args:
    image (image): Image of the scene.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, x_1, y_1, x_2, y_2):
"""

VIDEO_MODULES_SIGNATURES = """
\"\"\"
Loads a specific frame from the video as a PIL Image.

Args:
    video_dir (string): Path to directory of extracted video frames.
    frame_index (int): 0-based index of the frame to load.

Returns:
    image: The frame as a PIL Image.
\"\"\"
def get_frame(video_dir, frame_index):

\"\"\"
Tracks an object across video frames starting from a point on an initial frame.
Returns a dictionary mapping frame indices to bounding boxes for every frame where the object is visible.

Args:
    video_dir (string): Path to directory of extracted video frames.
    init_frame_index (int): Frame index where the object is first identified.
    x (int): X coordinate of the object in the initial frame (pixel space).
    y (int): Y coordinate of the object in the initial frame (pixel space).

Returns:
    dict: {frame_index (int): [x1, y1, x2, y2]} bounding boxes per frame where the object is visible.
\"\"\"
def track(video_dir, init_frame_index, x, y):

\"\"\"
Locates objects in a specific video frame. Object prompts should be simple and contain few words.

Args:
    video_dir (string): Path to directory of extracted video frames.
    frame_index (int): Frame index to search.
    object_prompt (string): Description of object to locate.

Returns:
    list: A list of bounding boxes [xmin, ymin, xmax, ymax] for all of the objects located in pixel space.
\"\"\"
def loc_frame(video_dir, frame_index, object_prompt):

\"\"\"
Answers a question about an object in a specific video frame.

Args:
    video_dir (string): Path to directory of extracted video frames.
    frame_index (int): Frame index to analyze.
    question (string): Question about the object.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object. Pass None for holistic questions about the whole frame.

Returns:
    string: Answer to the question.
\"\"\"
def vqa_frame(video_dir, frame_index, question, bbox):

\"\"\"
Returns the final result. Must be called at the end of every program.

Args:
    var: The final answer.

Returns:
    string: The final result as a string.
\"\"\"
def result(var):
"""
