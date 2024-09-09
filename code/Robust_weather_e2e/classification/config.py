import os

class GlobalConfig:
    """ base architecture configurations """
    weather_mode = False
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    # Weather classes (09.08)
    weather_classes = 2
    weather_loss_weight = 0.2

    # region
    # train_root_dir = '/home/midas/URP/transfuser-cvpr2021/data/14_weathers_minimal_data/Train'
    # validate_root_dir = '/home/midas/URP/transfuser-cvpr2021/data/14_weathers_minimal_data/Validation'
    # train_weathers = ['14_weathers_minimal_data']
    # train_towns = ['Town_01', 'Town_02', 'Town_03', 'Town_04', 'Town_05', 'Town_06', 'Town_07', 'Town_10']
    
    # val_towns = ['Town05']
    # train_data, val_data = [], []
    # for weather in train_weathers:
    #     train_weather_dir = os.path.join(train_root_dir, weather)
    #     for town in train_towns:    
    #         if os.path.isdir(os.path.join(train_weather_dir, town+'_short')):
    #             train_data.append(os.path.join(train_weather_dir, town+'_short'))
    #         if os.path.isdir(os.path.join(train_weather_dir, town+'_tiny')):
    #             train_data.append(os.path.join(train_weather_dir, town+'_tiny'))
    #         if os.path.isdir(os.path.join(train_weather_dir, town+'_long')):
    #             train_data.append(os.path.join(train_weather_dir, town+'_long'))
                
    # val_data.append(validate_root_dir)
    # endregion


    # At remote server?

    train_root_dir = '/home/midas/URP/transfuser-cvpr2021/data/14_weathers_minimal_data/Train'
    validate_root_dir = '/home/midas/URP/transfuser-cvpr2021/data/14_weathers_minimal_data/Validation'
                    #'/home/midas/URP/transfuser-cvpr2021/data/14_weathers_minimal_data/Validation'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    # train_towns = ['Town03', 'Town04', 'Town06']
    val_towns = ['Town05']
    train_data, val_data = [], []
    
        
    for town in train_towns:    
        if os.path.isdir(os.path.join(train_root_dir, town+'_short')):
            train_data.append(os.path.join(train_root_dir, town+'_short'))
        if os.path.isdir(os.path.join(train_root_dir, town+'_tiny')):
            train_data.append(os.path.join(train_root_dir, town+'_tiny'))
        if os.path.isdir(os.path.join(train_root_dir, town+'_long')):
            train_data.append(os.path.join(train_root_dir, town+'_long'))
    
    for town in val_towns:
        if os.path.isdir(os.path.join(validate_root_dir, town+'_short')):
            val_data.append(os.path.join(validate_root_dir, town+'_short'))

    # visualizing transformer attention maps
    viz_root = '/home/park/Park/009_weather_e2e/003_Code/MIDAS/Midas_Transfuser/data/expert/Validation' # 어디로...?
    # viz_towns = ['Town05_tiny']
    viz_towns = ['Town05_short']
    viz_data = []
    for town in viz_towns:
        viz_data.append(os.path.join(viz_root, town))

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256 # Crop image 256 x 256
    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-5 # learning rate

    # Conv Encoder # 8x8 anchor로 볼 수 있다.
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1  #drop out은 얼마나 버리고 갈지를 나타내는 수치로? 봐도 무방하다
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
