import copy
import numpy as np

def filter_gt_annos_by_distance(gt_annos):
    """
    杩囨护 gt_annos 涓窛绂诲師鐐硅秴杩?25 鐨勭洰鏍囥€?
    鍙傛暟:
        gt_annos (list of dict): 杈撳叆鐨勬爣娉ㄦ暟鎹垪琛紝姣忎釜鍏冪礌鏄竴涓瓧鍏革紝
                                 鍖呭惈 'location' 瀛楁鍙婂叾浠栧瓧娈点€?
    杩斿洖:
        list of dict: 杩囨护鍚庣殑鏍囨敞鏁版嵁鍒楄〃銆?    """
    filtered_annos = []

    for anno in gt_annos:
        new_anno = {}
        locations = np.array(anno['location'])  # 杞崲涓?NumPy 鏁扮粍浠ヤ究璁＄畻
        distances = np.linalg.norm(locations, axis=1)  # 璁＄畻鍒板師鐐圭殑璺濈
        valid_indices = np.where(distances <= 25)[0]  # 鑾峰彇鏈夋晥绱㈠紩

        # 閬嶅巻鎵€鏈夐敭鍊硷紝鍙繚鐣欎笌鏈夋晥绱㈠紩瀵瑰簲鐨勯」
        for key in anno:
            if isinstance(anno[key], list) and len(anno[key]) == len(locations):
                new_anno[key] = [anno[key][i] for i in valid_indices]
            else:
                new_anno[key] = anno[key]  # 闈炲垪琛ㄥ瓧娈电洿鎺ュ鍒?
        filtered_annos.append(new_anno)

    return filtered_annos

# 绀轰緥鐢ㄦ硶
original_gt_annos = [
    {
        'name': ['Car', 'Pre'],
        'truncated': [0.1, 0.2],
        'occluded': [1.1, 1.2],
        'alpha': [2.1, 2.2],
        'bbox': [[110, 120, 130, 140], [115, 125, 135, 145]],
        'location': [[1, 2, 3], [25, 3, 4]],
        'dimensions': [[1.5, 1.6, 1.7], [1.8, 1.9, 2.0]],
        'rotation_y': [0.5, 0.6],
        'score': [0.9, 0.8]
    }
]

# 娣辨嫹璐濆師濮嬫暟鎹苟杩囨护
copied_gt_annos = copy.deepcopy(original_gt_annos)
filtered_gt_annos = filter_gt_annos_by_distance(copied_gt_annos)

# 杈撳嚭澶勭悊鍚庣殑缁撴灉
print("鍘熷鏁版嵁锛?, original_gt_annos)
print("澶勭悊鍚庣殑鏂版暟鎹細", filtered_gt_annos)
# *************************#
# for anno in gt_annos:
#     locations = np.array(anno['location'])
#     distances = np.linalg.norm(locations, axis=1)
#     valid_indices = np.where(distances <= 25)[0]
#     print(distances)
#     for key in anno:
#         if len(anno[key]) == len(locations):
#             anno[key] = np.array([anno[key][i] for i in valid_indices],anno[key].dtype)
# for anno in dt_annos:
#     locations = np.array(anno['location'])
#     distances = np.linalg.norm(locations, axis=1)
#     valid_indices = np.where(distances <= 25)[0]
#
#     for key in anno:
#         if len(anno[key]) == len(locations):
#             anno[key] = np.array([anno[key][i] for i in valid_indices],anno[key].dtype)
# *************************#
