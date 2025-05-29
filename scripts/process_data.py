import cv2
import numpy as np
import torch
import glob
import os

def select_open_mouth():
    path = '/mnt/gregory/smile/data/detect/images/train/'
    for img_path in glob.glob(path + '*.png'):
        depth_path = img_path.replace('detect', 'depth').replace('images/train', 'ffhq_gray')
        label_path = img_path.replace('images', 'labels').replace('png', 'txt')
        mmask_path = img_path.replace('detect/images', 'mmask')

        im = cv2.imread(img_path) # (1024,1024,3)
        mmask = cv2.imread(mmask_path)
        w,h = im.shape[:2]
        depth = cv2.imread(depth_path)
        img_sz = 256
        if isinstance(img_sz, int):
            size = (img_sz, img_sz)
        else:
            size = img_sz
        crop_size = 0
        if crop_size > 0:
            im = cv2.resize(im[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
            mmask = cv2.resize(mmask[crop_size:-crop_size,crop_size:-crop_size], size, interpolation=cv2.INTER_LINEAR)
        
        edge = np.zeros_like(im)
        mask = np.zeros_like(im)
        with open(label_path, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            tooth_ids = [int(x[0]) for x in lb] 
            f.close()
        if crop_size > 0:
            segments = [((segment*w-crop_size)/(w-2*crop_size)*size[0]).astype(np.int32) for segment in segments]
        else:
            segments = [(segment*w).astype(np.int32) for segment in segments]

        for segment_idx in range(len(segments)):
            id = tooth_ids[segment_idx]+1
            segment = segments[segment_idx]
            for i in range(len(segment) - 1):
                start_point = tuple(segment[i])    # 当前点
                end_point = tuple(segment[i + 1])  # 下一点
                cv2.line(edge, start_point, end_point, color=(255, 255, 255), thickness=2)
            cv2.line(edge, tuple(segment[-1]), tuple(segment[0]), color=(255, 255, 255), thickness=2)
            
            segment = np.reshape(segment, (1, -1, 2))
            cv2.fillPoly(mask, segment, color=(255, 255, 255))
        cv2.imwrite(img_path.replace('detect/images/train', 'tmask'), mask)
        continue
        upper = np.logical_and(mask > 0, mask <= 16)
        lower = np.logical_and(mask > 16, mask <= 32) 
        open_mouth = False
        if lower.sum() > 6000 and upper.sum() - lower.sum() <4000:
            open_mouth = True
        if lower.sum() ==0 or upper.sum() ==0:
            continue
        if lower.sum() < 4000 or upper.sum() <4000:
            continue
        if np.where(upper)[0].max()+10 < np.where(lower)[0].min():
            open_mouth = True  
        if open_mouth:
            save_path = img_path.replace('images/train', 'cropped')
            cv2.imwrite(save_path, im)
            cv2.imwrite(save_path.replace('cropped', 'cropped_label'), mask)
    
def change_color():
    def compute_color_stats(image, mask):
        """计算mask区域内LAB颜色空间的均值和标准差"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab.astype("float32"))
        mask = mask > 0
        return {
            'l_mean': l[mask].mean(),
            'l_std': l[mask].std(),
            'a_mean': a[mask].mean(),
            'a_std': a[mask].std(),
            'b_mean': b[mask].mean(),
            'b_std': b[mask].std()
        }

    def histogram_matching(source, target, source_mask, target_mask):
        """基于直方图的颜色迁移（不要求mask形状一致）"""
        # 转换为LAB空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        
        # 只处理mask区域
        source_pixels = source_lab[source_mask > 0]
        target_pixels = target_lab[target_mask > 0]
        
        # 对每个通道进行直方图匹配
        matched = target_lab.copy()
        for ch in range(1, 3):  # 只处理a、b通道（保持明度不变）
            # 计算累积直方图
            src_hist, _ = np.histogram(source_pixels[:, ch], bins=256, range=(0,255))
            tgt_hist, _ = np.histogram(target_pixels[:, ch], bins=256, range=(0,255))
            
            src_cdf = np.cumsum(src_hist) / src_hist.sum()
            tgt_cdf = np.cumsum(tgt_hist) / tgt_hist.sum()
            
            # 建立映射关系
            lut = np.interp(tgt_cdf, src_cdf, np.arange(256))
            matched[:,:,ch][target_mask > 0] = lut[target_lab[:,:,ch][target_mask > 0]]
        
        return cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    def remove_tooth_discoloration(img_path, mask_path, output_path, intensity=.1):
        """
        专业牙齿去渍算法
        参数：
        - intensity: 校正强度（0.0~1.0）
        """
        # 1. 读取图像和mask
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 转换为LAB色彩空间
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab.astype("float32"))
        
        # 3. 检测异色区域（基于ab通道异常值）
        tooth_pixels = lab[mask > 0]
        a_mean, a_std = tooth_pixels[:,1].mean(), tooth_pixels[:,1].std()
        b_mean, b_std = tooth_pixels[:,2].mean(), tooth_pixels[:,2].std()
        
        # 4. 自适应颜色校正（核心算法）
        def correct_channel(ch, mean, std):
            """非线性校正函数"""
            delta = ch - mean
            # 双曲线衰减模型：保留±1σ内的变化，衰减之外的异常值
            scale = 1 / (1 + np.abs(delta)/(3*std)) ** intensity
            return mean + delta * scale
        
        a_corrected = correct_channel(a, a_mean, a_std)
        b_corrected = correct_channel(b, b_mean, b_std)
        
        # 5. 纹理恢复（保持原始L通道的细节）
        corrected_lab = cv2.merge([l, a_corrected, b_corrected])
        
        # 6. 高频细节融合
        orig_hf = cv2.GaussianBlur(lab - cv2.GaussianBlur(lab, (0,0), 3), (0,0), 1)
        result_lab = corrected_lab
        
        # 7. 边缘羽化
        blurred_mask = cv2.GaussianBlur(mask, (7,7), 0) / 255.0
        final = cv2.cvtColor(result_lab.clip(0,255).astype("uint8"), cv2.COLOR_LAB2RGB)
        final = img * (1 - blurred_mask[..., np.newaxis]) + final * blurred_mask[..., np.newaxis]
        
        cv2.imwrite(output_path, cv2.cvtColor(final.astype("uint8"), cv2.COLOR_RGB2BGR))
    image_paths = glob.glob('/mnt/gregory/smile/data/detect/images/train/*.png')
    mask_paths = [img_path.replace('detect/images/train','tmask') for img_path in image_paths]
    img_path = '/mnt/gregory/smile/data/detect/images/train/00103.png'
    ref_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(img_path.replace('detect/images/train','tmask'), cv2.IMREAD_GRAYSCALE)
    for img_path, mask_path in zip(image_paths, mask_paths):

        # ref_stats = compute_color_stats(ref_img, ref_mask)
        
        target_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        target_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 方法1：直方图匹配（推荐）
        result = histogram_matching(ref_img, target_img, ref_mask, target_mask)
        
        # 方法2：统计匹配（备选方案）
        # result = statistical_transfer(ref_stats, target_img, target_mask)
        
        # 边缘融合
        blurred_mask = cv2.GaussianBlur(target_mask, (5,5), 0) / 255.0
        result = (target_img * (1 - blurred_mask[..., np.newaxis]) + 
                    result * blurred_mask[..., np.newaxis])
        output_dir = '.'
        # 保存结果
        cv2.imwrite(img_path.replace('images/train','changed_image'), 
                    cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # break
    # remove_tooth_discoloration(os.path.join(output_dir, f"result.jpg"), mask_path, os.path.join(output_dir, f"result1.jpg"))
if __name__ == '__main__':
    change_color()