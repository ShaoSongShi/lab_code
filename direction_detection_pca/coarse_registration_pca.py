import numpy as np
import tifffile
from scipy.ndimage import affine_transform
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import os
import concurrent.futures
from functools import partial
import time
from PIL import Image
import csv
from natsort import natsorted
import pandas as pd

def load_tiff_stack(path):
    """加载多层TIFF图像"""
    stack = tifffile.imread(path)
    if stack.ndim == 3:  # Z x H x W
        return stack
    elif stack.ndim == 4:  # Z x C x H x W (多通道)
        return stack.mean(axis=1)  # 转为灰度
    else:
        raise ValueError("不支持的TIFF维度")

def get_2d_mask(single_layer):
    """创建二值化三维mask"""
    normalized = single_layer.astype(np.float32) / single_layer.max()
    #print(normalized)
    # 阈值处理
    thresh_value = 0.0001  # 可调整的阈值
    # 转换为灰度图像
    gray = normalized
    # 阈值处理（150为阈值，255为最大值，THRESH_BINARY为二值化类型）
    ret, binary = cv2.threshold(gray, thresh_value, 1, cv2.THRESH_BINARY) # ret是实际使用的阈值，binary是二值化结果
    #print(f"实际使用的阈值：{ret}")
    #print(binary)
        # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 高斯模糊
    binary = cv2.GaussianBlur(binary, (5, 5), 0)

    return binary

def extract_contours(binary_image, min_area=10):
    """提取二值图像中的大块轮廓，并进行平滑处理"""
    if binary_image.dtype != np.uint8:
        binary_image = (binary_image > 0).astype(np.uint8) * 255

    # 提取轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选大面积轮廓
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not large_contours:
        raise ValueError("未找到符合条件的大块轮廓，请调整 min_area 参数或检查输入图像。")

    # 找到最大轮廓
    max_contour = max(large_contours, key=cv2.contourArea)

    # 平滑轮廓
    epsilon = 0.005 * cv2.arcLength(max_contour, True)
    smoothed_contour = cv2.approxPolyDP(max_contour, epsilon, True)

    # 可视化轮廓
    # visualize_contours(binary_image, [smoothed_contour])

    return smoothed_contour.reshape(-1, 2)  # 返回二维数组

def visualize_contours(image, contours):
    """可视化原始图像和提取的轮廓"""
    # 创建一个彩色图像以便绘制轮廓
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Large Contours Visualization")
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.show()

def compute_pca(contour_points):
    """对轮廓点进行PCA"""
    # 确保 contour_points 的形状是 (n_points, 2)
    if contour_points.ndim == 3 and contour_points.shape[1] == 1:
        contour_points = contour_points.reshape(-1, 2)  # 去掉中间的多余维度

    pca = PCA(n_components=1)
    pca.fit(contour_points)
    print("PCA Components:\n", pca.components_)
    #print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)
    return pca.components_

def compute_2d_pca(mask):
    """计算2维PCA主成分"""
    points = np.array(np.where(mask > 0)).T  # 获取所有体素坐标
    pca = PCA(n_components=1)
    pca.fit(points)
    #print(pca.components_)
    return pca.components_  # 返回1个主成分向量

def compute_2d_rotation_matrix(source_pca, target_pca):
    """计算2维旋转矩阵"""
    source_dir = source_pca[0] / np.linalg.norm(source_pca[0])  # 归一化
    target_dir = target_pca[0] / np.linalg.norm(target_pca[0])  # 归一化
    
    # 计算旋转角度
    cos_theta = np.dot(source_dir, target_dir)
    sin_theta = np.sqrt(1 - cos_theta**2)
    #print(cos_theta)
    
    # 构造旋转矩阵
    R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    #print(f"R:{R}")
    return R

def generate_deformation_field(transform, image_shape):
    """生成位移场"""
    height, width = image_shape
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    
    # 构建齐次坐标
    homog_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords.ravel())], axis=1)
    
    # 应用变换矩阵
    transformed = (transform @ homog_coords.T).T
    dx = transformed[:, 0].reshape(x_coords.shape) - x_coords
    dy = transformed[:, 1].reshape(y_coords.shape) - y_coords
    
    return np.stack([dx, dy], axis=-1)  # H x W x 2
def apply_2d_rotation(stack, R, center=None):
    """应用2维旋转"""
    aligned_stack = []

    png = stack[int(len(stack) / 2)]
    
    # 获取图像的尺寸
    height, width = stack[0].shape
    
    # 如果没有指定中心，则使用图像的中心
    if center is None:
        center = np.array([width / 2.0, height / 2.0])
    
    # 创建平移矩阵，先将图像平移到原点
    T1 = np.array([[1, 0, -center[0]],
                   [0, 1, -center[1]],
                   [0, 0, 1]])
    
    # 创建旋转矩阵，并将其扩展为3x3矩阵
    R_expanded = np.eye(3)
    R_expanded[:2, :2] = R
    
    # 创建反向平移矩阵，将图像平移回原来的位置
    T2 = np.array([[1, 0, center[0]],
                   [0, 1, center[1]],
                   [0, 0, 1]])
    
    # 组合变换矩阵: T2 * R * T1
    transform = T2 @ R_expanded @ T1

    # 生成位移场
    deformation = generate_deformation_field(transform, (height, width))

    # 应用仿射变换
    rotated_data = affine_transform(png, transform[:2, :2], offset=transform[:2, 2],
                                    order=1, mode='constant', cval=0.0, prefilter=False)
    '''
    for z_ in range(len(stack)):
        data = stack[z_].astype(np.float32)
        
        # 应用仿射变换
        rotated_data = affine_transform(data, transform[:2, :2], offset=transform[:2, 2],
                                        order=1, mode='constant', cval=0.0, prefilter=False)
        aligned_stack.append(rotated_data)
    return np.array(aligned_stack),deformation
    '''
    return rotated_data,deformation


def plot_save_with_arrow(img, start, end, aligned_png_path, title=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    # print(f"start:{start}")
    # print(f"end:{end}")
    # 放大箭头尺寸：head_width 和 head_length 调大
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_width=20, head_length=20, fc='red', ec='red', linewidth=3)
    
    # 计算角度（以度为单位）
    # 计算角度（以垂直向上为0°基准，顺时针方向为正）
    d = np.array([end[0] - start[0], end[1] - start[1]])  # 方向向量 [dx, dy]
    angle_rad = np.arctan2(d[0], d[1])
    angle_deg = np.degrees(angle_rad) 
    angle_deg = (angle_deg + 360) % 360  # 标准化到0-360度
    
    plt.text(10, 20, f"Angle: {angle_deg:.1f}°", color='red', fontsize=12)
    
    if title:
        plt.title(title)
    else:
        plt.title("Direction Detection")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(aligned_png_path)
    # plt.show()
    return angle_deg
    
def pca_align_2d(template_stack, target_stack):
    """2维PCA对齐主函数"""
    z_template = len(template_stack)
    z_target = len(target_stack)
    # 获取模板和目标的三维mask
    # 取中间层
    # print(int(z_template/2))
    # print(int(z_target/2))
    
    template_mask = get_2d_mask(template_stack[int(z_template/2)])
    target_mask = get_2d_mask(target_stack[int(z_target/2)])

    template_max_contour = extract_contours(template_mask)
    target_max_contour = extract_contours(target_mask)

    template_contour_pca = compute_pca(template_max_contour)
    #template_contour_pca =[0,1]
    target_contour_pca = compute_pca(target_max_contour)

    
    
    # 计算各自的主方向
    #template_pca = compute_2d_pca(template_mask)
    #template_pca = [0,1]
    #target_pca = compute_2d_pca(target_mask)
    
    # 计算旋转矩阵
    # R = compute_2d_rotation_matrix(target_pca, template_pca)
    R_contour = compute_2d_rotation_matrix(target_contour_pca, template_contour_pca)
    
    # 应用2维旋转
    aligned,deformation = apply_2d_rotation(target_stack, R_contour)

    aligned_png = aligned

    direction = np.dot(R_contour ,target_contour_pca[0])/np.linalg.norm(np.dot(R_contour ,target_contour_pca[0]))
    points_array = np.array(target_max_contour)
    center_x = np.mean(points_array[:, 0])
    center_y = np.mean(points_array[:, 1])
    center = np.array([center_x, center_y])
    # print(f"center:{center}")
    arrow_start = center
    arrow_end = center + direction * 60
    # print(f"direction:{direction}")

    return aligned,deformation,aligned_png,arrow_start,arrow_end

    
def align_zstack_2d_pca(template_path, target_path, output_path,deformation_path,aligned_png_path,csv_file,png_path):
    # 1. 加载数据
    template = load_tiff_stack(template_path)
    target = load_tiff_stack(target_path)
    
    # 2. PCA对齐
    aligned,deformation,aligned_png,arrow_start,arrow_end = pca_align_2d(template, target)
    cv2.imwrite(png_path, aligned_png)
    
    # 3. 保存结果
    # tifffile.imwrite(output_path, aligned.astype(template.dtype))
    print(f"Aligned stack saved to {output_path}")

    np.save(deformation_path, deformation)
    print(f"变形场已保存至: {deformation_path}")

    
    angle_deg = plot_save_with_arrow(aligned_png, arrow_start, arrow_end, aligned_png_path)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        png_path = os.path.join(os.path.basename(output_path).split('.')[0] + '.png')
        writer.writerow([png_path, angle_deg])  # 使用csv writer写入一行
        print(f"角度已保存至: {csv_file}")
        # print([os.path.basename(output_path), angle_deg])
    print(f"png已保存至: {png_path}")

def load_images_from_folder(folder_path):
    images_paths = []
    '''
    prefix_to_remove = 'PCAaligned_'
    # aligned_filenames = sorted(os.listdir("aligned_images"))
    aligned_filenames = os.listdir('/home/user/ShenRuihong/aligned_240529_RLD60_f')
    aligned_image_names = []
    for aligned_filename in aligned_filenames:
        if aligned_filename.startswith(prefix_to_remove):
                    filename = aligned_filenames[len(prefix_to_remove):]
                    aligned_image_names.append(filename)
    '''
    # filenames = os.listdir(folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
        # if filename.endswith(".tif") or filename.endswith(".tiff") and filename not in aligned_image_names:
            file_path = os.path.join(folder_path, filename)
            images_paths.append(file_path)
    return images_paths

def process_single_image(template_path, image_path, aligned_dir, deformation_dir,aligned_png_dir, csv_file,png_dir):
    """处理单个图像的任务函数"""
    
    try:
        base_name = os.path.basename(image_path)
        png_path = os.path.join(png_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        output_path = os.path.join(aligned_dir, f"PCAaligned_{base_name}")
        deformation_path = os.path.join(deformation_dir, f"Deformation_{os.path.splitext(os.path.basename(image_path))[0]}.npy")
        output_png_path = os.path.join(aligned_png_dir, f"PCAaligned_{os.path.splitext(os.path.basename(image_path))[0]}.png")
        print(f"Processing: {base_name}")
        align_zstack_2d_pca(
            template_path=template_path,
            target_path=image_path,
            output_path=output_path,
            deformation_path=deformation_path,
            aligned_png_path=output_png_path,
            csv_file = csv_file,
            png_path = png_path
        )

        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    '''
    csv_file = 'fish_direction_detection_results.csv'
    aligned_folder_path = "aligned_images"
    fixed_folder_path = "fixed_target"
    deformation_folder_path = "deformation_field"
    template_path="Ref-zbb1_downsampled.tif"
    aligned_png_dir = "aligned_png"
    mask_dir = "mask"

    for image_path in os.listdir(fixed_folder_path):
        print(f"Aligning image: {image_path}")
        process_single_image(template_path, 
                                image_path = os.path.join(fixed_folder_path,image_path),
                                aligned_dir=aligned_folder_path,
                                deformation_dir=deformation_folder_path,
                                aligned_png_dir=aligned_png_dir,
                                csv_file = csv_file
                                )
    '''
    aligned_folder_path = "./aligned"
    fixed_folder_path = "/home/d2/ShenRuihong/XLFM_dataset/fixed_fish/f1/RLD60"
    deformation_folder_path = "./deformation_field"
    template_path="/home/d2/ShenRuihong/XLFM_dataset/reg/template/Ref-zbb1_downsampled.tif"
    aligned_png_dir = "./aligned_png_f1"
    csv_file = './fish_direction_detection_results_f1.csv'
    png_dir = './png'

    fixed_target_images_paths = load_images_from_folder(fixed_folder_path)
    '''
    for image_path in fixed_target_images_paths:
        print(f"Aligning image: {image_path}")
        align_zstack_2d_pca(
            template_path=template_path,
            target_path=image_path,
            output_path=os.path.join(aligned_folder_path,f'PCAaligned_{os.path.basename(image_path)}')
    )
    '''
        # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 创建部分函数固定模板路径和输出目录
        process_func = partial(process_single_image, 
                             template_path, 
                             # image path在并行时传入
                             aligned_dir=aligned_folder_path,
                             deformation_dir=deformation_folder_path,
                             aligned_png_dir=aligned_png_dir,
                             csv_file = csv_file,
                             png_dir = png_dir)
        
        # 提交所有任务
        futures = [executor.submit(process_func, path) for path in fixed_target_images_paths]
        
        # 等待所有任务完成并收集结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # 计算总耗时
    total_time = time.time() - start_time
    avg_time = total_time / len(fixed_target_images_paths) if len(fixed_target_images_paths) > 0 else 0
    
    # 统计处理结果
    success_count = sum(results)
    print("\n===== Processing Summary =====")
    print(f"Total images processed: {len(fixed_target_images_paths)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(fixed_target_images_paths) - success_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.2f} seconds")
    print(f"Processing speed: {len(fixed_target_images_paths)/total_time:.2f} images/second")
