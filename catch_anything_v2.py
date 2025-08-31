import cv2
import numpy as np
from scipy import stats

# ======================【预定义参数区域】====================== #
VIDEO_PATH = "test_video/IMG_9045.mov"  # 📁 视频路径
HSV_LOWER = np.array((28, 0, 113))         # 🎨 HSV直方图下界
HSV_UPPER = np.array((180, 255, 255))      # 🎨 HSV直方图上界
view_r = 1.3                                # 👁️ 模板扩展视野倍数
area_stable_thresh = 0.2                   # 📏 面积变化容忍度
stable_count_thresh = 50                   # ✅ 模板更新所需稳定帧数
match_interval = 50                        # 🔁 每多少帧执行一次模板匹配（原frame_count % 50）
area_thresh_range = (0.5, 1.6)             # 🚨 面积异常恢复阈值 (下限，上限)
ESC_KEY = 27                               # 🧹 退出键
RESET_KEYS = [ord('r'), ord('R')]          # 🔁 手动重置键
FEATURE_MAX_CORNERS = 300                  # 💡 角点数量上限
save_video = False                              # 是否输出视频地址
save_path = "output_video/tracking_IMG_9045.MOV_V2.mp4"   # 输出地址

# ============================================================ #



# === 根据ROI区域大小动态调整参数 ===
def adjust_parameters_by_roi_size(w, h):
    area = w * h
    if area < 200:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.0005, minDistance=1, blockSize=3)
        lk_params = dict(winSize=(5, 5), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        canny_thresh = (30, 100)
    else:
        feature_params = dict(maxCorners=FEATURE_MAX_CORNERS, qualityLevel=0.01, minDistance=1, blockSize=7)
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        canny_thresh = (50, 150)
    return feature_params, lk_params, canny_thresh


# === 基于颜色相似性恢复目标区域 ===
def recover_by_color_similarity(frame, template_image, prev_scale):
    if template_image is None or template_image.size == 0:
        return None

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    lab_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2Lab)

    h, w = lab_template.shape[:2]
    roi = lab_template[h//3:2*h//3, w//3:2*w//3]
    roi_pixels = roi.reshape(-1, 3)

    dominant_color = np.array([
        stats.mode(roi_pixels[:, i], keepdims=False).mode for i in range(3)
    ], dtype=np.float32)

    diff = lab_frame.astype(np.float32) - dominant_color.reshape(1, 1, 3)
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    adaptive_threshold = np.percentile(distance.flatten(), 10)

    mask = (distance < adaptive_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh, fw = frame.shape[:2]
    cx1, cx2, cy1, cy2 = fw // 3, fw * 2 // 3, fh // 3, fh * 2 // 3
    best_match, min_diff = None, float('inf')

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x, center_y = x + w // 2, y + h // 2
        if cx1 <= center_x <= cx2 and cy1 <= center_y <= cy2 and w * h > 100:
            diff = abs(w * h - prev_scale)
            if diff < min_diff:
                min_diff = diff
                best_match = (x, y, w, h)

    return best_match


# === 模板匹配函数 ===
def template_match(frame, template):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = gray_template.shape
    return max_loc[0], max_loc[1], tw, th, max_val


# === CamShift 恢复 ===
def recover_target(frame, back_proj, last_bbox, prev_scale):
    x, y, w, h = last_bbox
    if w <= 0 or h <= 0:
        return last_bbox

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    _, track_window = cv2.CamShift(back_proj, (x, y, w, h), term_crit)
    x, y, w, h = track_window
    if 0.7 * prev_scale < w * h < 1.3 * prev_scale:
        return (x, y, w, h)
    return last_bbox


# === 主流程入口 ===
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    if not ret:
        print("❌ 无法读取视频")
        return

    bbox = cv2.selectROI("Select Drone", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Drone")
    x, y, w, h = bbox
    prev_scale = w * h

    initial_template = template_image = first_frame[y:y+h, x:x+w].copy()
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    hsv_roi = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(hsv_roi, HSV_LOWER, HSV_UPPER)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    feature_params, lk_params, _ = adjust_parameters_by_roi_size(w, h)
    p0 = cv2.goodFeaturesToTrack(old_gray[y:y+h, x:x+w], mask=None, **feature_params)
    if p0 is not None:
        p0[:, 0, 0] += x
        p0[:, 0, 1] += y

    color = np.random.randint(0, 255, (FEATURE_MAX_CORNERS, 3))
    mask = np.zeros_like(first_frame)
    frame_count = stable_count = 0
    last_area = prev_scale
    reset_flag = False

    cv2.namedWindow('Drone Tracking', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Template Image area now', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        warning_flag = 0
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        if 'template_image' in locals() and template_image is not None and template_image.size > 0:
            cv2.imshow('Template Image area now', template_image)
        if p0 is not None and len(p0) >= 3:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new, good_old = (p1[st == 1], p0[st == 1]) if p1 is not None else ([], [])

            if len(good_new) >= 3:
                median_point = np.median(good_new, axis=0)
                distances = np.linalg.norm(good_new - median_point, axis=1)
                keep = distances < 50
                good_new, good_old = good_new[keep], good_old[keep]

                if len(good_new) >= 3:
                    min_x, min_y = np.min(good_new, axis=0)
                    max_x, max_y = np.max(good_new, axis=0)

                    # x, y, w, h = int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)
                    if warning_flag != 1:
                        x, y, w, h = int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)
                    # else:
                    #     x, y, w, h = tx, ty, tw, th
                    area = w * h
                    feature_params, lk_params, _ = adjust_parameters_by_roi_size(w, h)

                    # print(f"area: {area}\nprev_scale: {prev_scale}")

                    # === 面积稳定检测 ===
                    if abs(area - last_area) / max(last_area, 1) < area_stable_thresh:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_area = area

                    if stable_count >= 50:
                        print("✅ 检测到优质模板 → 更新模板")
                        template_image = frame[y:y + int(view_r * h), x:x + int(view_r * w)].copy()

                        stable_count = 0

                    # === 定期模板匹配 ===
                    if frame_count % 50 == 0 or area == 0:
                        if area == 0:
                            warning_flag = 1
                            print("🔄 面积为0 → 执行模板匹配")
                            # ✅ 仅使用 initial_template 匹配
                            print("⚠️ area = 0 → 仅使用 initial_template 进行匹配")
                            if initial_template is not None and initial_template.size > 0:
                                tx, ty, tw, th, score = template_match(frame, initial_template)
                                if score > 0.7:
                                    print(f"🎯 initial_template 匹配成功 (score={score:.2f}) → 更新 template_image")
                                    x, y, w, h = tx, ty, tw, th
                                    prev_scale = w * h
                                    template_image = frame[y:y + int(view_r * h),
                                                     x:x + int(view_r * w)].copy()  # ✅ 更新优质模板

                                    # if 'template_image' in locals() and template_image is not None and template_image.size > 0:
                                    #     cv2.imshow('Template Image area 0', template_image)
                                    # ✅ 清空原有轨迹
                                    mask = np.zeros_like(frame)
                                    p0 = None
                                    # 重新初始化光流点
                                    p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None,
                                                                 **feature_params)
                                    # cv2.namedWindow("area = 0_frame_gray", cv2.WINDOW_NORMAL)
                                    # cv2.imshow("area = 0_frame_gray", frame_gray[y:y + h, x:x + w])

                                    if p0 is not None:
                                        p0[:, 0, 0] += x
                                        p0[:, 0, 1] += y
                                    else:
                                        print("面积为0但是找不到p0")

                            else:
                                print("❌ initial_template 无效 → 跳过匹配")

                        else:
                            print("🔄 每50帧时 → 执行初始模板匹配")
                            # ✅ 每50帧时，仍然使用当前优质模板匹配
                            if template_image is None or template_image.size == 0:
                                print("⚠️ 当前模板无效 → 使用 initial_template 替代")
                                template_image = initial_template.copy()
                            tx, ty, tw, th, score = template_match(frame, initial_template)
                            # print(score,"每50帧时")
                            if score > 0.8:

                                print(f"🎯 定期模板匹配成功 (score={score:.2f})")
                                x, y, w, h = tx, ty, tw, th
                                prev_scale = w * h
                                template_image = frame[y:y + int(view_r * h), x:x + int(view_r * w)].copy()
                                # 重新初始化光流点
                                p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                                # cv2.namedWindow("frame=50_frame_gray", cv2.WINDOW_NORMAL)
                                # cv2.imshow("frame=50_frame_gray", frame_gray[y:y + h, x:x + w])
                                if p0 is not None:
                                    p0[:, 0, 0] += x
                                    p0[:, 0, 1] += y
                                else:
                                    # 清空误判点
                                    p0 = None  # 清空旧光流点
                                    mask = np.zeros_like(frame)  # 清空轨迹图层
                                    print("定期匹配成功但是找不到p0")


                    # === 面积异常检测 ===
                    elif area < 0.5 * prev_scale or area > 1.6 * prev_scale:
                        # print(area/prev_scale)
                        warning_flag = 1
                        print("⚠️ 面积异常 → 初始模板匹配")
                        # if template_image is None or template_image.size == 0:
                        #     template_image = initial_template.copy()
                        # tx, ty, tw, th, score = template_match(frame, initial_template)
                        tx, ty, tw, th, score = template_match(frame, initial_template)
                        if score > 0.7:
                            warning_flag = 1
                            print(f"🎯 初始模板匹配成功 (score={score:.2f})")
                            # mask = np.zeros_like(frame)
                            x, y, w, h = tx, ty, tw, th
                            # print(x, y, w, h,"11111")
                            template_image = frame[y:y + h, x:x + w].copy()
                            # p0 = cv2.goodFeaturesToTrack(frame_gray[y:y+h, x:x+w], mask=None, **feature_params)
                            p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + int(view_r * h), x:x + int(view_r * w)],
                                                         mask=None, **feature_params)
                            # print(p0)
                            if p0 is not None:
                                p0[:, 0, 0] += x
                                p0[:, 0, 1] += y
                            else:
                                print("面积异常，但是找不到p0")
                            # template_image = frame[y:y + h, x:x + w].copy()
                            # if 'template_image' in locals() and template_image is not None and template_image.size > 0:
                            #     cv2.imshow('Template Image area change', template_image)
                            prev_scale = area
                        else:
                            prev_scale = area
                    else:
                        prev_scale = area
                        # template_image = frame[y:y+h, x:x+w].copy()

                    # 绘制跟踪
                    # if warning_flag == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # print(x, y, w, h, "222222")
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        if frame_count % 200 == 0:
                            mask = np.zeros_like(frame)
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 3, color[i % len(color)].tolist(), -1)

                    img = cv2.add(frame, mask)
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    img = frame
            else:
                # === 清空之前的光流轨迹 ===
                mask = np.zeros_like(frame)
                # === p0 为空情况也走 initial_template 恢复 ===
                print("⚠️ 光流点丢失 → 尝试 initial_template 匹配")
                if initial_template is not None and initial_template.size > 0:
                    tx, ty, tw, th, score = template_match(frame, initial_template)
                    if score > 0.5:
                        print(f"🎯 initial_template 匹配成功 (score={score:.2f}) → 重新初始化光流")
                        x, y, w, h = tx, ty, tw, th
                        prev_scale = w * h
                        # p0 = cv2.goodFeaturesToTrack(frame_gray[y:y+h, x:x+w], mask=None, **feature_params)
                        p0 = cv2.goodFeaturesToTrack(
                            frame_gray[y:y + int(view_r * h), x:x + int(view_r * w)], mask=None,
                            **feature_params)
                        if p0 is not None:
                            p0[:, 0, 0] += x
                            p0[:, 0, 1] += y
                    else:
                        print("❌ initial_template 匹配失败 → 尝试颜色找回")
                        result = recover_by_color_similarity(frame, initial_template, prev_scale)
                        if result is not None:
                            print("🎯 颜色找回成功 → 重新初始化光流")
                            x, y, w, h = result
                            prev_scale = w * h
                            # p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                            p0 = cv2.goodFeaturesToTrack(
                                frame_gray[y:y + int(view_r * h), x:x + int(view_r * w)], mask=None,
                                **feature_params)
                            if p0 is not None:
                                p0[:, 0, 0] += x
                                p0[:, 0, 1] += y
                        # else:
                        #     print("❌ 颜色找回失败 → CamShift 恢复")
                        #     x, y, w, h = recover_target(frame, back_proj, (x, y, w, h), prev_scale)
                        #     prev_scale = w * h
                        #     p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                        #     if p0 is not None:
                        #         p0[:, 0, 0] += x
                        #         p0[:, 0, 1] += y
                img = frame
        else:
            # === 清空之前的光流轨迹 ===
            mask = np.zeros_like(frame)
            # === p0 为空情况也走 initial_template 恢复 ===
            print("⚠️ 光流点丢失 → 尝试 initial_template 匹配")
            if initial_template is not None and initial_template.size > 0:
                tx, ty, tw, th, score = template_match(frame, initial_template)
                if score > 0.5:
                    print(f"🎯 initial_template 匹配成功 (score={score:.2f}) → 重新初始化光流")
                    x, y, w, h = tx, ty, tw, th
                    prev_scale = w * h
                    # p0 = cv2.goodFeaturesToTrack(frame_gray[y:y+h, x:x+w], mask=None, **feature_params)
                    p0 = cv2.goodFeaturesToTrack(
                        frame_gray[y:y + int(view_r * h), x:x + int(view_r * w)], mask=None,
                        **feature_params)
                    if p0 is not None:
                        p0[:, 0, 0] += x
                        p0[:, 0, 1] += y
                else:
                    print("❌ initial_template 匹配失败 → 尝试颜色找回")
                    result = recover_by_color_similarity(frame, initial_template, prev_scale)
                    if result is not None:
                        print("🎯 颜色找回成功 → 重新初始化光流")
                        x, y, w, h = result
                        prev_scale = w * h
                        # p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                        p0 = cv2.goodFeaturesToTrack(
                            frame_gray[y:y + h, x:x + w], mask=None,
                            **feature_params)
                        if p0 is not None:
                            p0[:, 0, 0] += x
                            p0[:, 0, 1] += y
                    # else:
                    #     print("❌ 颜色找回失败 → CamShift 恢复")
                    #     x, y, w, h = recover_target(frame, back_proj, (x, y, w, h), prev_scale)
                    #     prev_scale = w * h
                    #     p0 = cv2.goodFeaturesToTrack(frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                    #     if p0 is not None:
                    #         p0[:, 0, 0] += x
                    #         p0[:, 0, 1] += y

            img = frame

        if reset_flag:
            # 手动复位逻辑
            mask = np.zeros_like(frame)
            print("⚠️ 手动复位触发 → 尝试 initial_template 匹配")
            if initial_template is not None and initial_template.size > 0:
                tx, ty, tw, th, score = template_match(frame, initial_template)
                if score > 0.5:
                    print(f"🎯 initial_template 匹配成功 (score={score:.2f}) → 重新初始化光流")
                    x, y, w, h = tx, ty, tw, th
                    prev_scale = w * h
                    p0 = cv2.goodFeaturesToTrack(
                        frame_gray[y:y + int(view_r * h), x:x + int(view_r * w)],
                        mask=None, **feature_params)
                    if p0 is not None:
                        p0[:, 0, 0] += x
                        p0[:, 0, 1] += y
                else:
                    print("❌ initial_template 匹配失败 → 尝试颜色找回")
                    result = recover_by_color_similarity(frame, initial_template, prev_scale)
                    if result is not None:
                        print("🎯 颜色找回成功 → 重新初始化光流")
                        x, y, w, h = result
                        prev_scale = w * h
                        p0 = cv2.goodFeaturesToTrack(
                            frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                        if p0 is not None:
                            p0[:, 0, 0] += x
                            p0[:, 0, 1] += y
                    # else:
                    #     print("❌ 颜色找回失败 → CamShift 恢复")
                    #     x, y, w, h = recover_target(frame, back_proj, (x, y, w, h), prev_scale)
                    #     prev_scale = w * h
                    #     p0 = cv2.goodFeaturesToTrack(
                    #         frame_gray[y:y + h, x:x + w], mask=None, **feature_params)
                    #     if p0 is not None:
                    #         p0[:, 0, 0] += x
                    #         p0[:, 0, 1] += y

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # === 计算中心点坐标 ===
        center_x = x + w // 2
        center_y = y + h // 2
        center_text = f"Center: ({center_x}, {center_y})"

        # === 在画面右上角显示中心坐标 ===
        cv2.putText(
            img,
            center_text,
            (frame.shape[1] - 280, 30),  # 右上角（你也可以微调）
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),  # 黄色
            2
        )

        cv2.putText(
            img,
            "Press R to re-capture",
            (frame.shape[1] - 300, 60),  # 右上角（你也可以微调）
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),  # 黄色
            2
        )

        cv2.imshow('Drone Tracking', img)
        if save_video:
            out.write(img)  # === 保存当前帧到视频 ===
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC键退出
            break
        elif key == ord('r') or key == ord('R'):
            print("🔁 手动触发复位 → 重新寻找特征点")
            reset_flag = True
        else:
            reset_flag = False

    cap.release()
    cv2.destroyAllWindows()
    if save_video:
        out.release()

if __name__ == "__main__":
    main()
