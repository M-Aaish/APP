import streamlit as st
import os
import sys
import math
import random
import numpy as np
from PIL import Image, ImageDraw
from numba import njit
import io
import cv2
import time

# ------------------------------------------------------------
# Utility Functions (Geometrize) 
# ------------------------------------------------------------

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

@njit
def image_difference_numba(arrA, arrB):
    diff = 0
    for i in range(arrA.shape[0]):
        for j in range(arrA.shape[1]):
            for k in range(4):  # RGBA channels
                d = int(arrA[i, j, k]) - int(arrB[i, j, k])
                diff += d * d
    return diff

def image_difference(imgA, imgB):
    arrA = np.array(imgA, dtype=np.uint8)
    arrB = np.array(imgB, dtype=np.uint8)
    return image_difference_numba(arrA, arrB)

def blend_image(base_img, shape_img):
    return Image.alpha_composite(base_img, shape_img)

def get_image_array(img):
    return np.array(img, dtype=np.uint8)

# ----- Shape Classes -----
class BaseShape:
    def __init__(self):
        self.color = (255, 0, 0, 128)
    
    def randomize_color(self):
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(20, 200)
        )
    
    def copy(self):
        raise NotImplementedError("copy() not implemented")
    
    def randomize(self, width, height):
        raise NotImplementedError("randomize() not implemented")
    
    def mutate(self, width, height, amount=1.0):
        raise NotImplementedError("mutate() not implemented")
    
    def rasterize(self, width, height):
        raise NotImplementedError("rasterize() not implemented")

class TriangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.points = [(0,0), (0,0), (0,0)]
    
    def copy(self):
        new_shape = TriangleShape()
        new_shape.color = self.color
        new_shape.points = list(self.points)
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.points = [
            (random.randint(0, width-1), random.randint(0, height-1)),
            (random.randint(0, width-1), random.randint(0, height-1)),
            (random.randint(0, width-1), random.randint(0, height-1))
        ]
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15,15), 0, 255)
            g = clamp(g + random.randint(-15,15), 0, 255)
            b = clamp(b + random.randint(-15,15), 0, 255)
            a = clamp(a + random.randint(-15,15), 20, 255)
            self.color = (r, g, b, a)
        new_points = []
        for (x,y) in self.points:
            if random.random() < 0.5:
                x = clamp(x + int(random.randint(-5,5)*amount), 0, width-1)
            if random.random() < 0.5:
                y = clamp(y + int(random.randint(-5,5)*amount), 0, height-1)
            new_points.append((x,y))
        self.points = new_points
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.polygon(self.points, fill=self.color)
        return img

class RectangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.x1 = self.y1 = self.x2 = self.y2 = 0
    
    def copy(self):
        new_shape = RectangleShape()
        new_shape.color = self.color
        new_shape.x1, new_shape.y1, new_shape.x2, new_shape.y2 = self.x1, self.y1, self.x2, self.y2
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.x1 = random.randint(0, width-1)
        self.y1 = random.randint(0, height-1)
        self.x2 = clamp(self.x1 + random.randint(-50,50), 0, width-1)
        self.y2 = clamp(self.y1 + random.randint(-50,50), 0, height-1)
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15,15), 0, 255)
            g = clamp(g + random.randint(-15,15), 0, 255)
            b = clamp(b + random.randint(-15,15), 0, 255)
            a = clamp(a + random.randint(-15,15), 20, 255)
            self.color = (r, g, b, a)
        if random.random() < 0.5:
            self.x1 = clamp(self.x1 + int(random.randint(-5,5)*amount), 0, width-1)
        if random.random() < 0.5:
            self.y1 = clamp(self.y1 + int(random.randint(-5,5)*amount), 0, height-1)
        if random.random() < 0.5:
            self.x2 = clamp(self.x2 + int(random.randint(-5,5)*amount), 0, width-1)
        if random.random() < 0.5:
            self.y2 = clamp(self.y2 + int(random.randint(-5,5)*amount), 0, height-1)
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(img, 'RGBA')
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        draw.rectangle([x1, y1, x2, y2], fill=self.color)
        return img

class EllipseShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.x1 = self.y1 = self.x2 = self.y2 = 0
    
    def copy(self):
        new_shape = EllipseShape()
        new_shape.color = self.color
        new_shape.x1, new_shape.y1, new_shape.x2, new_shape.y2 = self.x1, self.y1, self.x2, self.y2
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.x1 = random.randint(0, width-1)
        self.y1 = random.randint(0, height-1)
        self.x2 = clamp(self.x1 + random.randint(-50,50), 0, width-1)
        self.y2 = clamp(self.y1 + random.randint(-50,50), 0, height-1)
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15,15), 0, 255)
            g = clamp(g + random.randint(-15,15), 0, 255)
            b = clamp(b + random.randint(-15,15), 0, 255)
            a = clamp(a + random.randint(-15,15), 20, 255)
            self.color = (r, g, b, a)
        if random.random() < 0.5:
            self.x1 = clamp(self.x1 + int(random.randint(-5,5)*amount), 0, width-1)
        if random.random() < 0.5:
            self.y1 = clamp(self.y1 + int(random.randint(-5,5)*amount), 0, height-1)
        if random.random() < 0.5:
            self.x2 = clamp(self.x2 + int(random.randint(-5,5)*amount), 0, width-1)
        if random.random() < 0.5:
            self.y2 = clamp(self.y2 + int(random.randint(-5,5)*amount), 0, height-1)
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(img, 'RGBA')
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        draw.ellipse([x1, y1, x2, y2], fill=self.color)
        return img

def create_shape(shape_type):
    if shape_type == 'triangle':
        return TriangleShape()
    elif shape_type == 'rectangle':
        return RectangleShape()
    elif shape_type == 'ellipse':
        return EllipseShape()
    else:
        raise ValueError("Unknown shape type: {}".format(shape_type))

def simulated_annealing_shape(base_img, target_img, shape, iterations, start_temp, end_temp, step_scale=1.0):
    width, height = target_img.size
    current_shape = shape.copy()
    shape_img = current_shape.rasterize(width, height)
    blended = blend_image(base_img, shape_img)
    current_diff = image_difference(target_img, blended)
    best_shape = current_shape.copy()
    best_diff = current_diff
    for i in range(iterations):
        T = start_temp * ((end_temp / start_temp) ** (i / iterations))
        new_shape = current_shape.copy()
        new_shape.mutate(width, height, amount=step_scale)
        shape_img = new_shape.rasterize(width, height)
        candidate = blend_image(base_img, shape_img)
        diff = image_difference(target_img, candidate)
        delta = diff - current_diff
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_shape = new_shape
            current_diff = diff
            if diff < best_diff:
                best_shape = new_shape.copy()
                best_diff = diff
    return best_shape, best_diff

def refine_shape(base_img, target_img, shape, coarse_iter, fine_iter, coarse_start_temp, coarse_end_temp, fine_start_temp, fine_end_temp):
    best_shape, best_diff = simulated_annealing_shape(base_img, target_img, shape,
                                                       iterations=coarse_iter,
                                                       start_temp=coarse_start_temp,
                                                       end_temp=coarse_end_temp,
                                                       step_scale=1.0)
    best_shape, best_diff = simulated_annealing_shape(base_img, target_img, best_shape,
                                                       iterations=fine_iter,
                                                       start_temp=fine_start_temp,
                                                       end_temp=fine_end_temp,
                                                       step_scale=0.5)
    return best_shape, best_diff

def run_geometrize(target_img, shape_type, shape_count, resize_factor,
                   coarse_iterations=1000, fine_iterations=500,
                   coarse_start_temp=100.0, coarse_end_temp=10.0,
                   fine_start_temp=10.0, fine_end_temp=1.0):
    target_img = target_img.convert("RGBA")
    orig_w, orig_h = target_img.size
    if resize_factor != 1.0:
        new_w = int(orig_w * resize_factor)
        new_h = int(orig_h * resize_factor)
        target_img = target_img.resize((new_w, new_h), Image.LANCZOS)
    width, height = target_img.size
    current_img = Image.new("RGBA", (width, height), (255,255,255,255))
    current_diff = image_difference(target_img, current_img)
    img_placeholder = st.empty()
    progress_placeholder = st.empty()
    for i in range(shape_count):
        shape = create_shape(shape_type)
        shape.randomize(width, height)
        best_shape, best_diff = refine_shape(
            base_img=current_img,
            target_img=target_img,
            shape=shape,
            coarse_iter=coarse_iterations,
            fine_iter=fine_iterations,
            coarse_start_temp=coarse_start_temp,
            coarse_end_temp=coarse_end_temp,
            fine_start_temp=fine_start_temp,
            fine_end_temp=fine_end_temp
        )
        if best_diff < current_diff:
            shape_img = best_shape.rasterize(width, height)
            current_img = blend_image(current_img, shape_img)
            current_diff = best_diff
        img_placeholder.image(np.array(current_img), width=350)
        progress_placeholder.text(f"Shape count: {i+1}/{shape_count}")
    return current_img


# ------------------------------------------------------------
# BELOW: MERGED CODE FROM "Oil-Painting.py"
# ------------------------------------------------------------
# We define a function that does what your Oil-Painting.py main code did,
# but as a direct call. We omit argparse and read parameters from arguments.

def run_oil_painting_merged(
    input_image_path: str,
    brush_path: str = "./brush/brush-0.png",
    p_value: int = 4,
    seed: int = 0,
    force: bool = True,
    SSAA: int = 4,
    freq: int = 100,
    stroke_order_type: int = 0,
    output_root: str = "./output"
):
    """
    Merged stroke-based oil painting rendering logic from Oil-Painting.py.
    :param input_image_path: path to the input image
    :param brush_path: path to the brush template
    :param p_value: reciprocal of the maximum sampling rate (1/p)
    :param seed: random seed
    :param force: force recomputation of the anchor Map
    :param SSAA: super-sampling anti-aliasing factor
    :param freq: save one frame every (freq) strokes
    :param stroke_order_type: 0 for default size order, 1 for random order
    :param output_root: top-level output directory
    :return: path to the final result image
    """

    # 1) Prepare default config
    default = {
        "image": input_image_path,
        "brush": brush_path,
        "p_max": p_value,       # the reciprocal of the max sampling rate
        "seed": seed,
        "force": force,
        "SSAA" : SSAA,
        "freq" : freq,
        "stroke_order_type": stroke_order_type,
        # fixed from the original code
        "padding": 5,
        "n_iter": 15,
        "k_size": 5,
        "figsize": 6,
        "pointsize": (8.0, 8.0),
        "ratio": 3,
        "threshold_hsv": (30,None,15),
        "kernel_radius": 5,
        "ETF_iter": 15,
        "background_dir": None
    }

    # 2) Re-implement logic from main:
    import argparse

    # The original code used argparse; we'll replicate that logic here:
    # We'll treat p_value as the integer from the slider or default 4
    # and build other parameters accordingly.
    p_max = 1.0 / default["p_max"]
    p_min = p_max / 100
    ratio = default["ratio"]
    max_width = np.sqrt(1 / p_min)
    min_width = np.sqrt(1 / p_max) - 1
    max_length = int(ratio * max_width)
    min_length = ratio * min_width
    padding = default["padding"]
    kernel_radius = default["kernel_radius"]

    # Create output directory
    filename = os.path.basename(default["image"])
    filename = filename.split('.')[0]
    output_path = os.path.join(output_root, filename + f"-p-{default['p_max']}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + "/anchor")
        os.makedirs(output_path + "/stroke")
        os.makedirs(output_path + "/process")

    # Import the modules from the original script:
    # (simulate_RGB, drawpatch, ETF, quicksort, voronoi_sampler, search_and_render)
    # We'll assume you have these modules in the same folder or accessible in your PYTHONPATH.
    from simulate_RGB import Gassian_HSV
    from ETF.edge_tangent_flow import ETF
    from quicksort import quickSort
    from voronoi_sampler import K_Means_Sampler
    from search_and_render import Search_Stroke, Render_Stroke

    # 3) K-Means Sampler
    np.random.seed(default["seed"])
    point_num, density, gradient_magnitude, point_path = K_Means_Sampler(
        output_dir=output_path+"/anchor",
        filename=default["image"],
        p_max=p_max,
        p_min=p_min,
        border_copy=padding,
        k_size=default["k_size"],
        n_iter=default["n_iter"],
        figsize=default["figsize"],
        pointsize=default["pointsize"],
        display=False,
        force=default["force"],
        save=True
    )

    # 4) Load & prep input image
    input_bgr = cv2.imread(default["image"], cv2.IMREAD_COLOR)
    # Optionally resize to 256x256 if you want to replicate the old approach
    # but the original code doesn't do that. If needed, do:
    # input_bgr = cv2.resize(input_bgr, (256, 256))
    cv2.imwrite(output_path + "/input_bgr.png", input_bgr)
    input_bgr = cv2.copyMakeBorder(input_bgr, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)
    input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    (H0,W0) = input_gray.shape

    # 5) ETF
    time_start=time.time()
    ETF_filter = ETF(
        img=input_gray,
        output_path=output_path+'/mask',
        kernel_radius=kernel_radius,
        iter_time=default["ETF_iter"],
        background_dir=default["background_dir"]
    )
    angle = ETF_filter.forward().numpy()
    angle_hatch = angle + 90
    angle_hatch[angle_hatch>90] -= 180
    # print('ETF Filtering time:', int(time.time()-time_start),"seconds")

    # 6) Search patch
    time_start=time.time()
    points = np.load(point_path)
    patch_sequence = Search_Stroke(
        points, density, input_gray, input_hsv, gradient_magnitude, ratio,
        angle, angle_hatch, min_width, min_length, max_width, max_length, default["threshold_hsv"]
    )

    # 7) Stroke Order
    if default["stroke_order_type"] == 1:
        random.shuffle(patch_sequence)
    elif default["stroke_order_type"] == 0:
        random.shuffle(patch_sequence)
        quickSort(patch_sequence,0,len(patch_sequence)-1)

    # 8) Render Stroke
    SSAA = default["SSAA"]
    freq = default["freq"]
    brush = cv2.imread(default["brush"], cv2.IMREAD_GRAYSCALE)

    wihte = Gassian_HSV((H0*SSAA-2*padding*SSAA, W0*SSAA-2*padding*SSAA, 3))
    cv2.imwrite(output_path + "/process/{0:04d}.png".format(0), cv2.cvtColor(wihte, cv2.COLOR_HSV2BGR))
    Canvas, Mask = Render_Stroke(brush, patch_sequence, input_gray, output_path, max_length,
                                 SSAA=SSAA, BORDERCOPY=padding, FREQ=freq, save=True)

    result = Canvas[max_length*SSAA:-max_length*SSAA, max_length*SSAA:-max_length*SSAA]
    cv2.imwrite(output_path + "/Oil_drawing.png", cv2.cvtColor(result, cv2.COLOR_HSV2BGR))

    # 9) Pad Blank Area
    mask = Mask
    mask[max_length*SSAA+padding*SSAA-1,:] = 1
    mask[-max_length*SSAA-padding*SSAA,:] = 1
    mask[:, max_length*SSAA+padding*SSAA-1] = 1
    mask[:,-max_length*SSAA-padding*SSAA] = 1

    while True:
        result = cv2.imread(output_path + "/Oil_drawing.png", cv2.IMREAD_COLOR)
        mask_cut = mask[max_length*SSAA:-max_length*SSAA, max_length*SSAA:-max_length*SSAA]
        cv2.imwrite(output_path + "/mask.png", mask_cut.astype("uint8")*255)
        connect_num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - mask_cut.astype("uint8")*255, connectivity=8
        )
        Points = []
        for i in range(centroids.shape[0]):
            p = centroids[i]
            if (p[0] >= padding*SSAA and p[1] >= padding*SSAA and 
                p[0] < result.shape[1]-padding*SSAA and 
                p[1] < result.shape[0]-padding*SSAA and 
                stats[i][4]>0 and 
                stats[i][4]<result.shape[0]*result.shape[1]/4):
                p[0], p[1] = p[0]/SSAA, p[1]/SSAA
                Points.append([p[0], result.shape[0]/SSAA - p[1]])
        Points = np.array(Points)
        if Points.shape[0] == 0:
            final_path = output_path + "/Final_Result.png"
            cv2.imwrite(final_path, result[padding*SSAA:-padding*SSAA, padding*SSAA:-padding*SSAA])
            cv2.imwrite(output_path + "/process/Final_Result.png",
                        result[padding*SSAA:-padding*SSAA, padding*SSAA:-padding*SSAA])
            # Return the path to final result
            return final_path
        else:
            for point in Points:
                cv2.circle(result,
                           (int(np.around(point[0]*SSAA)),
                            int(np.around((result.shape[0]/SSAA-point[1])*SSAA))),
                           3, (0,0,255), 3)
            cv2.imwrite(output_path + "/anchor.png", result)

            pad_sequence = Search_Stroke(
                np.array(Points), density, input_gray, input_hsv, gradient_magnitude, ratio,
                angle, angle_hatch, min_width, min_length, max_width, max_length, default["threshold_hsv"]
            )
            if default["stroke_order_type"] == 0:
                quickSort(pad_sequence,0,len(pad_sequence)-1)

            pad_canvas, pad_mask = Render_Stroke(
                brush, pad_sequence, input_gray, output_path, max_length,
                SSAA=SSAA, BORDERCOPY=padding, FREQ=freq, save=False
            )
            pad_canvas_cut = pad_canvas[max_length*SSAA:-max_length*SSAA, max_length*SSAA:-max_length*SSAA]
            pad_canvas_cut = cv2.cvtColor(pad_canvas_cut, cv2.COLOR_HSV2BGR)
            for point in Points:
                cv2.circle(pad_canvas_cut,
                           (int(np.around(point[0]*SSAA)),
                            int(np.around((result.shape[0]/SSAA-point[1])*SSAA))),
                           3, (0,0,255), 3)
            cv2.imwrite(output_path + "/pad_canvas_cut.png", pad_canvas_cut)

            Oil_drawing = cv2.imread(output_path + "/Oil_drawing.png", cv2.IMREAD_COLOR)
            Oil_drawing = cv2.cvtColor(Oil_drawing, cv2.COLOR_BGR2HSV)
            m = pad_mask*(1-mask)
            m = m[max_length*SSAA:-max_length*SSAA, max_length*SSAA:-max_length*SSAA, np.newaxis]
            Oil_drawing = np.uint8(m*pad_canvas_cut + (1-m)*Oil_drawing)
            Oil_drawing = cv2.cvtColor(Oil_drawing, cv2.COLOR_HSV2BGR)
            cv2.imwrite(output_path + "/Oil_drawing.png", Oil_drawing)

            mask += pad_mask
            mask[mask>0] = 1

        min_length += ratio
        min_width += 1

# ------------------------------------------------------------
# Main App Layout (Geometrize UI + Oil Painting Option)
# ------------------------------------------------------------

st.title("Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the uploaded image once
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="Input Image", use_column_width=True)
    
    # Checkboxes to select processing pipelines
    oil_option = st.checkbox("Run Oil-Painting")
    geom_option = st.checkbox("Run Geometrize")
    
    # Show parameter widgets for each option
    if oil_option:
        st.subheader("Oil-Painting Options")
        p_value = st.slider("Select parameter (--p)", min_value=1, max_value=10, value=4)
    if geom_option:
        st.subheader("Geometrize Options")
        shape_type = st.selectbox("Select shape type", ("triangle", "rectangle", "ellipse"))
        shape_count = st.number_input("Number of shapes", min_value=1, value=300, step=1)
        resize_factor = st.slider("Resize factor (for Geometrize)", 0.25, 0.5, 0.5, step=0.01)
    
    if st.button("Process Image"):
        results = {}
        # Convert the uploaded file to a local path
        # Save the user-uploaded image so we have a real file path for oil painting code
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        input_img.save(file_path)

        # Run Oil-Painting pipeline if selected
        if oil_option:
            # Instead of calling a separate script, call the merged function directly:
            final_oil_path = run_oil_painting_merged(
                input_image_path=file_path,
                brush_path="./brush/brush-0.png",  # Or any brush you want
                p_value=p_value,
                seed=0,
                force=True,
                SSAA=4,
                freq=100,
                stroke_order_type=0,
                output_root="./output"
            )
            # If it succeeded, load the result:
            if os.path.exists(final_oil_path):
                oil_img = Image.open(final_oil_path)
                results["Oil-Painting"] = oil_img
            else:
                st.error("Oil-Painting processed image not found.")

        # Run Geometrize pipeline if selected
        if geom_option:
            geom_img = run_geometrize(input_img, shape_type, shape_count, resize_factor)
            results["Geometrize"] = geom_img
        
        # Display results: side-by-side if both are processed, otherwise singly.
        if results:
            if len(results) == 2:
                col1, col2 = st.columns(2)
                if "Oil-Painting" in results:
                    col1.image(results["Oil-Painting"], caption="Oil-Painting Result", use_column_width=True)
                if "Geometrize" in results:
                    col2.image(results["Geometrize"], caption="Geometrize Result", use_column_width=True)
            else:
                for key, img in results.items():
                    st.image(img, caption=f"{key} Result", use_column_width=True)

def geometrize_app():
    """
    Alternative function for multi-page usage. 
    If you want to import geometrize_app into another Streamlit file,
    you can call `geometrize_app()` from there. It has a unique key for the file uploader.
    """
    st.title("Image Processing App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="geometrize_upload")
    if uploaded_file is not None:
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption="Input Image", use_column_width=True)
        
        oil_option = st.checkbox("Run Oil-Painting")
        geom_option = st.checkbox("Run Geometrize")
        
        if oil_option:
            st.subheader("Oil-Painting Options")
            p_value = st.slider("Select parameter (--p)", min_value=1, max_value=10, value=4)
        if geom_option:
            st.subheader("Geometrize Options")
            shape_type = st.selectbox("Select shape type", ("triangle", "rectangle", "ellipse"))
            shape_count = st.number_input("Number of shapes", min_value=1, value=300, step=1)
            resize_factor = st.slider("Resize factor (for Geometrize)", 0.25, 0.5, 0.5, step=0.01)
        
        if st.button("Process Image", key="geom_process_btn"):
            results = {}
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", uploaded_file.name)
            input_img.save(file_path)

            # Run Oil-Painting pipeline if selected
            if oil_option:
                final_oil_path = run_oil_painting_merged(
                    input_image_path=file_path,
                    brush_path="./brush/brush-0.png",
                    p_value=p_value,
                    seed=0,
                    force=True,
                    SSAA=4,
                    freq=100,
                    stroke_order_type=0,
                    output_root="./output"
                )
                if os.path.exists(final_oil_path):
                    oil_img = Image.open(final_oil_path)
                    results["Oil-Painting"] = oil_img
                else:
                    st.error("Oil-Painting processed image not found.")
            
            # Run Geometrize pipeline if selected
            if geom_option:
                geom_img = run_geometrize(input_img, shape_type, shape_count, resize_factor)
                results["Geometrize"] = geom_img
            
            if results:
                if len(results) == 2:
                    col1, col2 = st.columns(2)
                    if "Oil-Painting" in results:
                        col1.image(results["Oil-Painting"], caption="Oil-Painting Result", use_column_width=True)
                    if "Geometrize" in results:
                        col2.image(results["Geometrize"], caption="Geometrize Result", use_column_width=True)
                else:
                    for key, img in results.items():
                        st.image(img, caption=f"{key} Result", use_column_width=True)

if __name__ == "__main__":
    # If you run this file directly (not as an import), it will use the top-level UI
    geometrize_app()
