import streamlit as st
import subprocess
import os
import sys
import math
import random
import numpy as np
from PIL import Image, ImageDraw
from numba import njit
import io

# ------------------------------------------------------------
# Utility Functions and Geometrize Algorithm
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
# Main App Layout
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
        
        # Run Oil-Painting pipeline if selected
        if oil_option:
            os.makedirs("uploads", exist_ok=True)
            # Resize input image to 256x256 for oil painting
            resized_img = input_img.resize((256, 256))
            file_path = os.path.join("uploads", uploaded_file.name)
            resized_img.save(file_path)
            command = [sys.executable, "Oil-Painting.py", "--f", file_path, "--p", str(p_value)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_dir = os.path.join("output", f"{base_name}-p-{p_value}")
                final_image_path = os.path.join(output_dir, "Final_Result.png")
                if os.path.exists(final_image_path):
                    oil_img = Image.open(final_image_path)
                    results["Oil-Painting"] = oil_img
                else:
                    st.error("Oil-Painting processed image not found.")
            else:
                st.error("Error in Oil-Painting process:")
                st.text(result.stderr)
        
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
