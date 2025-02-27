import streamlit as st
try:
    from streamlit.runtime.scriptrunner import RerunException, RerunData
except ImportError:
    RerunException = None
from shape_art_generator import main_page as shape_art_generator_page
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from EnDe import decode, encode
from painterfun import oil_main  # Importing the oil_main function
import mixbox
import itertools
import math
import os
from pathlib import Path

# Set page config for the merged app.
st.set_page_config(page_title="Merged App", layout="wide")

##############################################################################
# --- Utility Functions (Image Generator, Shape Detector, Oil Painting, etc.)
##############################################################################

def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def group_similar_colors(rgb_vals, threshold=10):
    grouped_colors = []  # List to store groups of similar colors
    counts = []  # List to store counts for each group

    for color in rgb_vals:
        found_group = False
        for i, group in enumerate(grouped_colors):
            if color_distance(color, group[0]) < threshold:
                grouped_colors[i].append(color)
                counts[i] += 1
                found_group = True
                break
        if not found_group:
            grouped_colors.append([color])
            counts.append(1)
    return [(group[0], count) for group, count in zip(grouped_colors, counts)]

def oil_painting_page():
    st.title("Oil Painting Image Generator")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    intensity = st.number_input("Enter the intensity (integer):", min_value=1, max_value=100, value=10)
    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.write("Upload an image to see it here")
    if st.button("Generate"):
        if uploaded_file is not None:
            input_image_cv = np.array(input_image)
            if len(input_image_cv.shape) == 2:
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_GRAY2RGB)
            elif input_image_cv.shape[2] == 4:
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_RGBA2RGB)
            output_image_cv = oil_main(input_image_cv, intensity)
            output_image_cv = (output_image_cv * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image_cv)
            with col2:
                st.image(output_image, caption="Processed Image", use_container_width=True)
            img_byte_arr = BytesIO()
            output_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            st.download_button(
                label="Download Processed Image",
                data=img_byte_arr,
                file_name="processed_image.png",
                mime="image/png"
            )

def color_mixing_app():
    st.title("RGB Color Mixing")
    if 'colors' not in st.session_state:
        st.session_state.colors = [
            {"rgb": [255, 0, 0], "weight": 0.3},  # Red
            {"rgb": [0, 255, 0], "weight": 0.6}   # Green
        ]

    def rgb_to_latent(rgb):
        return mixbox.rgb_to_latent(rgb)

    def latent_to_rgb(latent):
        return mixbox.latent_to_rgb(latent)

    def get_mixed_rgb(colors):
        z_mix = [0] * mixbox.LATENT_SIZE
        total_weight = sum(c["weight"] for c in colors)
        for i in range(len(z_mix)):
            z_mix[i] = sum(c["weight"] * rgb_to_latent(c["rgb"])[i] for c in colors) / total_weight
        return latent_to_rgb(z_mix)

    def add_new_color():
        st.session_state.colors.append({"rgb": [255, 255, 255], "weight": 0.1})

    def delete_color(index):
        st.session_state.colors.pop(index)

    for idx, color in enumerate(st.session_state.colors):
        with st.expander(f"Color {idx + 1}"):
            r = st.number_input(f"Red value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][0])
            g = st.number_input(f"Green value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][1])
            b = st.number_input(f"Blue value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][2])
            col_weight = st.slider(f"Weight for Color {idx + 1}", 0.0, 1.0, value=color['weight'], step=0.05)
            color["rgb"] = [r, g, b]
            color["weight"] = col_weight
            st.markdown(f"<div style='width: 100px; height: 100px; background-color: rgb({r}, {g}, {b});'></div>", unsafe_allow_html=True)
            if len(st.session_state.colors) > 2:
                delete_button = st.button(f"Delete Color {idx + 1}", key=f"delete_button_{idx}")
                if delete_button:
                    delete_color(idx)
                    st.session_state.colors = st.session_state.colors
    if st.button("Add Color"):
        add_new_color()
        st.session_state.colors = st.session_state.colors

    mixed_rgb = get_mixed_rgb(st.session_state.colors)
    st.subheader("Mixed Color")
    st.write(f"RGB: {mixed_rgb}")
    st.markdown(f"<div style='width: 200px; height: 200px; background-color: rgb{mixed_rgb};'></div>", unsafe_allow_html=True)

def image_generator_app():
    st.header("Image Generator")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])
    num_shapes = st.number_input("Enter the number of shapes to encode:", min_value=1, value=100)
    if shape_option == "Triangle":
        max_triangle_size = st.number_input("Enter the maximum triangle size:", min_value=1, value=50)
        min_triangle_size = st.number_input("Enter the minimum triangle size (for filling gaps):", min_value=1, value=15)
    elif shape_option in ["Rectangle", "Circle"]:
        min_size = st.number_input("Enter the minimum size of the shape:", min_value=1, value=10)
        max_size = st.number_input("Enter the maximum size of the shape:", min_value=1, value=15)
    else:
        shape_size = st.number_input("Enter the size of the shape:", min_value=1, value=10)
    col1, col2 = st.columns([1, 1])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error reading the image. Please try another file.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with col1:
                st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate"):
        if uploaded_file is not None:
            shape = shape_option
            if shape_option == "Triangle":
                encoded_image, boundaries = encode(img, shape, output_path="",
                                                   num_shapes=num_shapes,
                                                   max_size=max_triangle_size,
                                                   min_size=min_triangle_size)
            elif shape_option in ["Rectangle", "Circle"]:
                encoded_image, boundaries = encode(img, shape, output_path="",
                                                   num_shapes=num_shapes, min_size=min_size, max_size=max_size,
                                                   min_radius=min_size, max_radius=max_size)
            else:
                encoded_image, boundaries = encode(img, shape, output_path="",
                                                   num_shapes=num_shapes, shape_size=shape_size)
            encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(encoded_image_rgb, caption=f"Encoded {shape_option} Image", use_container_width=True)
            is_success, buffer = cv2.imencode(".png", encoded_image)
            if is_success:
                st.download_button(
                    label="Download Encoded Image",
                    data=buffer.tobytes(),
                    file_name="encoded_image.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload an image first.")

def shape_detector_app():
    st.header("Shape Detector")
    uploaded_file = st.file_uploader("Upload an Encoded Image", type=["jpg", "jpeg", "png"])
    shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])
    min_size_det = st.number_input("Enter the minimum size to detect:", min_value=1, value=3)
    max_size_det = st.number_input("Enter the maximum size to detect:", min_value=1, value=10)
    
    encoded_image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        encoded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if encoded_image is None:
            st.error("Error reading the image. Please try another file.")
    
    col1, col2 = st.columns(2)
    if uploaded_file is not None and encoded_image is not None:
        with col1:
            uploaded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
            st.image(uploaded_image_rgb, caption="Uploaded Encoded Image", use_container_width=True)
    
    if st.button("Decode"):
        if uploaded_file is not None and encoded_image is not None:
            shape = shape_option
            # Detect boundaries based on size limits.
            gray = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            detected_boundaries = []
            if shape == "Triangle":
                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    if len(approx) == 3:
                        tri = approx.reshape(-1, 2)
                        xs = tri[:, 0]
                        ys = tri[:, 1]
                        width = xs.max() - xs.min()
                        height = ys.max() - ys.min()
                        if width >= min_size_det and width <= max_size_det and height >= min_size_det and height <= max_size_det:
                            detected_boundaries.append(tri)
            elif shape == "Rectangle":
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w >= min_size_det and w <= max_size_det and h >= min_size_det and h <= max_size_det:
                        detected_boundaries.append((x, y, w, h))
            elif shape == "Circle":
                for cnt in contours:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    radius = int(radius)
                    if radius >= min_size_det and radius <= max_size_det:
                        detected_boundaries.append((int(x), int(y), radius))
            # Decode using detected boundaries.
            binary_img, annotated_img, rgb_vals = decode(encoded_image, shape, boundaries=detected_boundaries, max_size=max_size_det, min_size=min_size_det)
            annotated_image_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(annotated_image_rgb, caption=f"Decoded Annotated {shape_option} Image", use_container_width=True)
            
            grouped_colors = group_similar_colors(rgb_vals, threshold=10)
            grouped_colors = sorted(grouped_colors, key=lambda x: x[1], reverse=True)
            st.subheader("Grouped Colors (Ranked by Count)")
            # Display each grouped color along with a button.
            for idx, (color, count) in enumerate(grouped_colors):
                st.markdown(
                    f"<div style='display: flex; align-items: center;'>"
                    f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); "
                    f"width: 30px; height: 30px; border: 1px solid #000; margin-right: 10px;'></div>"
                    f"<div>{'RGB: ' + str(color) + ' - Count: ' + str(count)}</div>"
                    f"</div>", unsafe_allow_html=True)
                if st.button(f"Generate Recipe for RGB: {color}", key=f"color_{idx}"):
                    st.session_state.preselected_color = color
                    st.session_state.app_mode = "Color Recipe"
                    st.experimental_rerun()
        else:
            st.warning("Please upload an image first.")

def painter_recipe_generator():
    st.title("Painter App - Recipe Generator")
    st.write("Enter your desired paint color to generate paint recipes using base colors.")
    db_choice = st.selectbox("Select a color database:", list(databases.keys()))
    selected_db_dict = convert_db_list_to_dict(databases[db_choice])
    method = st.radio("Select input method:", ["Color Picker", "RGB Sliders"])
    if method == "Color Picker":
        desired_hex = st.color_picker("Pick a color", "#ffffff")
        desired_rgb = tuple(int(desired_hex[i:i+2], 16) for i in (1, 3, 5))
    else:
        st.write("Select RGB values manually:")
        r = st.slider("Red", 0, 255, 255)
        g = st.slider("Green", 0, 255, 255)
        b = st.slider("Blue", 0, 255, 255)
        desired_rgb = (r, g, b)
        desired_hex = rgb_to_hex(r, g, b)
    st.write("**Desired Color:**", desired_hex)
    display_color_block(desired_rgb, label="Desired")
    step = st.slider("Select percentage step for recipe generation:", 4.0, 10.0, 10.0, step=0.5)
    if st.button("Generate Recipes"):
        recipes = generate_recipes(desired_rgb, selected_db_dict, step=step)
        if recipes:
            st.write("### Top 3 Paint Recipes")
            for idx, (recipe, mixed, err) in enumerate(recipes):
                st.write(f"**Recipe {idx+1}:** (Error = {err:.2f})")
                cols = st.columns(4)
                with cols[0]:
                    st.write("Desired:")
                    display_color_block(desired_rgb, label="Desired")
                with cols[1]:
                    st.write("Result:")
                    display_color_block(mixed, label="Mixed")
                with cols[2]:
                    st.write("Composition:")
                    for name, perc in recipe:
                        if perc > 0:
                            base_rgb = tuple(selected_db_dict[name]["rgb"])
                            st.write(f"- **{name}**: {perc:.1f}%")
                            display_color_block(base_rgb, label=name)
                with cols[3]:
                    st.write("Difference:")
                    st.write(f"RGB Distance: {err:.2f}")
        else:
            st.error("No recipes found.")

def painter_colors_database():
    st.title("Colors DataBase")
    st.write("Select an action:")
    if "subpage" not in st.session_state:
        st.session_state.subpage = None
    row1_cols = st.columns(3)
    with row1_cols[0]:
        if st.button("Data Bases"):
            st.session_state.subpage = "databases"
    with row1_cols[1]:
        if st.button("Add Colors"):
            st.session_state.subpage = "add"
    with row1_cols[2]:
        if st.button("Remove Colors"):
            st.session_state.subpage = "remove_colors"
    row2_cols = st.columns(3)
    with row2_cols[0]:
        if st.button("Create Custom Data Base"):
            st.session_state.subpage = "custom"
    with row2_cols[1]:
        if st.button("Remove Database"):
            st.session_state.subpage = "remove_database"
    with row2_cols[2]:
        st.write("")
    if st.session_state.subpage == "databases":
        show_databases_page()
    elif st.session_state.subpage == "add":
        show_add_colors_page()
    elif st.session_state.subpage == "remove_colors":
        show_remove_colors_page()
    elif st.session_state.subpage == "custom":
        show_create_custom_db_page()
    elif st.session_state.subpage == "remove_database":
        show_remove_database_page()

##############################################################################
# --- NEW: Color Recipe Page (Preselected RGB from Shape Detector)
##############################################################################
def color_recipe_app():
    st.title("Color Recipe Generator")
    # Retrieve the preselected color from session state (default to white)
    preselected_color = st.session_state.get("preselected_color", [255, 255, 255])
    st.write(f"Preselected Color: RGB{tuple(preselected_color)}")
    color_box_html = f"<div style='background-color: rgb({preselected_color[0]}, {preselected_color[1]}, {preselected_color[2]}); width:200px; height:200px; border:1px solid #000;'></div>"
    st.markdown(color_box_html, unsafe_allow_html=True)
    
    # Allow database selection and slider (step between 4 and 10)
    db_choice = st.selectbox("Select a color database:", list(databases.keys()))
    step = st.slider("Select percentage step for recipe generation:", 4.0, 10.0, 10.0, step=0.5)
    
    if st.button("Generate Recipes"):
        selected_db_dict = convert_db_list_to_dict(databases[db_choice])
        recipes = generate_recipes(tuple(preselected_color), selected_db_dict, step=step)
        if recipes:
            st.write("### Top 3 Paint Recipes")
            for idx, (recipe, mixed, err) in enumerate(recipes):
                st.write(f"**Recipe {idx+1}:** (Error = {err:.2f})")
                cols = st.columns(4)
                with cols[0]:
                    st.write("Desired:")
                    display_color_block(tuple(preselected_color), label="Desired")
                with cols[1]:
                    st.write("Result:")
                    display_color_block(mixed, label="Mixed")
                with cols[2]:
                    st.write("Composition:")
                    for name, perc in recipe:
                        if perc > 0:
                            base_rgb = tuple(selected_db_dict[name]["rgb"])
                            st.write(f"- **{name}**: {perc:.1f}%")
                            display_color_block(base_rgb, label=name)
                with cols[3]:
                    st.write("Difference:")
                    st.write(f"RGB Distance: {err:.2f}")
        else:
            st.error("No recipes found.")

##############################################################################
# --- Main Navigation wrapped in a main() function
##############################################################################
def main():
    # Ensure app_mode exists in session state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Image Generator"

    app_mode = st.sidebar.radio("Select Mode", [
        "Image Generator", 
        "Shape Detector", 
        "Oil Painting Generator", 
        "Colour Merger", 
        "Recipe Generator", 
        "Colors DataBase",
        "Foogle Man Repo",
        "Color Recipe"  # New mode for Color Recipe page
    ], key="app_mode")

    if st.sidebar.button("Refresh App"):
        read_color_file.clear()  # Clear cached data.
        if RerunException is not None:
            raise RerunException(RerunData())  # Force a rerun.
        else:
            st.warning("Automatic refresh is not supported. Please reload your browser.")

    if app_mode == "Image Generator":
        image_generator_app()
    elif app_mode == "Shape Detector":
        shape_detector_app()
    elif app_mode == "Oil Painting Generator":
        oil_painting_page()
    elif app_mode == "Colour Merger":
        color_mixing_app()
    elif app_mode == "Recipe Generator":
        painter_recipe_generator()
    elif app_mode == "Colors DataBase":
        painter_colors_database()
    elif app_mode == "Foogle Man Repo":
        shape_art_generator_page()
    elif app_mode == "Color Recipe":
        color_recipe_app()

if __name__ == "__main__":
    main()
