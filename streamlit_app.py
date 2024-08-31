import streamlit as st
from PIL import Image
from src.traditional_methods import apply_traditional_editing

st.title('🎈 The Dolly App')

st.info('This app is a simple demonstration of Dolly Bansal')


# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', width=400)

    # Enter scale factor at the top
    scale_factor = st.number_input('Enter scale factor:', min_value=1.0, max_value=10.0, value=2.0)

    # List to store the selected methods
    if 'selected_methods' not in st.session_state:
        st.session_state.selected_methods = []

    # Combine all methods into a single radio button group
    methods = [
        'Bilinear Interpolation', 'Bicubic Interpolation', 'Lanczos Resampling',
        'Nearest-Neighbor Interpolation', 'Gaussian Filtering', 'Median Filtering',
        'Unsharp Masking', 'Fourier-Based Methods', 'Wavelet-Based Methods', 
        'Spline Interpolation'
    ]
    
    # Display radio buttons to select a method
    selected_method = st.radio("Select a method:", methods)

    # Button to add the selected method to the list
    if st.button('Add Method'):
        if selected_method and selected_method not in st.session_state.selected_methods:
            st.session_state.selected_methods.append(selected_method)

    # Button to delete the last method from the list
    if st.button('Delete Last Item'):
        if st.session_state.selected_methods:
            st.session_state.selected_methods.pop()

    # Display the selected methods in typewriter format with yellow background
    if st.session_state.selected_methods:
        st.write("### Selected Methods:")
        for i, method in enumerate(st.session_state.selected_methods):
            st.markdown(
                f"<div style='background-color: brown; color: white; font-family: Courier, monospace; padding: 5px;'>{i}: {method}</div>",
                unsafe_allow_html=True
            )

    # Button to apply the selected methods to the image
    if st.button('Apply Selected Methods'):
        if st.session_state.selected_methods:
            edited_image = apply_traditional_editing(image, st.session_state.selected_methods, scale_factor)

            # Use containers to display images side by side
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Original Image', use_column_width=True)
                with col2:
                    st.image(edited_image, caption='Traditionally Edited Image', use_column_width=True)
        else:
            st.warning("Please select at least one filtering method.")
else:
    st.write("Please upload an image file.")