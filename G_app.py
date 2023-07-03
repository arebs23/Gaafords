import streamlit as st
from PIL import Image
import os

def main():
    st.title('Image Processing App')

    # Get the list of images
    folder_path = '/home/aregbs/Desktop/gibson-afford/gen_data/data'
    images = os.listdir(folder_path)

    # Initialize session state
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0

    # Get the current image
    current_image_path = os.path.join(folder_path, images[st.session_state.current_image_index])
    image = Image.open(current_image_path)
        
    st.write('Grid Image :')

    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    # Display the same image in all 3 columns
    with col1:
        st.image(image, caption=f"Column 1: Image {st.session_state.current_image_index + 1}", use_column_width=True)
        if st.button("Caption matches image? - Column 1"):
            st.write("User said YES for Column 1.")

    with col2:
        st.image(image, caption=f"Column 2: Image {st.session_state.current_image_index + 1}", use_column_width=True)
        if st.button("Caption matches image? - Column 2"):
            st.write("User said YES for Column 2.")

    with col3:
        st.image(image, caption=f"Column 3: Image {st.session_state.current_image_index + 1}", use_column_width=True)
        if st.button("Caption matches image? - Column 3"):
            st.write("User said YES for Column 3.")

    if st.button("Next"):
        # Increment image index and rerun the script
        if st.session_state.current_image_index < len(images) - 1:
            st.session_state.current_image_index += 1
        else:
            st.write("No more images to display.")

if __name__ == "__main__":
    main()
