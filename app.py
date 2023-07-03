import streamlit as st
from PIL import Image

def main():
    st.title('Image Processing App')

    image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
    
    if image_file is not None:
        image = Image.open(image_file)
        
        st.write('Original Image :')
        st.image(image, use_column_width=True)

        st.write('Grid Image :')

        # Create a 3-column layout
        col1, col2, col3 = st.columns(3)
        
        # Display the same image in all 3 columns
        with col1:
            st.image(image, caption="Column 1", use_column_width=True)

        with col2:
            st.image(image, caption="Column 2", use_column_width=True)

        with col3:
            st.image(image, caption="Column 3", use_column_width=True)

if __name__ == "__main__":
    main()
