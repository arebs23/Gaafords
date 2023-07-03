import streamlit as st
from PIL import Image
import os
import pandas as pd

def main():
    st.title('Image Processing App')

    # Get the list of images
    folder_path = '/home/aregbs/Desktop/gibson-afford/gen_data/data'
    images = os.listdir(folder_path)

    # Initialize session state
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'yes_score' not in st.session_state:
        st.session_state.yes_score = 0
    if 'no_score' not in st.session_state:
        st.session_state.no_score = 0

    # Get the current image
    current_image_path = os.path.join(folder_path, images[st.session_state.current_image_index])
    image = Image.open(current_image_path)
        
    st.write('Grid Image :')

    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    # Display the same image in all 3 columns with unique 'yes' and 'no' buttons
    for i, col in enumerate([col1, col2, col3], start=1):
        with col:
            col.image(image, caption=f"Image {st.session_state.current_image_index + 1}", use_column_width=True)
            if col.button("Yes", key=f"Yes{i}"):
                st.session_state.yes_score += 1
                col.write("User said YES.")
            if col.button("No", key=f"No{i}"):
                st.session_state.no_score += 1
                col.write("User said NO.")

    total_votes = st.session_state.yes_score + st.session_state.no_score
    if total_votes > 0:
        avg_yes = st.session_state.yes_score / total_votes * 100
        avg_no = st.session_state.no_score / total_votes * 100
    else:
        avg_yes = avg_no = 0

    st.write(f"Yes Score: {st.session_state.yes_score} ({avg_yes:.2f}%)")
    st.write(f"No Score: {st.session_state.no_score} ({avg_no:.2f}%)")

    # Create a DataFrame for the bar chart
    df = pd.DataFrame({
        'Average': [avg_yes, avg_no]
    }, index=['Yes', 'No'])

    # Display the bar chart
    st.bar_chart(df)

    if st.button("Next"):
        # Increment image index and rerun the script
        if st.session_state.current_image_index < len(images) - 1:
            st.session_state.current_image_index += 1
        else:
            st.write("No more images to display.")

if __name__ == "__main__":
    main()
