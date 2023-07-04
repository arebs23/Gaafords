import streamlit as st
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import io

def main():
    st.title('Image Processing App')

    # Get the list of subdirectories
    parent_folder_path = 'ecai_effect'
    subdirectories = [d for d in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, d))]

    # Initialize session state
    if 'current_subdir_index' not in st.session_state:
        st.session_state.current_subdir_index = 0
    if 'yes_score' not in st.session_state:
        st.session_state.yes_score = [0] * len(subdirectories)
    if 'no_score' not in st.session_state:
        st.session_state.no_score = [0] * len(subdirectories)

    # Get the current subdirectory
    current_subdir_path = os.path.join(parent_folder_path, subdirectories[st.session_state.current_subdir_index])
    images = [os.path.join(current_subdir_path, f) for f in os.listdir(current_subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    original_images = [img for img in images if 'original' in img.lower()]
    other_images = [img for img in images if 'original' not in img.lower()]
    images = original_images + other_images
    current_images = [Image.open(image_path) for image_path in images]

    st.write('Grid Images :')

    # Create a column layout based on the number of current images
    cols = st.columns(len(current_images))

    # Display each image in a column with unique 'yes' and 'no' buttons
    for i, col in enumerate(cols):
        with col:
            # Get the image's filename (without the extension) to use as the caption
            filename_without_ext = os.path.splitext(os.path.basename(images[i]))[0]
            col.image(current_images[i], caption=f"Image: {filename_without_ext}", use_column_width=True)
            if 'original' not in filename_without_ext:
                if col.button("Yes", key=f"Yes{st.session_state.current_subdir_index}_{i}"):
                    st.session_state.yes_score[st.session_state.current_subdir_index] += 1
                    col.write("User said YES.")
                if col.button("No", key=f"No{st.session_state.current_subdir_index}_{i}"):
                    st.session_state.no_score[st.session_state.current_subdir_index] += 1
                    col.write("User said NO.")

    # Save user responses to a CSV file
    df_response = pd.DataFrame({
        'subdirectory': subdirectories,
        'yes_score': st.session_state.yes_score,
        'no_score': st.session_state.no_score
    })
    df_response.to_csv('response_scores.csv', index=False)

    # Load existing responses, if any
    if os.path.isfile('response_scores.csv'):
        df_response = pd.read_csv('response_scores.csv')

        total_yes_score = df_response['yes_score'].sum()
        total_no_score = df_response['no_score'].sum()

        total_votes = total_yes_score + total_no_score
        if total_votes > 0:
            avg_yes = total_yes_score / total_votes * 100
            avg_no = total_no_score / total_votes * 100
        else:
            avg_yes = avg_no = 0

        st.write(f"Yes Score: {total_yes_score} ({avg_yes:.2f}%)")
        st.write(f"No Score: {total_no_score} ({avg_no:.2f}%)")

        # Create a DataFrame for the bar chart
        df = pd.DataFrame({
            'Average': [avg_yes, avg_no]
        }, index=['Yes', 'No'])

        # Generate the bar chart using matplotlib
        st.bar_chart(df)
        # fig, ax = plt.subplots()
        # df.plot(kind='bar', ax=ax)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Use st.download_button to download the plot
        # st.download_button(
        #     label="Download bar chart",
        #     data=buf,
        #     file_name='bar_chart.png',
        #     mime='image/png'
        # )

        # # Display the bar chart
        # st.pyplot(fig)

    if st.button("Next"):
        # Increment subdirectory index and rerun the script
        if st.session_state.current_subdir_index < len(subdirectories) - 1:
            st.session_state.current_subdir_index += 1
            st.experimental_rerun()
        else:
            st.write("No more subdirectories to display.")

if __name__ == "__main__":
    main()
